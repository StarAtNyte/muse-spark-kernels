"""Screen rk2 configs vs vjeux rk_kernel on H200: correctness (reference.check_implementation) + paired cold-L2 graph timing."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R

spec = importlib.util.spec_from_file_location("sub", "/w/run/sub.py")
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
t0 = time.time()
ext = sub.build(sub.ORIG_SRC)
print(f"built in {time.time()-t0:.1f}s", flush=True)
nsm = torch.cuda.get_device_properties(0).multi_processor_count
print("nsm", nsm, torch.cuda.get_device_name(0), flush=True)

W = R.generate_weights(R.WEIGHTS_SEED)
w = W['router']
hi = w.to(torch.bfloat16); lo = (w - hi.float()).to(torch.bfloat16)
cat = torch.cat([hi, lo], 1).contiguous()
WP = {n: sub.pack_w(hi, lo, 64 * n) for n in (1, 2)}
WQ = sub.pack_q8(w)
_rec = sub.q8_reconstruct(w)
print("q8 weight rel err: rms", float(((_rec - w.double()).norm() / w.double().norm())), "max", float(((_rec - w.double()).abs() / w.double().abs().clamp(min=1e-30)).max()),
      "| bf16 hi+lo rms", float(((hi.double() + lo.double() - w.double()).norm() / w.double().norm())), flush=True)
zc = torch.zeros(1024, dtype=torch.int32, device='cuda')
slots = {}
def slot(key):
    if key not in slots: slots[key] = 16 + len(slots)
    return slots[key]
RKS = {1: 16, 2: 12, 3: 10, 4: 8}
def rk_S(m):
    ntt = (m + 63) // 64
    return max(1, min(RKS[ntt], nsm // (4 * ntt)))

def run_base(x):
    out = torch.empty((x.shape[0], 256), dtype=torch.float32, device='cuda')
    ext.rk(cat, x, out, rk_S(x.shape[0]), zc)
    return out
MODE = 0
def mk(cfg, S, mode=0):
    def f(x):
        out = torch.empty((x.shape[0], 256), dtype=torch.float32, device='cuda')
        if cfg >= 100:
            ext.rk3(WQ, x, out, sub.RK3_BN[cfg], S, zc, slot((cfg, x.shape[0], S)))
            return out
        ext.rk2(cat, x, out, cfg, S, zc, slot((cfg, x.shape[0], S)), mode, WP[sub.CFGS[cfg][0]] if cfg in sub.PACKED else cat)
        return out
    return f
def grid(cfg, m, S):
    nwg, bn = sub.CFGS[cfg]
    return (4 // nwg) * ((m + bn - 1) // bn) * S

SWEEP = {
    128: [(0, 12), (0, 16), (1, 16), (1, 24), (1, 32), (3, 16), (3, 24), (3, 32), (4, 32), (4, 48), (4, 64),
          (6, 32), (6, 48), (6, 64), (7, 64), (7, 96), (7, 128)],
    256: [(0, 8), (1, 8), (1, 12), (1, 16), (2, 16), (2, 24), (2, 32), (3, 8), (3, 12), (3, 16), (4, 16), (4, 24),
          (4, 32), (5, 32), (5, 48), (5, 64), (6, 16), (6, 24), (6, 32), (7, 32), (7, 48), (7, 64)],
}
ONLY = os.environ.get("ONLY")
if ONLY:
    SWEEP = {int(k): [tuple(c) for c in v] for k, v in json.loads(ONLY).items()}

# correctness: every swept config at its m, plus odd m with the same config
bad = []
for m, cands in SWEEP.items():
    for cfg, S in cands:
        if grid(cfg, m, S) > nsm or cfg >= 110: continue
        for mm in sorted({m, m - 31, m - 1}):
            if mm <= 32 or grid(cfg, mm, S) > nsm: continue
            for seed, dist in ((123, 'normal'), (9, 'outliers')):
                d = R.generate_input(mm, 'router', seed=seed, dist=dist)
                out = mk(cfg, S)(d['x']); torch.cuda.synchronize()
                ok, msg = R.check_implementation(d, out, W)
                if not ok:
                    bad.append((cfg, S, mm, dist, msg)); print("FAIL", cfg, S, mm, dist, msg, flush=True)
for m in (33, 128, 200, 256):
    d = R.generate_input(m, 'router', seed=5, dist='outliers')
    ok, msg = R.check_implementation(d, run_base(d['x']), W)
    print("base check", m, ok, msg, flush=True)
print("correctness failures:", len(bad), flush=True)

flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)
def capture(fn, x):
    fn(x); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn(x)
    g.replay(); torch.cuda.synchronize()
    return g, out
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
WARM = os.environ.get('WARM') == '1'
def el(g):
    if not WARM: flush.add_(1)
    ev[0].record(); g.replay(); ev[1].record(); ev[1].synchronize()
    return ev[0].elapsed_time(ev[1]) * 1000

REPS = int(os.environ.get("REPS", "41"))
res = []
badset = {(c, s) for c, s, *_ in bad}
for m, cands in SWEEP.items():
    d = R.generate_input(m, 'router', seed=419)
    x = d['x']
    gb, ob = capture(run_base, x)
    for cfg, S in cands:
        if grid(cfg, m, S) > nsm or (cfg, S) in badset: continue
        gc, oc = capture(mk(cfg, S), x)
        ok, msg = R.check_implementation(d, oc, W)
        bt, ct = [], []
        for i in range(REPS):
            if i & 1: c = el(gc); b = el(gb)
            else: b = el(gb); c = el(gc)
            bt.append(b); ct.append(c)
        r = {'m': m, 'cfg': cfg, 'nwg_bn': sub.CFGS[cfg], 'S': S, 'grid': grid(cfg, m, S), 'ok': ok,
             'base_us': round(statistics.median(bt), 3), 'cand_us': round(statistics.median(ct), 3),
             'cand_min': round(min(ct), 3), 'base_min': round(min(bt), 3)}
        r['delta_us'] = round(r['base_us'] - r['cand_us'], 3)
        res.append(r); print("RESULT", json.dumps(r), flush=True)
# empty-graph overhead reference
gz = torch.cuda.CUDAGraph()
with torch.cuda.graph(gz): torch.empty(1, device='cuda')
ABL = json.loads(os.environ.get("ABL", "[]"))
for m, cfg, S in ABL:
    d = R.generate_input(m, 'router', seed=419); x = d['x']
    gs = {mode: capture(mk(cfg, S, mode), x)[0] for mode in (0, 1, 2, 3, 4, 5, 7)}
    t = {mode: [] for mode in gs}
    for i in range(REPS):
        for mode, g in gs.items(): t[mode].append(el(g))
    print("ABLATION", json.dumps({'m': m, 'cfg': sub.CFGS[cfg], 'S': S, **{f'mode{k}': round(statistics.median(v), 3) for k, v in t.items()}}), flush=True)
tiny = torch.zeros(4, device='cuda')
gt = capture(lambda _: tiny.add_(1), None)[0]
print("TINY_GRAPH_US", round(statistics.median([el(gt) for _ in range(REPS)]), 3), flush=True)
print("SUMMARY", json.dumps(sorted(res, key=lambda r: (r['m'], r['cand_us']))), flush=True)
