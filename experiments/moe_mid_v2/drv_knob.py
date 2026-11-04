"""moe_mid_v2 round 1: bubble accounting at bs16 (MID build) and bs32 (WGS MAIN build) on the judge image.
Ablations (timing upper bounds only, outputs are wrong): NOAB = no phase A->B readiness waits, NODQ = 1-op decode.
Plus a DBG-only (globaltimer per CTA) timeline of the unchanged builds. Cold L2 via flush.sum(), CUDA graphs,
rotating order per rep, several judge-like seeds per shape."""
import importlib.util, json, math, statistics, sys, time, concurrent.futures
import torch
T0 = time.time()
def el(): return round(time.time() - T0, 1)
sys.path.insert(0, "/w/kernels/muse_moe_w4a16")
import reference as R
sp = importlib.util.spec_from_file_location("sub", "/w/run/sub.py"); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)

a = S.CUDA_SRC
def rep(s, old, new, n=None):
    c = s.count(old); assert c >= 1 and (n is None or c == n), (c, old[:60]); return s.replace(old, new)
for k, v in (("OKC_WGSUB", "70"), ("OKC_WGSU", "80"), ("OKC_WGSDFIN", "2")):
    a = rep(a, f"#define {k} {v}", f"#ifndef {k}\n#define {k} {v}\n#endif\n", 1)
S.ALT_SRCS["knob"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
lo, cnt = S.LOCAL_START, S.LOCAL_EXPERTS
prmt = S.PRMT_SCALES and S._scales_prmt_exact(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
sc6 = bool(prmt) and S._sc6_ok(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
sfx = ("-DOKC_PRMTSC=1",) if prmt else ()
sfxm = sfx + (("-DOKC_SC6W=1",) if (sc6 and S.SC6_WGS) else ())
BASE = {16: tuple(S.MID_DEFINES) + sfx, 32: tuple(S._main_defines(sfx)) + sfxm}
ARMS = {32: {"ub40": ("-DOKC_WGSUB=40", "@knob"), "ub110": ("-DOKC_WGSUB=110", "@knob"), "u50": ("-DOKC_WGSU=50", "@knob"),
              "u110": ("-DOKC_WGSU=110", "@knob"), "df0": ("-DOKC_WGSDFIN=0", "@knob"), "df3": ("-DOKC_WGSDFIN=3", "@knob")},
        16: {"blk1": ("-DOKC_WSBLK=1",), "wsbl0": ("-DOKC_WSBL=0",), "ns4": ("-DOKC_NS=4",)}}
def armdefs(bs, v):
    keys = {x.split("=")[0] for x in v if x.startswith("-D")}
    return tuple(d for d in BASE[bs] if d.split("=")[0] not in keys) + v
todo = [armdefs(bs, v) for bs in ARMS for v in ARMS[bs].values()]
pool = concurrent.futures.ThreadPoolExecutor(max_workers=9)
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f, d in zip(futs, todo):
    try: f.result()
    except Exception as ex: print("BUILD_FAIL", d, str(ex)[-400:], flush=True)
print("BUILT", el(), flush=True)
G = st["grid"]
flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

def launcher(defs, d, out, sc=None):
    ext = S._ext(defs)
    x = d["hidden_states"].contiguous(); ids = d["topk_ids"].to(torch.int32).contiguous(); tw = d["topk_weights"].float().contiguous()
    return lambda: ext.okc_run(st["w"], sc or st["scratch"], x, ids, tw, out, 0, 32, G)
def capture(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.sum(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); e1.synchronize(); return e0.elapsed_time(e1) * 1000
eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg): pass
empty = statistics.median([t1(eg) for _ in range(30)]); print("EMPTY_US", round(empty, 2), flush=True)

# base correctness (production dispatch) on the two shapes
for bs in (16, 32):
    d = R.generate_input(bs, "ep8", 99 + bs); o = S.kernel_fn(d, st); torch.cuda.synchronize()
    print("CHECK", bs, R.check_implementation(d, o, w), flush=True)

# paired ablation timing
rows = []
for bs in (16, 32):
    for k in range(6):
        if el() > 255: print("SKIP_BUDGET", flush=True); break
        seed = 1000 * bs + 37 * k
        d = R.generate_input(bs, "ep8", seed)
        ids = d["topk_ids"].long(); distinct = int(ids[ids < 32].unique().numel())
        arms = {"base": BASE[bs], **{n: armdefs(bs, v) for n, v in ARMS[bs].items() if armdefs(bs, v) in S._EXTS or ("-DOKC_DQ7=1" not in armdefs(bs, v) and armdefs(bs, v) + ("-DOKC_DQ7=1",) in S._EXTS)}}
        gs = {n: capture(launcher(df, d, torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda"))) for n, df in arms.items()}
        names = list(arms); T = {n: [] for n in names}
        for r in range(14):
            order = names[r % len(names):] + names[:r % len(names)]
            if r & 1: order = order[::-1]
            for n in order: T[n].append(t1(gs[n]) - empty)
        row = dict(bs=bs, seed=seed, distinct=distinct, **{n: round(statistics.median(v), 2) for n, v in T.items()},
                   **{f"r_{n}": round(statistics.median(sorted(b / c for b, c in zip(T["base"], T[n]))), 4) for n in names[1:]})
        rows.append(row); print("ROW", json.dumps(row), flush=True)
        del gs
for bs in (16, 32):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), "distinct", round(statistics.mean(r["distinct"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
