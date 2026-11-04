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
a = rep(a, "#define OKC_DEBUG 0", "#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif\n#ifndef OKC_NOAB\n#define OKC_NOAB 0\n#endif\n#ifndef OKC_NODQ\n#define OKC_NODQ 0\n#endif", 1)
for k in ("ld_acquire(p.doneA + j) < RB_A", "ld_acquire(p.doneA + j) < UB_A", "ld_acquire(mf_da(p, S) + j * 16 + st) < 8"):
    a = rep(a, k, "(!OKC_NOAB && " + k + ")")
a = rep(a, "constexpr uint32_t M = 0x81C081C0u;\n  r0 = q & M;",
        "constexpr uint32_t M = 0x81C081C0u;\n  r0 = q & M;\n#if OKC_NODQ\n  r1 = r0 ^ 0x00400040u; r2 = r0 ^ 0x00800080u; r3 = r0 ^ 0x01000100u; return;\n#endif", 1)
S.ALT_SRCS["abl"] = a

w = R.generate_weights(R.WEIGHTS_SEED)
lo, cnt = S.LOCAL_START, S.LOCAL_EXPERTS
prmt = S.PRMT_SCALES and S._scales_prmt_exact(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
sc6 = bool(prmt) and S._sc6_ok(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
sfx = ("-DOKC_PRMTSC=1",) if prmt else ()
sfxm = sfx + (("-DOKC_SC6W=1",) if (sc6 and S.SC6_WGS) else ())
BASE = {16: tuple(S.MID_DEFINES) + sfx, 32: tuple(S._main_defines(sfx)) + sfxm}
VARS = {"noab": ("-DOKC_NOAB=1", "@abl"), "nodq": ("-DOKC_NODQ=1", "@abl"), "dbg": ("-DOKC_DEBUG=1", "@abl")}
todo = [BASE[bs] + v for bs in BASE for v in VARS.values()]
pool = concurrent.futures.ThreadPoolExecutor(max_workers=6)
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f in futs: f.result()
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

# timeline (DBG stamps only)
NAMES = {0: "start", 1: "routed", 7: "gathered/firstBwait", 6: "joined", 2: "A_done", 3: "B_done", 4: "barrier", 5: "final"}
for bs in (16, 32):
    for seed in (5725184378082808319 + bs, 77 + bs):
        d = R.generate_input(bs, "ep8", seed)
        ids = d["topk_ids"].long(); distinct = int(ids[ids < 32].unique().numel())
        out = torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda")
        dbg = torch.zeros(G * 8 + G * 32, dtype=torch.int64, device="cuda")
        fn = launcher(BASE[bs] + VARS["dbg"], d, out, list(st["scratch"])[:3] + [dbg])
        for r in range(4):
            flush.sum(); dbg.zero_(); torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
        tt = dbg[:G * 8].view(G, 8).cpu().double(); ex = dbg[G * 8:].view(G, 32).cpu().double()
        t00 = tt[:, 0][tt[:, 0] > 0].min(); res = {}
        for k, n in NAMES.items():
            col = tt[:, k][tt[:, k] > 0]
            if col.numel():
                c = ((col - t00) / 1000).sort().values
                res[n] = [round(float(c[0]), 2), round(float(c[c.numel() // 4]), 2), round(float(c[c.numel() // 2]), 2), round(float(c[-1]), 2)]
        if bs == 16:  # MID: phase-B doneA wait cycles (warp 0 lane 0), first wait, count
            res["Bwait_us_mean"] = round(float(ex[:, 20].mean()) / 1.98e3, 2); res["Bwait1_us_mean"] = round(float(ex[:, 21].mean()) / 1.98e3, 2)
            res["Bwait_us_max"] = round(float(ex[:, 20].max()) / 1.98e3, 2)
        print("TL", json.dumps(dict(bs=bs, seed=seed, distinct=distinct, **res)), flush=True)

# paired ablation timing
rows = []
for bs in (16, 32):
    for k in range(6):
        if el() > 255: print("SKIP_BUDGET", flush=True); break
        seed = 1000 * bs + 37 * k
        d = R.generate_input(bs, "ep8", seed)
        ids = d["topk_ids"].long(); distinct = int(ids[ids < 32].unique().numel())
        arms = {"base": BASE[bs], "noab": BASE[bs] + VARS["noab"], "nodq": BASE[bs] + VARS["nodq"]}
        gs = {n: capture(launcher(df, d, torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda"))) for n, df in arms.items()}
        names = list(arms); T = {n: [] for n in names}
        for r in range(16):
            order = names[r % 3:] + names[:r % 3]
            if r & 1: order = order[::-1]
            for n in order: T[n].append(t1(gs[n]) - empty)
        row = dict(bs=bs, seed=seed, distinct=distinct, **{n: round(statistics.median(v), 2) for n, v in T.items()},
                   **{f"r_{n}": round(statistics.median(sorted(b / c for b, c in zip(T["base"], T[n]))), 4) for n in names[1:]})
        rows.append(row); print("ROW", json.dumps(row), flush=True)
        del gs
for bs in (16, 32):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r["r_" + n]) for r in rr)), 4) for n in ("noab", "nodq")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), "distinct", round(statistics.mean(r["distinct"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
