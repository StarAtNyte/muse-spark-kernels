"""moe_fresh_v1 round 2: B2L LSU-mode (B2LM=1) knob sweep (L2 prefetch of the CTA's later phase-B weight units while phase A drains) on the
skinny builds bs1 (TINY), bs8 (TINY2 t2_wsi), bs16 (MID). Base = the production build (B2L=0, same code as the parent).
Reference checks on every arm; cold-L2 CUDA-graph replays, rotating order."""
import importlib.util, json, math, statistics, sys, time, concurrent.futures
import torch
T0 = time.time()
def el(): return round(time.time() - T0, 1)
sys.path.insert(0, "/w/kernels/muse_moe_w4a16")
import reference as R
sp = importlib.util.spec_from_file_location("cand", "/w/run/cand.py"); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w); print("PREPARED", el(), flush=True)
sfxs = tuple(st.get("sfxs", st["sfx"]))
BASE = {1: tuple(S.TINY_DEFINES) + sfxs, 8: tuple(st["ps"][8]), 16: tuple(S.MID_DEFINES) + sfxs}
def K(b, t): return ("-DOKC_B2L=%d" % b, "-DOKC_B2LT=%d" % t, "-DOKC_B2LM=1")
ARMS = {1: {"l2t0": K(2, 0), "l4t0": K(4, 0), "l4t3": K(4, 3)},
        8: {"l2t0": K(2, 0), "l4t0": K(4, 0), "l8t0": K(8, 0), "l4t3": K(4, 3)},
        16: {"l2t0": K(2, 0), "l4t0": K(4, 0), "l4t3": K(4, 3)}}
todo = [BASE[bs] + v for bs in ARMS for v in ARMS[bs].values()]
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    futs = [pool.submit(S._ext, d) for d in todo]
    for f, d in zip(futs, todo):
        try: f.result()
        except Exception as ex: print("BUILD_FAIL", d, str(ex)[-1500:], flush=True)
print("BUILT", el(), flush=True)
G = st["grid"]
flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
def launcher(defs, d, out):
    ext = S._ext(defs)
    x = d["hidden_states"].contiguous(); ids = d["topk_ids"].to(torch.int32).contiguous(); tw = d["topk_weights"].float().contiguous()
    return lambda: ext.okc_run(st["w"], st["scratch"], x, ids, tw, out, 0, 32, G)
def capture(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.sum(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); e1.synchronize(); return e0.elapsed_time(e1) * 1000
fails = 0
def ok(tag, d, o):
    global fails
    r = R.check_implementation(d, o, w)
    if not r[0]: fails += 1; print("CHECK_FAIL", tag, r[1], flush=True)
# correctness of every arm (graph replay output), incl. skewed and test-like sizes routed to the same build
for bs, extra in ((1, ()), (8, ()), (16, ())):
    for n, v in ARMS[bs].items():
        for seed, dist in ((5, "normal"), (6, "skewed"), (7, "normal")):
            d = R.generate_input(bs, "ep8", seed, dist=dist); out = torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda")
            g = capture(launcher(BASE[bs] + v, d, out)); g.replay(); torch.cuda.synchronize(); ok((bs, n, seed, dist), d, out)
print("CHECKS_DONE fails", fails, el(), flush=True)
rows = []
for bs in (1, 8, 16):
    for k in range(6):
        if el() > 270: print("SKIP_BUDGET", flush=True); break
        d = R.generate_input(bs, "ep8", 419 + 7919 * k)
        arms = {"base": BASE[bs], **{n: BASE[bs] + v for n, v in ARMS[bs].items()}}
        gs = {n: capture(launcher(df, d, torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda"))) for n, df in arms.items()}
        names = list(arms); T = {n: [] for n in names}
        for r in range(16):
            order = names[r % len(names):] + names[:r % len(names)]
            if r & 1: order = order[::-1]
            for n in order: T[n].append(t1(gs[n]))
        row = dict(bs=bs, seed=k, **{n: round(statistics.median(v), 2) for n, v in T.items()},
                   **{f"r_{n}": round(statistics.median(sorted(b / c for b, c in zip(T["base"], T[n]))), 4) for n in names[1:]})
        rows.append(row); print("ROW", json.dumps(row), flush=True)
        del gs
for bs in (1, 8, 16):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE fails", fails, el(), flush=True)
