
# ---- moe_ovl_v1 round 5: phase-A cost table flattened toward the measured per-ntt stage times (fat CTAs end A
# 20-30 us before skinny ones, deterministic; r4)
S = load_sub("/w/run/sub.py", "sub")
a = patch_common(S.CUDA_SRC)
CA = [112, 135, 140, 158, 173, 185, 200, 207]
def fa(f): return [round(112 + (c - 112) * f) for c in CA]
ARMS_T = {"ctl": 1.0, "a08": 0.8, "a07": 0.7, "a06": 0.6, "a05": 0.5}
P = None
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
ARMS = {"base": B.MAIN}
for n, f in ARMS_T.items():
    S.ALT_SRCS[n] = set_tables(a, A=fa(f)); ARMS[n] = B.MAIN + ("@" + n,)
    print("TABLE", n, fa(f), flush=True)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = list(ARMS.values())[1:]
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f, d in zip(futs, todo): f.result()
B.setup(st)
for bs, dist in ((33, "normal"), (128, "skewed"), (257, "normal"), (64, "normal")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    for n, df in ARMS.items(): print("CHECK", bs, dist, n, B.check(df, d), flush=True)
SEEDS = {32: [321, 323, 325], 128: [5041754901133868307, 1281, 1282, 1284, 1286], 256: [5041887071557145533, 2561, 2563, 2566, 2568]}
rows = []
for bs in (128, 256, 32):
    for seed in SEEDS[bs]:
        if el() > 820: print("SKIP_BUDGET", flush=True); break
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        row = dict(bs=bs, seed=seed, D=rs["D"], **B.paired(ARMS, d, reps=14))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (32, 128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
