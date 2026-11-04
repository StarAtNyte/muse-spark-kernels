
# ---- round 2: knob arms (unit cost, table scale) + early weight-producer start, paired at bs 32/128/256
S = load_sub("/w/run/sub.py", "sub")
a = patch_early_w(patch_common(S.CUDA_SRC))
BA = [112, 135, 140, 158, 173, 185, 200, 207]; BB = [100, 104, 115, 123, 132, 144, 156, 174]
def sc(t, base, f): return [round(base + (v - base) * f) for v in t]
S.ALT_SRCS["pe"] = a
S.ALT_SRCS["u160"] = set_tables(a, U=160, UB=140)
S.ALT_SRCS["u40"] = set_tables(a, U=40, UB=35)
S.ALT_SRCS["s09"] = set_tables(a, A=sc(BA, 112, 0.9), B=sc(BB, 100, 0.9))
S.ALT_SRCS["s115"] = set_tables(a, A=sc(BA, 112, 1.15), B=sc(BB, 100, 1.15))
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
ARMS = {"base": B.MAIN, "early": B.MAIN + ("-DOKC_EARLYW=1", "@pe"), "u160": B.MAIN + ("@u160",), "u40": B.MAIN + ("@u40",),
        "s09": B.MAIN + ("@s09",), "s115": B.MAIN + ("@s115",)}
pool = concurrent.futures.ThreadPoolExecutor(max_workers=6)
futs = [pool.submit(S._ext, d) for d in list(ARMS.values())[1:]]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f in futs: f.result()
print("BUILT", el(), flush=True)
B.setup(st); print("EMPTY_US", round(B.empty, 2), flush=True)
for bs, dist in ((33, "normal"), (128, "skewed"), (257, "normal"), (256, "normal")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    for n, df in ARMS.items():
        print("CHECK", bs, dist, n, B.check(df, d), flush=True)
SEEDS = {32: [321, 322, 323], 128: [5041754901133868307, 1281, 1282, 1284, 1286], 256: [5041887071557145533, 2561, 2563, 2564, 2566]}
rows = []
for bs in (128, 256, 32):
    for seed in SEEDS[bs]:
        if el() > 820: print("SKIP_BUDGET", flush=True); break
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        row = dict(bs=bs, seed=seed, D=rs["D"], dup=rs["dup"], **B.paired(ARMS, d, reps=12))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (32, 128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
