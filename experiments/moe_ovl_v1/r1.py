
# ---- moe_ovl_v1 round 1: dual accumulators (DAN) / early weight-slot release (ERN) in the WGS MAIN consumer
S = load_sub("/w/run/sub.py", "sub")
a = patch_ovl(patch_abl(patch_common(S.CUDA_SRC)))
S.ALT_SRCS["ovl"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
P = B.MAIN + ("@ovl",)
ARMS = {"base": B.MAIN, "ctl": P, "da2": P + ("-DOKC_DAN=2",), "da4": P + ("-DOKC_DAN=4",), "da8": P + ("-DOKC_DAN=8",),
        "er2": P + ("-DOKC_ERN=2",), "er8": P + ("-DOKC_ERN=8",), "da2er2": P + ("-DOKC_DAN=2", "-DOKC_ERN=2")}
ALL = {"ctl_L2": P + ("-DOKC_ABLW=2",), "da2_L2": P + ("-DOKC_DAN=2", "-DOKC_ABLW=2")}
pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
todo = list(ARMS.values())[1:] + list(ALL.values())
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
bad = set()
for f, d in zip(futs, todo):
    try: f.result()
    except Exception as ex: print("BUILD_FAIL", d, str(ex)[-1500:], flush=True); bad.add(d)
ARMS = {n: d for n, d in ARMS.items() if d not in bad}; ALL = {n: d for n, d in ALL.items() if d not in bad}
for n, d in list(ARMS.items()) + list(ALL.items()): print("ATTR", n, S._ext(d).okc_attrs(), flush=True)
B.setup(st)
for bs, dist in ((33, "normal"), (128, "skewed"), (257, "normal"), (256, "normal"), (64, "normal")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    for n, df in ARMS.items():
        ok = B.check(df, d)
        if n == "base" or not ok[0] or bs == 257: print("CHECK", bs, dist, n, ok, flush=True)
SEEDS = {32: [321, 323], 128: [5041754901133868307, 1281, 1282, 1284], 256: [5041887071557145533, 2561, 2563, 2566]}
rows = []
for bs in (128, 256, 32):
    for seed in SEEDS[bs]:
        if el() > 800: print("SKIP_BUDGET", flush=True); break
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        arms = dict(ARMS, **ALL) if bs != 32 else ARMS
        row = dict(bs=bs, seed=seed, D=rs["D"], **B.paired(arms, d, reps=12))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (32, 128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
