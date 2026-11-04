
# ---- round 4: wgs_seg pipeline (allL2 showed the MAIN build is consumer-bound: no-HBM bound only +5-6%)
S = load_sub("/w/run/sub.py", "sub")
a = patch_pipe(patch_abl(patch_common(S.CUDA_SRC)))
S.ALT_SRCS["pp"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
P = B.MAIN + ("@pp",)
ARMS = {"base": B.MAIN, "g4n1": P + ("-DOKC_GS4N=1",), "g4n2": P + ("-DOKC_GS4N=2",), "wd2n2": P + ("-DOKC_WD2N=2",),
        "wd3n1": P + ("-DOKC_WD2N=1", "-DOKC_WDV=3"), "wd2n8": P + ("-DOKC_WD2N=8",), "g4wd2n1": P + ("-DOKC_GS4N=1", "-DOKC_WD2N=1")}
ALL = {n + "_L2": d + ("-DOKC_ABLW=2",) for n, d in (("pp", P), ("wd2n8", ARMS["wd2n8"]))}
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = list(ARMS.values())[1:] + list(ALL.values())
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f, d in zip(futs, todo):
    try: f.result()
    except Exception as ex: print("BUILD_FAIL", d, str(ex)[-600:], flush=True)
print("BUILT", el(), flush=True)
for n, d in list(ARMS.items()) + list(ALL.items()):
    try: print("ATTR", n, S._ext(d).okc_attrs(), flush=True)
    except Exception as ex: print("ATTRFAIL", n, str(ex)[-200:])
ARMS.pop("g4n2")
B.setup(st); print("EMPTY_US", round(B.empty, 2), flush=True)
for bs, dist in ((33, "normal"), (128, "skewed"), (257, "normal"), (256, "normal"), (64, "normal")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    for n, df in ARMS.items():
        print("CHECK", bs, dist, n, B.check(df, d), flush=True)
SEEDS = {32: [321, 323], 128: [5041754901133868307, 1281, 1282, 1284], 256: [5041887071557145533, 2561, 2563, 2566]}
rows = []
for bs in (128, 256, 32):
    for seed in SEEDS[bs]:
        if el() > 820: print("SKIP_BUDGET", flush=True); break
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        arms = dict(ARMS, **ALL) if bs != 32 else ARMS
        row = dict(bs=bs, seed=seed, D=rs["D"], **B.paired(arms, d, reps=12))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (32, 128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
