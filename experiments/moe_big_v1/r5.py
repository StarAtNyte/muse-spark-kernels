
# ---- round 5: final-pass / epilogue traffic: BF16 ybuf, register-held final rows (nofin bound was 2-3.3%)
S = load_sub("/w/run/sub.py", "sub")
a = patch_fin(patch_abl(patch_common(S.CUDA_SRC)))
S.ALT_SRCS["pf"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
P = B.MAIN + ("@pf",)
ARMS = {"base": B.MAIN, "ybf": P + ("-DOKC_YBFX=1",), "hold": P + ("-DOKC_FHOLD=1",), "ctl": P, "nofin": P + ("-DOKC_NOFIN=1",)}
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = list(ARMS.values())[1:]
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f, d in zip(futs, todo):
    try: f.result()
    except Exception as ex: print("BUILD_FAIL", d, str(ex)[-800:], flush=True)
print("BUILT", el(), flush=True)
for n, d in ARMS.items(): print("ATTR", n, S._ext(d).okc_attrs(), flush=True)
B.setup(st); print("EMPTY_US", round(B.empty, 2), flush=True)
for bs, dist in ((33, "normal"), (128, "skewed"), (257, "normal"), (256, "normal"), (64, "normal")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    for n, df in ARMS.items():
        if n != "nofin": print("CHECK", bs, dist, n, B.check(df, d), flush=True)
SEEDS = {32: [321, 323], 128: [5041754901133868307, 1281, 1282, 1284, 1286], 256: [5041887071557145533, 2561, 2563, 2564, 2566]}
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
