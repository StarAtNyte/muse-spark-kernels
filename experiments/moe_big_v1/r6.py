
# ---- round 6: decode-cost bounds in the WGS MAIN consumer (wrong outputs except base/ctl)
S = load_sub("/w/run/sub.py", "sub")
a = patch_dq(patch_pipe(patch_abl(patch_common(S.CUDA_SRC))))
S.ALT_SRCS["dq"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
P = B.MAIN + ("@dq",)
ARMS = {"base": B.MAIN, "ctl": P, "nodq": P + ("-DOKC_NODQ=1",), "noscm": P + ("-DOKC_NOSCM=1",), "both": P + ("-DOKC_NODQ=1", "-DOKC_NOSCM=1"),
        "both_L2": P + ("-DOKC_NODQ=1", "-DOKC_NOSCM=1", "-DOKC_ABLW=2")}
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = list(ARMS.values())[1:]
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f, d in zip(futs, todo):
    try: f.result()
    except Exception as ex: print("BUILD_FAIL", d, str(ex)[-800:], flush=True)
for n, d in ARMS.items(): print("ATTR", n, S._ext(d).okc_attrs(), flush=True)
B.setup(st)
SEEDS = {32: [321, 323], 128: [5041754901133868307, 1281, 1282, 1284], 256: [5041887071557145533, 2561, 2563, 2566]}
rows = []
for bs in (128, 256, 32):
    for seed in SEEDS[bs]:
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        row = dict(bs=bs, seed=seed, D=rs["D"], **B.paired(ARMS, d, reps=12))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (32, 128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
