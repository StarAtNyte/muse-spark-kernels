
# ---- moe_ovl_v1 round 2: single-pass final with epilogue sum-of-squares partials (FIN1), + timeline of the base
S = load_sub("/w/run/sub.py", "sub")
a = patch_fin1(patch_fin(patch_abl(patch_common(S.CUDA_SRC))))
S.ALT_SRCS["f1"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
P = B.MAIN + ("@f1",)
ARMS = {"base": B.MAIN, "ctl": P, "ybf": P + ("-DOKC_YBFX=1",), "fin1": P + ("-DOKC_YBFX=1", "-DOKC_FIN1=1"), "nofin": P + ("-DOKC_NOFIN=1",)}
DBGD = P + ("-DOKC_DEBUG=1",)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = list(ARMS.values())[1:] + [DBGD]
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
bad = set()
for f, d in zip(futs, todo):
    try: f.result()
    except Exception as ex: print("BUILD_FAIL", d, str(ex)[-2500:], flush=True); bad.add(d)
ARMS = {n: d for n, d in ARMS.items() if d not in bad}
for n, d in ARMS.items(): print("ATTR", n, S._ext(d).okc_attrs(), flush=True)
B.setup(st)
for bs, dist in ((33, "normal"), (128, "skewed"), (257, "normal"), (256, "normal"), (17, "normal"), (64, "normal")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    for n, df in ARMS.items():
        if n == "nofin": continue
        print("CHECK", bs, dist, n, B.check(df, d), flush=True)
if DBGD not in bad:
    for bs, seed in ((256, 2563), (128, 1281), (32, 321)):
        d = R.generate_input(bs, "ep8", seed); B.st = B.st if hasattr(B, "st") else None
        res = B.timeline(DBGD, d, reps=6)
        for k in range(8):
            v = torch.stack([r[0][:, k] for r in res]).median(0).values
            v = v[v > -1e5]
            print("TL", bs, k, "min/med/max us", round(v.min().item(), 2), round(v.median().item(), 2), round(v.max().item(), 2), flush=True)
SEEDS = {32: [321, 323], 128: [5041754901133868307, 1281, 1282, 1284], 256: [5041887071557145533, 2561, 2563, 2566]}
rows = []
for bs in (128, 256, 32):
    for seed in SEEDS[bs]:
        if el() > 800: print("SKIP_BUDGET", flush=True); break
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        row = dict(bs=bs, seed=seed, D=rs["D"], **B.paired(ARMS, d, reps=14))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (32, 128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
