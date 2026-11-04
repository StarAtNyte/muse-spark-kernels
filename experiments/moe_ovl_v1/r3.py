
# ---- moe_ovl_v1 round 3: why FIN1 loses at bs128/256: epilogue-only (1) / final-only (2, wrong output) / both (3), timelines
S = load_sub("/w/run/sub.py", "sub")
a = patch_fin1(patch_fin(patch_abl(patch_common(S.CUDA_SRC))))
S.ALT_SRCS["f1"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
P = B.MAIN + ("@f1",)
Y = P + ("-DOKC_YBFX=1",)
ARMS = {"base": B.MAIN, "ybf": Y, "epi": Y + ("-DOKC_FIN1=1",), "fino": Y + ("-DOKC_FIN1=2",), "fin1": Y + ("-DOKC_FIN1=3",)}
DB = {"ybf": Y + ("-DOKC_DEBUG=1",), "fin1": Y + ("-DOKC_FIN1=3", "-DOKC_DEBUG=1")}
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = list(ARMS.values())[1:] + list(DB.values())
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f, d in zip(futs, todo): f.result()
B.setup(st)
for bs, dist in ((128, "skewed"), (257, "normal")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    print("CHECK", bs, dist, "fin1", B.check(ARMS["fin1"], d), flush=True)
for bs, seed in ((256, 2563), (256, 2561), (128, 1281)):
    d = R.generate_input(bs, "ep8", seed)
    for n, df in DB.items():
        res = B.timeline(df, d, reps=8)
        out = []
        for k in (1, 2, 3, 4, 5):
            v = torch.stack([r[0][:, k] for r in res]).median(0).values
            out.append(f"s{k} {v.min().item():.1f}/{v.median().item():.1f}/{v.max().item():.1f}")
        print("TL", bs, seed, n, " | ".join(out), flush=True)
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
