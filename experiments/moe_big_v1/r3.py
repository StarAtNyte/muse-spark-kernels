
# ---- round 3: bounding ablations (outputs wrong except base): dup jobs' weights from L2, all weights from L2,
# no final pass, no x gather copies; plus a stamps-only timeline
S = load_sub("/w/run/sub.py", "sub")
a = patch_abl(patch_common(S.CUDA_SRC))
S.ALT_SRCS["ab"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
ARMS = {"base": B.MAIN, "dupL2": B.MAIN + ("-DOKC_ABLW=1", "@ab"), "allL2": B.MAIN + ("-DOKC_ABLW=2", "@ab"),
        "nofin": B.MAIN + ("-DOKC_NOFIN=1", "@ab"), "nogath": B.MAIN + ("-DOKC_NOGATH=1", "@ab")}
DBG = B.MAIN + ("-DOKC_DEBUG=1", "@ab")
pool = concurrent.futures.ThreadPoolExecutor(max_workers=6)
futs = [pool.submit(S._ext, d) for d in list(ARMS.values())[1:] + [DBG]]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f in futs: f.result()
print("BUILT", el(), flush=True)
B.setup(st); print("EMPTY_US", round(B.empty, 2), flush=True)
SEEDS = {128: [5041754901133868307, 1281, 1282, 1284, 1286], 256: [5041887071557145533, 2561, 2563, 2564, 2566]}
for bs in (128, 256):
    for seed in SEEDS[bs][:3]:
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        res = B.timeline(DBG, d, reps=8)
        G = B.G; summ = {}
        for k, nm in ((1, "routed"), (7, "gathered"), (2, "A"), (3, "B"), (5, "final")):
            vals = sorted(float(statistics.median(float(r[0][c, k]) for r in res)) for c in range(G))
            summ[nm] = [round(vals[0], 1), round(vals[G // 2], 1), round(vals[-1], 1)]
        print("TL", bs, seed, json.dumps(dict(D=rs["D"], dup=rs["dup"], **summ)), flush=True)
rows = []
for bs in (128, 256):
    for seed in SEEDS[bs]:
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        row = dict(bs=bs, seed=seed, D=rs["D"], dup=rs["dup"], **B.paired(ARMS, d, reps=12))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), {n: round(statistics.mean(r[n] for r in rr), 1) for n in ARMS}, flush=True)
print("DONE", el(), flush=True)
