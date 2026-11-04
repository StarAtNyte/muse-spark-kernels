
# ---- round 1: routing stats + floors, light timeline with stage histograms (cost-curve fit), paired base/noab/flat/steep
S = load_sub("/w/run/sub.py", "sub")
a = patch_common(S.CUDA_SRC)
BA = [112, 135, 140, 158, 173, 185, 200, 207]; BB = [100, 104, 115, 123, 132, 144, 156, 174]
S.ALT_SRCS["p0"] = a
S.ALT_SRCS["flat"] = set_tables(a, A=[round(112 + (v - 112) * 0.6) for v in BA], B=[round(100 + (v - 100) * 0.6) for v in BB])
S.ALT_SRCS["steep"] = set_tables(a, A=[round(112 + (v - 112) * 1.4) for v in BA], B=[round(100 + (v - 100) * 1.4) for v in BB])
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
ARMS = {"base": B.MAIN, "noab": B.MAIN + ("-DOKC_NOAB=1", "@p0"), "flat": B.MAIN + ("@flat",), "steep": B.MAIN + ("@steep",)}
DBG = B.MAIN + ("-DOKC_DEBUG=1", "@p0")
pool = concurrent.futures.ThreadPoolExecutor(max_workers=6)
futs = [pool.submit(S._ext, d) for d in list(ARMS.values())[1:] + [DBG]]
# routing statistics (judge generate_input, default dist)
for bs in (128, 256):
    agg = []
    for seed in range(200):
        d = R.generate_input(bs, "ep8", 10007 * seed + bs)
        s = route_stats(d); s.update(floors(s)); agg.append(s)
    def m(k): return round(statistics.mean(x[k] for x in agg), 2)
    print("RSTAT", bs, json.dumps({k: m(k) for k in ("D", "routes", "maxc", "fat", "jobs", "dup", "bw42", "bw42_dup", "comp600", "comp600_pad")}),
          "dup_dist", json.dumps({k: sum(1 for x in agg if x["dup"] == k) for k in range(8)}), flush=True)
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f in futs: f.result()
print("BUILT", el(), flush=True)
B.setup(st); print("EMPTY_US", round(B.empty, 2), flush=True)
SEEDS = {128: [5041754901133868307, 1281, 1282, 1283, 1284, 1285], 256: [5041887071557145533, 2561, 2562, 2563, 2564, 2565]}
for bs in (128, 256):
    d = R.generate_input(bs, "ep8", 4242 + bs)
    for n, df in list(ARMS.items()) + [("dbg", DBG)]:
        print("CHECK", bs, n, B.check(df, d), flush=True)
# timeline + histograms
fitrows = {0: [], 1: []}
for bs in (128, 256):
    for seed in SEEDS[bs][:4]:
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        res = B.timeline(DBG, d)
        tt, ex = res[-1]
        G = B.G
        summ = {}
        for k, nm in ((1, "routed"), (7, "gathered"), (6, "joined"), (2, "A"), (3, "B"), (4, "barrier"), (5, "final")):
            vals = sorted(float(statistics.median(float(r[0][c, k]) for r in res)) for c in range(G))
            summ[nm] = [round(vals[0], 1), round(vals[G // 2], 1), round(vals[-1], 1)]
        da = [float(statistics.median(float(r[1][c, 30]) for r in res)) / 1.98e3 for c in range(G)]
        summ["doneAs_spin_us_mean"] = round(statistics.mean(da), 2)
        print("TL", bs, seed, json.dumps(dict(D=rs["D"], dup=rs["dup"], jobs=rs["jobs"], maxc=rs["maxc"], **summ)), flush=True)
        for c in range(G):
            tA = statistics.median(float(r[0][c, 2] - r[0][c, 6]) for r in res)
            tB = statistics.median(float(r[0][c, 3] - r[0][c, 2]) for r in res)
            hA = [float(ex[c, i]) for i in range(9)]; hB = [float(ex[c, 12 + i]) for i in range(9)]
            fitrows[0].append((hA, tA)); fitrows[1].append((hB, tB))
        cr = [dict(c=c, A=round(float(tt[c, 2]), 1), B=round(float(tt[c, 3]), 1), hA=[int(ex[c, i]) for i in range(9)], hB=[int(ex[c, 12 + i]) for i in range(9)], da=round(da[c], 1)) for c in range(0, G, 11)]
        print("CTAS", bs, seed, json.dumps(cr), flush=True)
# least-squares cost curves (us per stage by n-tiles; per-unit cost) on the per-CTA phase times
for pb in (0, 1):
    X = torch.tensor([r[0] for r in fitrows[pb]], dtype=torch.float64); y = torch.tensor([r[1] for r in fitrows[pb]], dtype=torch.float64)
    keep = X[:, :8].sum(0) > 0
    cols = [i for i in range(9) if i == 8 or bool(keep[i])]
    sol = torch.linalg.lstsq(X[:, cols], y[:, None]).solution.flatten()
    resid = (X[:, cols] @ sol - y)
    print("FIT", "AB"[pb], json.dumps({("unit" if i == 8 else f"n{i + 1}"): round(float(v), 4) for i, v in zip(cols, sol)}),
          "rms_us", round(float(resid.pow(2).mean().sqrt()), 2), "y_mean", round(float(y.mean()), 1), "y_spread", round(float(y.max() - y.min()), 1), flush=True)
# paired timing
rows = []
for bs in (128, 256):
    for seed in SEEDS[bs]:
        if el() > 800: print("SKIP_BUDGET", flush=True); break
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        row = dict(bs=bs, seed=seed, D=rs["D"], dup=rs["dup"], **B.paired(ARMS, d, reps=14))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
