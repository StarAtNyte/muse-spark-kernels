
# ---- moe_ovl_v1 round 6: phase-B boundaries offset by predicted phase-A lateness (BOFF)
S = load_sub("/w/run/sub.py", "sub")
a = patch_boff(patch_common(S.CUDA_SRC))
S.ALT_SRCS["bo"] = a
S.ALT_SRCS["bof"] = patch_boff(patch_common(S.CUDA_SRC), T="112, 126, 129, 140, 149, 156, 165, 169")
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
P = B.MAIN + ("@bo",)
ARMS = {"base": B.MAIN, "ctl": P, "l05": P + ("-DOKC_BOFF=50",), "l10": P + ("-DOKC_BOFF=100",), "l15": P + ("-DOKC_BOFF=150",),
        "l10f": B.MAIN + ("@bof", "-DOKC_BOFF=100")}
DB = {"ctl": P + ("-DOKC_DEBUG=1",), "l10": P + ("-DOKC_BOFF=100", "-DOKC_DEBUG=1")}
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = list(ARMS.values())[1:] + list(DB.values())
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
bad = set()
for f, d in zip(futs, todo):
    try: f.result()
    except Exception as ex: print("BUILD_FAIL", d, str(ex)[-2500:], flush=True); bad.add(d)
ARMS = {n: d for n, d in ARMS.items() if d not in bad}
B.setup(st)
for bs, dist in ((33, "normal"), (128, "skewed"), (257, "normal"), (64, "normal"), (17, "normal"), (200, "skewed")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    for n, df in ARMS.items(): print("CHECK", bs, dist, n, B.check(df, d), flush=True)
for bs, seed in ((256, 2563), (128, 1281)):
    d = R.generate_input(bs, "ep8", seed)
    for n, df in DB.items():
        if df in bad: continue
        res = B.timeline(df, d, reps=8)
        out = []
        for k in (2, 3, 5):
            v = torch.stack([r[0][:, k] for r in res]).median(0).values
            out.append(f"s{k} {v.min().item():.1f}/{v.median().item():.1f}/{v.max().item():.1f}")
        print("TL", bs, seed, n, " | ".join(out), flush=True)
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
