"""moe_big_v1 shared driver helpers (inlined into each driver by run_modal: the driver execs this file's text).
Patches the MAIN (WGS) CUDA source: overridable cost tables, NOAB ablation, light DBG (gtimer stamps + per-CTA stage
histograms by n-tiles + doneAs spin cycles)."""
import importlib.util, json, math, re, statistics, sys, time, concurrent.futures
import torch
T0 = time.time()
def el(): return round(time.time() - T0, 1)
sys.path.insert(0, "/w/kernels/muse_moe_w4a16")
import reference as R

def load_sub(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def rep(s, old, new, n=1):
    c = s.count(old); assert c == n, (c, old[:80]); return s.replace(old, new)

def patch_common(a):
    a = rep(a, "#define OKC_DEBUG 0", "#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif\n#ifndef OKC_NOAB\n#define OKC_NOAB 0\n#endif")
    # light DBG: doneAs spin cycles (x producer) into ext slot 30, stage histogram per CTA in wgs_ranges
    old = "if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }"
    new = ("if (pb && !(OKC_WGSX & 8)) { const long long c0_ = OKC_DEBUG ? clock64() : 0; while (!OKC_NOAB && ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32);"
           " if (OKC_DEBUG && p.dbg) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 30] += clock64() - c0_; }")
    a = rep(a, old, new)
    old = "  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n}"
    new = ("  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n#if OKC_DEBUG\n  if (p.dbg) { unsigned long long* e_ = p.dbg + gridDim.x * 8 + blockIdx.x * 32;\n"
           "    for (int pb = 0; pb < 2; ++pb) { int lu = -1; for (int s = S.wgr[2 * pb]; s < S.wgr[2 * pb + 1]; ++s) { const int j = s >> (9 - pb); int nt = (S.jC[j] + 7) >> 3; nt = nt > 8 ? 8 : nt;"
           " e_[pb * 12 + nt - 1] += 1; if ((s >> 4) != lu) { e_[pb * 12 + 8] += 1; lu = s >> 4; } } }\n  }\n#endif\n}")
    a = rep(a, old, new)
    return a

def set_tables(a, A=None, B=None, U=None, UB=None):
    if A is not None: a = re.sub(r"#define OKC_WGSA [^\n]*", "#define OKC_WGSA " + ", ".join(map(str, A)), a, count=1)
    if B is not None: a = re.sub(r"#define OKC_WGSB [^\n]*", "#define OKC_WGSB " + ", ".join(map(str, B)), a, count=1)
    if U is not None: a = re.sub(r"#define OKC_WGSU 80[^\n]*", f"#define OKC_WGSU {U}", a, count=1)
    if UB is not None: a = re.sub(r"#define OKC_WGSUB 70[^\n]*", f"#define OKC_WGSUB {UB}", a, count=1)
    return a

def route_stats(d, lo=0, cnt=32, sk=8, ft=64):
    ids = d["topk_ids"].long().flatten(); ids = ids[(ids >= lo) & (ids < lo + cnt)] - lo
    c = torch.bincount(ids, minlength=cnt).tolist()
    D = sum(1 for v in c if v > 0); routes = sum(c)
    fat = [v for v in c if v > sk]; dup = sum((v + ft - 1) // ft - 1 for v in fat)
    jobs = sum((v + ft - 1) // ft for v in fat) + sum(1 for v in c if 0 < v <= sk)
    return dict(D=D, routes=routes, maxc=max(c), fat=len(fat), jobs=jobs, dup=dup, counts=sorted([v for v in c if v], reverse=True))

def floors(st):
    E = 28311552
    bw = (st["D"]) * E / 4.2e12 * 1e6
    bwdup = (st["D"] + st["dup"]) * E / 4.2e12 * 1e6
    # compute: padded tokens (to 8 per job) x 2*3*4096^2 at 600 TFLOP/s
    pad = 0
    for v in st["counts"]:
        if v <= 8: pad += 8
        else:
            q = v
            while q > 0: pad += (min(q, 64) + 7) // 8 * 8; q -= 64
    comp = st["routes"] * 2 * 3 * 4096 ** 2 / 600e12 * 1e6
    comp_pad = pad * 2 * 3 * 4096 ** 2 / 600e12 * 1e6
    return dict(bw42=round(bw, 1), bw42_dup=round(bwdup, 1), comp600=round(comp, 1), comp600_pad=round(comp_pad, 1))

class Bench:
    def __init__(self, S, w):
        self.S = S; self.w = w
        lo, cnt = S.LOCAL_START, S.LOCAL_EXPERTS
        prmt = S.PRMT_SCALES and S._scales_prmt_exact(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
        sc6 = bool(prmt) and S._sc6_ok(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
        sfx = ("-DOKC_PRMTSC=1",) if prmt else ()
        self.sfxm = sfx + (("-DOKC_SC6W=1",) if (sc6 and S.SC6_WGS) else ())
        self.MAIN = tuple(S._main_defines(sfx)) + self.sfxm
        self.flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    def setup(self, st):
        self.st = st; self.G = st["grid"]
        eg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(eg): pass
        self.empty = statistics.median([self.t1(eg) for _ in range(30)])
    def launcher(self, defs, d, out, sc=None):
        ext = self.S._ext(defs); st = self.st; G = self.G
        x = d["hidden_states"].contiguous(); ids = d["topk_ids"].to(torch.int32).contiguous(); tw = d["topk_weights"].float().contiguous()
        return lambda: ext.okc_run(st["w"], sc or st["scratch"], x, ids, tw, out, 0, 32, G)
    def capture(self, fn):
        fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        g.replay(); torch.cuda.synchronize(); return g
    def t1(self, g):
        self.flush.sum(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); return e0.elapsed_time(e1) * 1000
    def check(self, defs, d):
        out = torch.empty((d["hidden_states"].shape[0], 4096), dtype=torch.bfloat16, device="cuda")
        self.launcher(defs, d, out)(); torch.cuda.synchronize()
        return R.check_implementation(d, out, self.w)
    def paired(self, arms, d, reps=16):
        bs = d["hidden_states"].shape[0]
        gs = {n: self.capture(self.launcher(df, d, torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda"))) for n, df in arms.items()}
        names = list(arms); T = {n: [] for n in names}
        for r in range(reps):
            order = names[r % len(names):] + names[:r % len(names)]
            if r & 1: order = order[::-1]
            for n in order: T[n].append(self.t1(gs[n]) - self.empty)
        del gs
        row = {n: round(statistics.median(v), 2) for n, v in T.items()}
        for n in names[1:]: row["r_" + n] = round(statistics.median(sorted(b / c for b, c in zip(T[names[0]], T[n]))), 4)
        return row
    def timeline(self, defs, d, reps=6):
        G = self.G; bs = d["hidden_states"].shape[0]
        out = torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda")
        dbg = torch.zeros(G * 8 + G * 32, dtype=torch.int64, device="cuda")
        fn = self.launcher(defs, d, out, list(self.st["scratch"])[:3] + [dbg])
        res = []
        for r in range(reps):
            self.flush.sum(); dbg.zero_(); torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
            if r < 2: continue
            tt = dbg[:G * 8].view(G, 8).cpu().double(); ex = dbg[G * 8:].view(G, 32).cpu().double()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            res.append(((tt - t00) / 1000, ex))
        return res

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
