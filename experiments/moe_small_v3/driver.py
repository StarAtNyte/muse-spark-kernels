"""moe_small_v3 driver: correctness (bs 1..8, normal + skewed, repeated launches), globaltimer timeline (SM_DBG build),
paired cold-L2 CUDA-graph timing of the parent dispatch vs the new small kernel (+ variants), on the judge image.
DRV_ARGS: json {"variants": {name: [defines]}, "shapes": [...], "seeds": n, "reps": n, "tl": bool}"""
import importlib.util, json, math, os, statistics, sys, time, concurrent.futures
import torch
T0 = time.time()
def el(): return round(time.time() - T0, 1)
A = json.loads(os.environ.get("DRV_ARGS") or "{}")
BUDGET = float(A.get("budget", 380))

class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sglang", "sgl_kernel"):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        from unittest.mock import MagicMock
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec
        return m
    def exec_module(self, module): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, "/w/kernels/muse_moe_w4a16")
import reference as R
sp = importlib.util.spec_from_file_location("sub", "/w/run/sub.py"); C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
w = R.generate_weights(R.WEIGHTS_SEED)
VARS = {k: tuple(v) for k, v in A.get("variants", {}).items()}
def vmaxm(dd):
    for x in dd:
        if x.startswith("-DSM_SROWS="): return min(8, int(x.split("=")[1]))
    return 8
DBGD = tuple(C.SMALL_DEFINES) + ("-DSM_DBG=1",)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=6)
futs = {n: pool.submit(C._small_ext, d) for n, d in list(VARS.items()) + [("dbg", DBGD)]}
st = C.prepare(w); print("PREPARED", el(), "v3" in st and st["v3"] is not None, flush=True)
for n, f in futs.items():
    try: f.result(); print("BUILT", n, C._small_ext(VARS.get(n, DBGD)).sm3_attrs(), el(), flush=True)
    except Exception as ex: print("BUILD_FAIL", n, str(ex)[-3000:], flush=True); VARS.pop(n, None)
G = st["grid"]
dev = "cuda"
VSCR = {n: C._small_state(dev) for n in VARS}   # each variant its own scratch (barrier protocols may differ)

def distinct(d):
    ids = d["topk_ids"].long(); return int(ids[ids < 32].unique().numel())

# ---------------- correctness
allok = True
checks = [(1, s, "normal") for s in range(6)] + [(b, 10 + b, "normal") for b in (2, 3, 4, 5, 6, 7, 8)] + \
         [(b, 30 + b, "skewed") for b in (1, 2, 4, 8)] + [(8, 40 + s, "normal") for s in range(4)] + [(4, 50 + s, "skewed") for s in range(3)]
for bs, seed, dist in checks:
    d = R.generate_input(bs, "ep8", seed, dist=dist)
    o1 = C.kernel_fn(d, st).clone(); o2 = C.kernel_fn(d, st); op = C._v3_parent_kernel(d, st); torch.cuda.synchronize()
    ok, msg = R.check_implementation(d, o1, w); ok2, _ = R.check_implementation(d, o2, w)
    rel = float((o1.float() - op.float()).norm() / op.float().norm().clamp_min(1e-30))
    allok &= bool(ok) and bool(ok2)
    print("CHECK", json.dumps(dict(bs=bs, seed=seed, dist=dist, D=distinct(d), ok=bool(ok), ok_rep=bool(ok2), same_rep=bool(torch.equal(o1, o2)), rel_vs_parent=f"{rel:.2e}", msg=str(msg)[:100])), flush=True)
for n, dd in VARS.items():
    for bs, seed, dist in ((1, 3, "normal"), (8, 4, "normal"), (4, 5, "skewed")):
        if bs > vmaxm(dd): continue
        d = R.generate_input(bs, "ep8", seed, dist=dist)
        o = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev); C._small_launch(st, d, o, dd, VSCR[n]); torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, o, w); allok &= bool(ok)
        print("VCHECK", n, bs, dist, bool(ok), str(msg)[:80], flush=True)
print("ALLOK_CORRECT", allok, el(), flush=True)

flush = torch.empty(512 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
def capture(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg): pass
empty = statistics.median([t1(eg) for _ in range(30)]); print("EMPTY_US", round(empty, 2), flush=True)

# ---------------- timeline
if A.get("tl", True):
    scr = C._small_state(dev); scr["dbg"] = torch.zeros(G * 16, dtype=torch.int64, device=dev)
    NM = {0: "start", 1: "issued", 2: "bar1", 6: "land0", 3: "A_cred", 9: "B_first", 7: "cmp_done", 4: "epi_done", 5: "release", 10: "final"}
    for bs, seed in ((1, 101), (1, 102), (8, 103), (8, 104)):
        d = R.generate_input(bs, "ep8", seed)
        o = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
        fn = lambda: C._small_launch(st, d, o, DBGD, scr)
        g = capture(fn)
        for r in range(3): t1(g)
        scr["dbg"].zero_(); torch.cuda.synchronize(); tt = t1(g)
        db = scr["dbg"].view(G, 16).cpu().double()
        t00 = db[:, 0].min(); res = {}
        for k, n in NM.items():
            col = db[:, k]; col = col[col > 0]
            if col.numel():
                c = ((col - t00) / 1000).sort().values
                res[n] = [round(float(c[0]), 2), round(float(c[c.numel() // 2]), 2), round(float(c[-1]), 2)]
        res["twait_us"] = [round(float(db[:, 8].mean()) / 1000, 2), round(float(db[:, 8].max()) / 1000, 2)]
        print("TL", json.dumps(dict(bs=bs, seed=seed, D=distinct(d), graph_us=round(tt - empty, 2), **res)), flush=True)
        del g

# ---------------- paired timing
shapes = A.get("shapes", [1, 8, 4]); nseeds = A.get("seeds", 4); reps = A.get("reps", 20)
rows = []
for bs in shapes:
    for k in range(nseeds):
        if el() > BUDGET: print("SKIP_BUDGET", bs, k, flush=True); break
        seed = 7000 + 131 * bs + k
        d = R.generate_input(bs, "ep8", seed)
        outs = {}
        arms = {"parent": lambda: C._v3_parent_kernel(d, st), "cand": lambda: C.kernel_fn(d, st)}
        for n, dd in VARS.items():
            if bs > vmaxm(dd): continue
            o = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev); outs[n] = o
            arms[n] = (lambda dd=dd, o=o, n=n: C._small_launch(st, d, o, dd, VSCR[n]))
        gs = {n: capture(f) for n, f in arms.items()}
        names = list(arms); T = {n: [] for n in names}
        for r in range(reps):
            order = names[r % len(names):] + names[:r % len(names)]
            if r & 1: order = order[::-1]
            for n in order: T[n].append(t1(gs[n]) - empty)
        for n, o in outs.items():
            ok, _ = R.check_implementation(d, o, w)
            if not ok: print("TIMED_CHECK_FAIL", n, bs, seed, flush=True); allok = False
        row = dict(bs=bs, seed=seed, D=distinct(d), **{n: round(statistics.median(v), 2) for n, v in T.items()},
                   **{f"r_{n}": round(statistics.median(sorted(a / b for a, b in zip(T["parent"], T[n]))), 4) for n in names[1:]})
        rows.append(row); print("ROW", json.dumps(row), flush=True)
        del gs
for bs in shapes:
    rr = [r for r in rows if r["bs"] == bs]
    if rr:
        keys = [k for k in rr[0] if k.startswith("r_")]
        print("SUMMARY", bs, {k: round(math.exp(statistics.mean(math.log(r[k]) for r in rr)), 4) for k in keys},
              {k: round(statistics.mean(r[k] for r in rr), 2) for k in rr[0] if not k.startswith("r_") and k not in ("bs", "seed")}, flush=True)
print("ALLOK", allok, "ELAPSED", el(), flush=True)
