"""Paired parent/candidate H200 screen (judge image). Parent = exact scored FP32 source, candidate = make_candidate.py output.
Cold-L2 CUDA-graph replays alternate parent/candidate order every repetition; every output is reference-checked."""
import importlib.util, json, math, statistics, sys, time, concurrent.futures
import torch
T0 = time.time()
BUDGET = 255.0
def el(): return round(time.time() - T0, 1)

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

def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
P = load("parent", "/w/run/parent.py"); C = load("cand", "/w/run/cand.py")
w = R.generate_weights(R.WEIGHTS_SEED)
print("LOADED", el(), flush=True)

# warm the candidate's two new MAIN builds in the background while the parent prepares (same flags prepare() will use)
lo, cnt = C.LOCAL_START, C.LOCAL_EXPERTS
prmt = C.PRMT_SCALES and C._scales_prmt_exact(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
sc6 = bool(prmt) and C._sc6_ok(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
sc6w = sc6 and C.SC6_WGS and C._wgs_on()
sfx = ("-DOKC_PRMTSC=1",) if prmt else ()
fake = {"sfx": sfx, "sfxm": sfx + (("-DOKC_SC6W=1",) if sc6w else ()), "sfxs": sfx + (("-DOKC_SC6S=1",) if (sc6 and C.SC6_SKINNY) else ())}
warm = [tuple(C._main_defines(sfx)) + fake["sfxm"], C._ps_defs(32, fake)]
print("WARM", warm, flush=True)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
futs = [pool.submit(C._ext, d) for d in warm]
sp_ = P.prepare(w); print("PARENT_PREPARED", el(), flush=True)
for f in futs: f.result()
print("WARM_DONE", el(), flush=True)
sc_ = C.prepare(w); print("CAND_PREPARED", el(), flush=True)
assert not (sc_["ps"].keys() - {8, 32}) and 32 in sc_["ps"]
print("CAND_PS32_DEFS", [d for d in sc_["ps"][32] if "WGS" in d or "NW" in d], flush=True)
print("CAND_ATTRS", C._ext(sc_["ps"][32]).okc_attrs(), "PARENT_MAIN_ATTRS", P._ext(tuple(P._main_defines(sp_["sfx"])) + sp_["sfxm"]).okc_attrs(), flush=True)

flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
def capture(M, st, d):
    o = M.kernel_fn(d, st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg): pass
empty = statistics.median([t1(eg) for _ in range(30)]); print("EMPTY_GRAPH_US", round(empty, 2), flush=True)

allok = True
# correctness (candidate, default and ps dispatch), incl. skewed (one expert holds all 32 tokens: x slot exactly full)
checks = [(4, 1, "normal"), (33, 2, "normal"), (257, 3, "normal"), (128, 4, "skewed"), (1, 5, "normal"),
          (32, 11, "skewed"), (32, 12, "skewed"), (32, 13, "normal"), (31, 14, "normal"), (30, 15, "skewed"), (17, 16, "skewed"), (24, 17, "normal"), (32, 18, "normal")]
for bs, seed, dist in checks:
    if el() > BUDGET - 120: print("SKIP_CHECKS_BUDGET", flush=True); break
    d = R.generate_input(bs, "ep8", seed, dist=dist)
    oc = C.kernel_fn(d, sc_); op = P.kernel_fn(d, sp_); torch.cuda.synchronize()
    ok, msg = R.check_implementation(d, oc, w); okp, _ = R.check_implementation(d, op, w)
    same = bool(torch.equal(oc, op))
    allok &= bool(ok)
    print("CHECK", json.dumps(dict(bs=bs, seed=seed, dist=dist, cand_ok=bool(ok), parent_ok=bool(okp), bitwise_equal=same, msg=str(msg)[:120])), flush=True)

rows = []
plan = [(32, s) for s in (419, 7919, 15241, 3, 77, 1001)] + [(16, 101), (16, 202), (256, 505), (256, 606), (128, 404)]
for bs, seed in plan:
    if el() > BUDGET:
        print("SKIP_TIMING_BUDGET", bs, seed, flush=True); continue
    d = R.generate_input(bs, "ep8", seed)
    gp, op = capture(P, sp_, d); gc, oc = capture(C, sc_, d)
    ok, msg = R.check_implementation(d, oc, w); allok &= bool(ok)
    if not ok: print("CHECK_FAIL", bs, seed, msg, flush=True)
    reps = 40 if bs == 32 else 24
    tp, tc = [], []
    for r in range(reps):
        if r % 2: tc.append(t1(gc)); tp.append(t1(gp))
        else: tp.append(t1(gp)); tc.append(t1(gc))
    ok2, msg2 = R.check_implementation(d, oc, w); allok &= bool(ok2)   # candidate output after the timed replays
    pr = sorted((a - empty) / (b - empty) for a, b in zip(tp, tc))
    row = dict(bs=bs, seed=seed, parent_us=round(statistics.median(tp) - empty, 2), cand_us=round(statistics.median(tc) - empty, 2),
               paired_med=round(pr[len(pr) // 2], 4), paired_q25=round(pr[len(pr) // 4], 4), paired_q75=round(pr[3 * len(pr) // 4], 4),
               ok_after=bool(ok2), same_graph_path=bool(bs not in (32,)))
    rows.append(row); print("ROW", json.dumps(row), flush=True)
    del gp, gc
per = {}
for r in rows: per.setdefault(r["bs"], []).append(r["paired_med"])
print("PER_BS", json.dumps({k: round(math.exp(statistics.mean(math.log(x) for x in v)), 4) for k, v in per.items()}), flush=True)
print("ALLOK", allok, "ELAPSED", el(), flush=True)
print("DONE", flush=True)
