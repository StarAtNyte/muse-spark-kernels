"""moe_dq5_v1 driver (judge image, one H200): parent (sub.py) vs DQ5 candidate (cand.py).
1. correctness of the candidate on the 5 MoE tests + the scored shapes, bit-equality vs the parent at bs 1..8;
2. SASS opcode counts of the TINY build (parent vs candidate variants);
3. paired cold-L2 CUDA-graph timing at bs 1/4/8 (arms: parent, cand = DQ5, dq6, dq7ctl = candidate text with the
   parent decode, nodq = 4-op fake decode ceiling, wrong outputs) and bs 16/32 (parent vs cand only).
DRV_ARGS json: {"shapes": [...], "seeds": n, "reps": n, "big": [16, 32], "bigseeds": n, "budget": s}"""
import importlib.util, json, math, os, re, statistics, subprocess, sys, time, concurrent.futures
import torch
T0 = time.time()
def el(): return round(time.time() - T0, 1)
A = json.loads(os.environ.get("DRV_ARGS") or "{}")
BUDGET = float(A.get("budget", 800))
sys.path.insert(0, "/w/kernels/muse_moe_w4a16")
import reference as R
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
P = load("par", "/w/run/sub.py"); C = load("cand", "/w/run/cand.py")
w = R.generate_weights(R.WEIGHTS_SEED)
stP = P.prepare(w); print("PREPARED parent", el(), flush=True)
stC = C.prepare(w); print("PREPARED cand", el(), "DQ5 in TINY", "-DOKC_DQ5=1" in C.TINY_DEFINES, flush=True)
dev = "cuda"

def cand_defs(M, variant):
    """the candidate's defines for size M, with the DQ5 flag swapped for the variant's"""
    if M in stC.get("ps", {}): d = tuple(stC["ps"][M])
    elif M <= C.TINY_MAX_M: d = tuple(C.TINY_DEFINES) + tuple(stC.get("sfxs", stC["sfx"]))
    else: d = tuple(C.TINY2_DEFINES) + tuple(stC.get("sfxs", stC["sfx"]))
    d = tuple(x for x in d if x != "-DOKC_DQ5=1")
    extra = {"cand": ("-DOKC_DQ5=1",), "dq6": ("-DOKC_DQ5=2",), "dq7ctl": (), "nodq": ("-DOKC_NODQ=1",)}[variant]
    at = [x for x in d if x.startswith("@")]
    return tuple(x for x in d if not x.startswith("@")) + extra + tuple(at)
VARS = ("cand", "dq6", "dq7ctl", "nodq")
SMALL = A.get("shapes", [1, 4, 8])
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = sorted({cand_defs(M, v) for M in SMALL for v in VARS})
for f in [pool.submit(C._ext, d) for d in todo]: f.result()
print("BUILT", len(todo), el(), flush=True)

# ---------------- SASS opcode counts (TINY build, bs1 path)
def sass_counts(ext):
    try:
        so = ext.__file__
        out = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True, timeout=120).stdout
        ops = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)", out)
        keys = ("LOP3", "SHF", "IMAD.SHL", "IMAD.WIDE", "IMAD", "HMUL2", "HMMA", "PRMT", "LDS")
        res = {}
        for k in keys:
            res[k] = sum(1 for o in ops if o == k or o.startswith(k + "."))
        res["total"] = len(ops)
        return res
    except Exception as ex:
        return str(ex)[-200:]
for M in (1, 8):
    pd = tuple(P.TINY_DEFINES) + tuple(stP.get("sfxs", stP["sfx"])) if M <= P.TINY_MAX_M else tuple(stP["ps"][M])
    print("SASS parent", M, sass_counts(P._ext(pd)), flush=True)
    for v in ("cand", "dq6", "dq7ctl"):
        print("SASS", v, M, sass_counts(C._ext(cand_defs(M, v))), flush=True)

# ---------------- correctness
allok = True
tests = [(4, "normal"), (33, "normal"), (257, "normal"), (128, "skewed"), (1, "normal")] + [(b, "normal") for b in (2, 3, 5, 6, 7, 8, 16, 32, 128, 256)] + [(b, "skewed") for b in (1, 4, 8)]
for i, (bs, dist) in enumerate(tests):
    for seed in (11 + i, 1234 + i):
        d = R.generate_input(bs, "ep8", seed, dist=dist)
        oc = C.kernel_fn(d, stC).clone(); oc2 = C.kernel_fn(d, stC).clone(); op = P.kernel_fn(d, stP); torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, oc, w); okp, msgp = R.check_implementation(d, op, w)
        allok &= bool(ok)
        print("CHECK", json.dumps(dict(bs=bs, dist=dist, seed=seed, ok=bool(ok), parent_ok=bool(okp), bit_eq_parent=bool(torch.equal(oc, op)),
                                       cand_rep_eq=bool(torch.equal(oc, oc2)), msg=str(msg)[:90], pmsg=str(msgp)[:90])), flush=True)
for M in SMALL:
    for v in ("cand", "dq6", "dq7ctl"):
        d = R.generate_input(M, "ep8", 77 + M)
        o = torch.empty((M, 4096), dtype=torch.bfloat16, device=dev)
        C._ext(cand_defs(M, v)).okc_run(stC["w"], stC["scratch"], d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(),
                                        d["topk_weights"].float().contiguous(), o, stC["lo"], stC["cnt"], stC["grid"])
        op = P.kernel_fn(d, stP); torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, o, w); allok &= bool(ok)
        print("VCHECK", v, M, bool(ok), "bit_eq_parent", bool(torch.equal(o, op)), str(msg)[:80], flush=True)
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

def distinct(d):
    ids = d["topk_ids"].long(); return int(ids[ids < 32].unique().numel())
rows = []
def timed(bs, seed, arms, reps):
    d = R.generate_input(bs, "ep8", seed)
    x = d["hidden_states"].contiguous(); ids = d["topk_ids"].to(torch.int32).contiguous(); tw = d["topk_weights"].float().contiguous()
    fns = {"parent": lambda: P.kernel_fn(d, stP), "cand": lambda: C.kernel_fn(d, stC)}
    for v in arms:
        if v in fns: continue
        o = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev); ext = C._ext(cand_defs(bs, v))
        fns[v] = (lambda ext=ext, o=o: ext.okc_run(stC["w"], stC["scratch"], x, ids, tw, o, stC["lo"], stC["cnt"], stC["grid"]))
    fns = {k: fns[k] for k in ["parent"] + list(arms)}
    gs = {n: capture(f) for n, f in fns.items()}
    names = list(fns); T = {n: [] for n in names}
    for r in range(reps):
        order = names[r % len(names):] + names[:r % len(names)]
        if r & 1: order = order[::-1]
        for n in order: T[n].append(t1(gs[n]) - empty)
    row = dict(bs=bs, seed=seed, D=distinct(d), **{n: round(statistics.median(v), 2) for n, v in T.items()},
               **{f"r_{n}": round(statistics.median(sorted(a / b for a, b in zip(T["parent"], T[n]))), 4) for n in names[1:]})
    rows.append(row); print("ROW", json.dumps(row), flush=True)

nseeds, reps = A.get("seeds", 5), A.get("reps", 20)
for k in range(nseeds):
    for bs in SMALL:
        if el() > BUDGET: print("SKIP_BUDGET", bs, k, flush=True); continue
        timed(bs, 9000 + 131 * bs + k, VARS, reps)
for k in range(A.get("bigseeds", 2)):
    for bs in A.get("big", [16, 32]):
        if el() > BUDGET: print("SKIP_BUDGET", bs, k, flush=True); continue
        timed(bs, 9500 + 131 * bs + k, ("cand",), reps)
for bs in SMALL + A.get("big", [16, 32]):
    rr = [r for r in rows if r["bs"] == bs]
    if rr:
        keys = [k for k in rr[0] if k.startswith("r_")]
        print("SUMMARY", bs, {k: round(math.exp(statistics.mean(math.log(r[k]) for r in rr)), 4) for k in keys},
              {k: round(statistics.mean(r[k] for r in rr), 2) for k in rr[0] if not k.startswith("r_") and k not in ("bs", "seed")}, flush=True)
print("ALLOK", allok, "ELAPSED", el(), flush=True)
