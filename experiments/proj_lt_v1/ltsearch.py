"""Exhaustive-ish cuBLASLt search for one (layer, m) cell on H200 (ctypes only, nothing compiled).

Env: CELL=layer:m, BUDGET=seconds. Timing = the judge's: one call per CUDA-graph replay, cold L2 (flush.sum()) before each.
Stages: A) heuristics (4 layouts x 2 compute types) + per-algo-id tile x stages x cluster grid (split-K 1);
B) coordinate refinement of the leaders over stages / custom / cluster / swizzle / split-K x reduction / compute type;
C) interleaved re-timing of the finalists against the parent plan (ag pin or best heuristic)."""
import ctypes as C, glob, json, os, random, re, statistics, sys, time
import torch

LAYER, M = os.environ['CELL'].split(':'); M = int(M)
BUDGET = float(os.environ.get('BUDGET', '600'))
T0 = time.time()
DIMS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
N, K = DIMS[LAYER]

# ---------------------------------------------------------------- enums from the image's header when present
ENUM = {}
for h in glob.glob('/usr/local/cuda*/include/cublasLt.h') + glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/cu*/include/cublasLt.h') + \
        glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/cublas/include/cublasLt.h') + glob.glob('/usr/include/cublasLt.h'):
    for name, val in re.findall(r'\b(CUBLASLT_\w+)\s*=\s*(\d+)', open(h, errors='ignore').read()):
        ENUM.setdefault(name, int(val))
    print('HEADER', h, len(ENUM), flush=True)
    break
def E(name, default):
    return ENUM.get(name, default)
RNAME = {}
for prefix in ('CUBLASLT_MATMUL_TILE_', 'CUBLASLT_MATMUL_STAGES_', 'CUBLASLT_CLUSTER_SHAPE_'):
    RNAME[prefix] = {v: k[len(prefix):] for k, v in ENUM.items() if k.startswith(prefix)}
CAP_SPLITK = E('CUBLASLT_ALGO_CAP_SPLITK_SUPPORT', 0); CAP_RED = E('CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK', 1)
CAP_SWZ = E('CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT', 2); CAP_TILES = E('CUBLASLT_ALGO_CAP_TILE_IDS', 6)
CAP_CUSTOM = E('CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX', 7); CAP_STAGES = E('CUBLASLT_ALGO_CAP_STAGES_IDS', 13)
CAP_NUM = E('CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS', 15)
print('CLUSTERS', sorted(v for k, v in ENUM.items() if k.startswith('CUBLASLT_CLUSTER_SHAPE_')))
print('CAPS', CAP_SPLITK, CAP_RED, CAP_SWZ, CAP_TILES, CAP_CUSTOM, CAP_STAGES, CAP_NUM, flush=True)
# config attrs: ID, TILE, SPLITK, RED, SWZ, CUSTOM, STAGES, INNER, CLUSTER
CFG_T = [C.c_int32, C.c_uint32, C.c_int32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint16, C.c_uint16]

for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
    try:
        L = C.CDLL(name); break
    except OSError:
        pass
P = C.c_void_p
class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]
class Heur(C.Structure):
    _fields_ = [("algo", Algo), ("workspaceSize", C.c_size_t), ("state", C.c_int), ("wavesCount", C.c_float), ("reserved", C.c_int * 4)]
for name, a in {"cublasLtCreate": [C.POINTER(P)], "cublasLtMatmulDescCreate": [C.POINTER(P), C.c_int, C.c_int],
                "cublasLtMatmulDescSetAttribute": [P, C.c_int, P, C.c_size_t],
                "cublasLtMatrixLayoutCreate": [C.POINTER(P), C.c_int, C.c_uint64, C.c_uint64, C.c_int64],
                "cublasLtMatrixLayoutSetAttribute": [P, C.c_int, P, C.c_size_t],
                "cublasLtMatmulPreferenceCreate": [C.POINTER(P)], "cublasLtMatmulPreferenceSetAttribute": [P, C.c_int, P, C.c_size_t],
                "cublasLtMatmulAlgoGetHeuristic": [P, P, P, P, P, P, P, C.c_int, C.POINTER(Heur), C.POINTER(C.c_int)],
                "cublasLtMatmulAlgoGetIds": [P, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int)],
                "cublasLtMatmulAlgoInit": [P, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(Algo)],
                "cublasLtMatmulAlgoCapGetAttribute": [C.POINTER(Algo), C.c_int, P, C.c_size_t, C.POINTER(C.c_size_t)],
                "cublasLtMatmulAlgoConfigSetAttribute": [C.POINTER(Algo), C.c_int, P, C.c_size_t],
                "cublasLtMatmulAlgoConfigGetAttribute": [C.POINTER(Algo), C.c_int, P, C.c_size_t, C.POINTER(C.c_size_t)],
                "cublasLtMatmulAlgoCheck": [P, P, P, P, P, P, C.POINTER(Algo), C.POINTER(Heur)],
                "cublasLtMatmul": [P] * 12 + [C.POINTER(Algo), P, C.c_size_t, P]}.items():
    f = getattr(L, name); f.argtypes, f.restype = a, C.c_int
H = P(); assert L.cublasLtCreate(C.byref(H)) == 0
BF16, F32 = 14, 0
COMPUTES = (68, 75)  # CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_16BF
MODES = ("tn", "nn", "rownn", "rowtn")
WS = 64 << 20
ws = torch.empty(WS, dtype=torch.uint8, device='cuda')

def layout(rows, cols, ld, row):
    d = P(); assert L.cublasLtMatrixLayoutCreate(C.byref(d), BF16, rows, cols, ld) == 0
    if row:
        o = C.c_int(1); L.cublasLtMatrixLayoutSetAttribute(d, 1, C.byref(o), 4)
    return d
DESC, LAY = {}, {}
for comp in COMPUTES:
    for mode in MODES:
        d = P(); assert L.cublasLtMatmulDescCreate(C.byref(d), comp, F32) == 0
        ta = C.c_int(1 if mode == "tn" else 0); tb = C.c_int(1 if mode == "rowtn" else 0)
        L.cublasLtMatmulDescSetAttribute(d, 3, C.byref(ta), 4); L.cublasLtMatmulDescSetAttribute(d, 4, C.byref(tb), 4)
        DESC[(comp, mode)] = d
for mode in MODES:
    if mode == "tn":
        LAY[mode] = (layout(K, N, K, 0), layout(K, M, K, 0), layout(N, M, N, 0))
    elif mode == "nn":
        LAY[mode] = (layout(N, K, N, 0), layout(K, M, K, 0), layout(N, M, N, 0))
    elif mode == "rownn":
        LAY[mode] = (layout(M, K, K, 1), layout(K, N, N, 1), layout(M, N, N, 1))
    else:
        LAY[mode] = (layout(M, K, K, 1), layout(N, K, K, 1), layout(M, N, N, 1))

torch.manual_seed(0)
W = torch.randn(N, K, device='cuda').to(torch.bfloat16)
WT = W.t().contiguous()
X = torch.randn(M, K, device='cuda').to(torch.bfloat16)
REF = X.float() @ W.float().T
OUT = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
alpha, beta = C.c_float(1), C.c_float(0)
def ptrs(mode):
    if mode == "tn": return W.data_ptr(), X.data_ptr()
    if mode == "nn": return WT.data_ptr(), X.data_ptr()
    if mode == "rownn": return X.data_ptr(), WT.data_ptr()
    return X.data_ptr(), W.data_ptr()

def getcfg(a):
    out = []
    for i, t in enumerate(CFG_T):
        v = t(0); w = C.c_size_t(); L.cublasLtMatmulAlgoConfigGetAttribute(C.byref(a), i, C.byref(v), C.sizeof(v), C.byref(w)); out.append(int(v.value))
    return tuple(out)
def mkalgo(comp, cfg):
    a = Algo()
    if L.cublasLtMatmulAlgoInit(H, comp, F32, BF16, BF16, BF16, BF16, cfg[0], C.byref(a)):
        return None
    for i in range(1, 9):
        v = CFG_T[i](cfg[i])
        if L.cublasLtMatmulAlgoConfigSetAttribute(C.byref(a), i, C.byref(v), C.sizeof(v)):
            return None
    return a
def check(comp, mode, a):
    A, B, Cc = LAY[mode]; r = Heur()
    s = L.cublasLtMatmulAlgoCheck(H, DESC[(comp, mode)], A, B, Cc, Cc, C.byref(a), C.byref(r))
    return s == 0 and r.state == 0 and r.workspaceSize <= WS
def mm(comp, mode, a, stream=None):
    A, B, Cc = LAY[mode]; pa, pb = ptrs(mode)
    s = torch.cuda.current_stream().cuda_stream if stream is None else stream
    return L.cublasLtMatmul(H, DESC[(comp, mode)], C.byref(alpha), P(pa), A, P(pb), B, C.byref(beta), P(OUT.data_ptr()), Cc,
                            P(OUT.data_ptr()), Cc, C.byref(a), P(ws.data_ptr()), WS, P(s))

flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
EV = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(200)]
def graph_of(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    return g
def tm_graph(g, reps):
    for i in range(reps):
        a, b = EV[i]
        flush.sum(); a.record(); g.replay(); b.record()
    torch.cuda.synchronize()
    return statistics.median(EV[i][0].elapsed_time(EV[i][1]) * 1000 for i in range(reps))

SEEN, RES = set(), []   # RES: (us, comp, mode, cfg)
NTRY = [0, 0]
def trycand(comp, mode, a, reps=7, tag=''):
    cfg = getcfg(a)
    key = (comp, mode, cfg)
    if key in SEEN:
        return None
    SEEN.add(key)
    NTRY[0] += 1
    if not check(comp, mode, a):
        return None
    OUT.zero_()
    if mm(comp, mode, a):
        return None
    torch.cuda.synchronize()
    err = float((OUT.float() - REF).norm() / REF.norm())
    if not (err < 5e-3):
        print('BADNUM', comp, mode, cfg, err, flush=True); return None
    NTRY[1] += 1
    try:
        g = graph_of(lambda: mm(comp, mode, a))
        t = tm_graph(g, reps); del g
    except Exception as e:
        print('GRAPHFAIL', comp, mode, cfg, str(e)[:80], flush=True); return None
    RES.append((t, comp, mode, cfg, err))
    return t

def caps(comp, aid):
    a = Algo()
    if L.cublasLtMatmulAlgoInit(H, comp, F32, BF16, BF16, BF16, BF16, aid, C.byref(a)):
        return None
    def arr(cap):
        w = C.c_size_t(); L.cublasLtMatmulAlgoCapGetAttribute(C.byref(a), cap, None, 0, C.byref(w))
        n = w.value // 4; buf = (C.c_uint32 * max(n, 1))()
        if n: L.cublasLtMatmulAlgoCapGetAttribute(C.byref(a), cap, buf, w.value, C.byref(w))
        return [int(buf[i]) for i in range(n)]
    def scal(cap):
        w = C.c_size_t(); L.cublasLtMatmulAlgoCapGetAttribute(C.byref(a), cap, None, 0, C.byref(w))
        v = (C.c_uint64 if w.value == 8 else C.c_uint32 if w.value == 4 else C.c_uint16)(0)
        L.cublasLtMatmulAlgoCapGetAttribute(C.byref(a), cap, C.byref(v), C.sizeof(v), C.byref(w)); return int(v.value)
    return {"tiles": arr(CAP_TILES), "stages": arr(CAP_STAGES), "splitk": scal(CAP_SPLITK), "red": scal(CAP_RED),
            "swz": scal(CAP_SWZ), "custom": scal(CAP_CUSTOM), "num": scal(CAP_NUM)}

def left():
    return BUDGET - (time.time() - T0)

# ---------------------------------------------------------------- parent plan (the ag pins) as reference
PINS = {("qkv_gate", 256): ("tn", 124, 0, 2), ("pooler", 256): ("rownn", 18, 0, 2), ("o", 256): ("tn", 20, 6, 2),
        ("unpooler", 256): ("rownn", 20, 4, 2), ("pooler", 128): ("nn", 15, 4, 2)}
REFPLAN = None
if (LAYER, M) in PINS:
    mode, tile, cu, cl = PINS[(LAYER, M)]
    REFPLAN = (68, mode, (66, tile, 1, 0, 0, cu, 35, 0, cl))
    a = mkalgo(68, REFPLAN[2]); t = trycand(68, mode, a, 15)
    print('PIN', REFPLAN, t, flush=True)

# ---------------------------------------------------------------- A1: heuristics
for comp in COMPUTES:
    for mode in MODES:
        pref = P(); L.cublasLtMatmulPreferenceCreate(C.byref(pref))
        wsz = C.c_uint64(WS); L.cublasLtMatmulPreferenceSetAttribute(pref, 1, C.byref(wsz), 8)
        A, B, Cc = LAY[mode]; res = (Heur * 128)(); cnt = C.c_int(0)
        L.cublasLtMatmulAlgoGetHeuristic(H, DESC[(comp, mode)], A, B, Cc, Cc, pref, 128, res, C.byref(cnt))
        hs = [getcfg(res[i].algo) for i in range(cnt.value)]
        print('HEUR', comp, mode, cnt.value, hs[:3], flush=True)
        for i in range(cnt.value):
            trycand(comp, mode, res[i].algo)
heur_best = min(RES) if RES else None
print('HEURBEST', heur_best, 'tried', NTRY, round(time.time() - T0, 1), flush=True)
if REFPLAN is None and heur_best:
    REFPLAN = heur_best[1:4]

# ---------------------------------------------------------------- A2: algo-id grid (compute 32F)
ids = (C.c_int * 256)(); cnt = C.c_int(0)
L.cublasLtMatmulAlgoGetIds(H, 68, F32, BF16, BF16, BF16, BF16, 256, ids, C.byref(cnt))
IDS = [ids[i] for i in range(cnt.value)]
CAPS = {aid: caps(68, aid) for aid in IDS}
for aid in IDS:
    c = CAPS[aid]
    print('ID', aid, 'tiles', len(c['tiles']), 'stages', len(c['stages']), 'sk', c['splitk'], 'red', c['red'], 'swz', c['swz'], 'cust', c['custom'], 'num', hex(c['num']), flush=True)
ids75 = (C.c_int * 256)(); cnt75 = C.c_int(0)
L.cublasLtMatmulAlgoGetIds(H, 75, F32, BF16, BF16, BF16, BF16, 256, ids75, C.byref(cnt75))
print('IDS75', sorted(ids75[i] for i in range(cnt75.value)), 'IDS68', sorted(IDS), flush=True)
CLUSTERS = sorted(v for k, v in ENUM.items() if k.startswith('CUBLASLT_CLUSTER_SHAPE_') and 'END' not in k) or list(range(0, 40))
order = sorted(IDS, key=lambda i: (i != 66, i))
random.seed(1)
for aid in order:
    c = CAPS[aid]
    tiles = c['tiles'] or [0]; stages = c['stages'] or [0]
    combos = []
    for mode in MODES:
        for tile in tiles:
            if aid in (66, 67):
                for sg in stages:
                    for cl in CLUSTERS:
                        combos.append((mode, (aid, tile, 1, 0, 0, 0, sg, 0, cl)))
            else:
                for sg in stages:
                    combos.append((mode, (aid, tile, 1, 0, 0, 0, sg, 0, 0)))
    if len(combos) > 200000:
        combos = random.sample(combos, 200000)
    if aid == 67:  # diagnose why no 67 config passes: statuses of a few plain inits
        for mode in MODES:
            for tile in tiles[:3]:
                for cu in (0, 1, 8):
                    for sk in (1, 2):
                        a = mkalgo(68, (67, tile, sk, 0 if sk == 1 else 2, 0, cu, stages[0], 0, 0))
                        if a is None:
                            print('D67 init/set fail', mode, tile, cu, sk); continue
                        A_, B_, C_ = LAY[mode]; r = Heur()
                        st_ = L.cublasLtMatmulAlgoCheck(H, DESC[(68, mode)], A_, B_, C_, C_, C.byref(a), C.byref(r))
                        print('D67', mode, tile, cu, sk, 'status', st_, 'state', r.state, 'ws', r.workspaceSize, flush=True)
        combos = [(mode, (67, tile, sk, rd, 0, cu, sg, 0, cl)) for mode in MODES for tile in tiles for sg in stages
                  for cl in (0, 2, 3, 5) for cu in (0, 1, 2, 4, 8, 16, 32, 64, 151) for sk, rd in ((1, 0), (2, 2), (4, 2))]
        combos = random.sample(combos, min(len(combos), 30000))
    n0 = len(RES)
    for mode, cfg in combos:
        if left() < BUDGET * (0.45 if aid not in (66, 67) else 0.55):
            break
        a = mkalgo(68, cfg)
        if a is not None:
            trycand(68, mode, a, 5)
    print('GRID id', aid, 'combos', len(combos), 'timed', len(RES) - n0, 'best', min(RES[n0:]) if len(RES) > n0 else None,
          round(time.time() - T0, 1), flush=True)

# ---------------------------------------------------------------- B: refinement of the leaders
RES.sort()
leaders, seenl = [], set()
for r in RES:
    k = (r[2], r[3][0], r[3][1])
    if k not in seenl:
        seenl.add(k); leaders.append(r)
    if len(leaders) >= 40:
        break
print('LEADERS', [(round(r[0], 2), r[1], r[2], r[3]) for r in leaders], flush=True)
for r in leaders:
    if left() < BUDGET * 0.15:
        break
    # refine around this leader only (not the global best): temporarily treat it as the base
    t, comp, mode, cfg, _ = r
    base = (t, comp, mode, cfg)
    c = CAPS.get(cfg[0]) or caps(68, cfg[0])
    for dim in ('stages', 'custom', 'cluster', 'swz', 'sk', 'custom', 'comp'):
        comp, mode, cfg = base[1:]
        if dim == 'stages': cands = [(comp, mode, cfg[:6] + (sg,) + cfg[7:]) for sg in c['stages']]
        elif dim == 'custom': cands = [(comp, mode, cfg[:5] + (cu,) + cfg[6:8] + (cl,)) for cu in range(min(c['custom'], 15) + 1) for cl in CLUSTERS]
        elif dim == 'cluster': cands = [(comp, mode, cfg[:8] + (cl,)) for cl in CLUSTERS]
        elif dim == 'swz': cands = [(comp, mode, cfg[:4] + (sw,) + cfg[5:]) for sw in range(2 if c['swz'] else 1)]
        elif dim == 'sk': cands = [(comp, mode, cfg[:2] + (sk, rd) + cfg[4:]) for sk in (2, 3, 4, 6, 8) for rd in (1, 2, 4)] if c['splitk'] else []
        else: cands = [(cp, mode, cfg) for cp in COMPUTES]
        for cp, md, cf in cands:
            if left() < BUDGET * 0.12:
                break
            a = mkalgo(cp, cf)
            if a is not None:
                tt = trycand(cp, md, a, 7)
                if tt is not None and tt < base[0]:
                    base = (tt, cp, md, cf)
    print('REFINED', (round(r[0], 2), r[2], r[3]), '->', (round(base[0], 2), base[1], base[2], base[3]), round(time.time() - T0, 1), flush=True)
print('TRIED', NTRY, 'timed', len(RES), flush=True)

# ---------------------------------------------------------------- C: interleaved finals vs the parent plan
RES.sort()
fin, seenf = [], set()
for r in RES:
    k = (r[1], r[2], r[3])
    if k not in seenf:
        seenf.add(k); fin.append(k)
    if len(fin) >= 12:
        break
if REFPLAN and tuple(REFPLAN) not in fin:
    fin.append(tuple(REFPLAN))
graphs = {}
for k in fin:
    a = mkalgo(k[0], k[2]) if k != (None,) else None
    graphs[k] = (a, graph_of(lambda a=a, k=k: mm(k[0], k[1], a)))
# torch default for context
graphs[('torch',)] = (None, graph_of(lambda: torch.mm(X, W.T, out=OUT)))
ts = {k: [] for k in graphs}
for rnd in range(6):
    ks = list(graphs)
    if rnd % 2: ks = ks[::-1]
    for k in ks:
        ts[k].append(tm_graph(graphs[k][1], 40))
refk = tuple(REFPLAN) if REFPLAN else None
reft = statistics.median(ts[refk]) if refk else None
rows = sorted((statistics.median(v), k) for k, v in ts.items())
for t, k in rows:
    ratio = [a / b for a, b in zip(ts[refk], ts[k])] if refk else [0]
    print('FINAL', json.dumps({'cell': f'{LAYER}:{M}', 'us': round(t, 3), 'ratio_vs_ref': round(statistics.median(ratio), 4),
                               'key': k, 'ref': refk is not None and k == refk}), flush=True)
print('DONE', LAYER, M, round(time.time() - T0, 1), flush=True)
