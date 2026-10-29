"""Paired H200 screen: single-token split-K MoE (sk1) vs the scored TINY path. Reference check of every variant,
including after repeated graph replays (generation double-buffering); alternating cold-L2 replays."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
for ns in (5,):
    S._sk1_ext(ns)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d, sk1, ns=4):
    S.SK1, S.SK1_NS = sk1, ns
    S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); S.SK1, S.SK1_NS = True, 4; return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
# eager correctness over many seeds first
bad = 0
for seed in range(40):
    d = R.generate_input(1, 'ep8', 1000 + seed)
    o = S.kernel_fn(d, st)
    ok, msg = R.check_implementation(d, o, w)
    if not ok: bad += 1; print('EAGER_FAIL', seed, msg, flush=True)
print('EAGER', 40 - bad, 'of 40 pass', flush=True)
if bad: raise SystemExit(1)
for seed in (419, 7919, 15241, 3, 77):
    d = R.generate_input(1, 'ep8', seed)
    ids = d['topk_ids'].long(); distinct = int(ids[(ids >= 0) & (ids < 32)].unique().numel())
    graphs = {}
    for name, (sk, ns) in {'base': (False, 4), 'sk1_ns4': (True, 4), 'sk1_ns5': (True, 5)}.items():
        g, o = cap(d, sk, ns)
        for rep in range(5):  # several replays: the generation parity flips each launch
            g.replay(); torch.cuda.synchronize()
            ok, msg = R.check_implementation(d, o, w)
            if not ok: raise RuntimeError(f'{name} seed{seed} replay{rep}: {msg}')
        graphs[name] = g
    ts = {k: [] for k in graphs}; names = list(graphs)
    for rep in range(40):
        for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
    med = {k: statistics.median(v) for k, v in ts.items()}
    print('ROW', json.dumps(dict(seed=seed, distinct=distinct, **{k: round(v, 2) for k, v in med.items()},
          **{f'gain_{k}': round(med['base'] / v, 3) for k, v in med.items() if k != 'base'}, msg=msg)), flush=True)
print('DONE', flush=True)
