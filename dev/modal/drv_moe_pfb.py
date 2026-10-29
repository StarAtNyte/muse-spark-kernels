"""Paired H200 screen: phase-B L2 prefetch (OKC_PFB) in the TINY (bs 1) and TINY2 (bs 8) MoE builds.
Same module, variants selected by swapping the defines tuples between graph captures; alternating cold-L2 replays."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
T0, T2 = tuple(S.TINY_DEFINES), tuple(S.TINY2_DEFINES)
VARIANTS = {'base': (), 'pfb1': ('-DOKC_PFB=1',), 'pfb2': ('-DOKC_PFB=2',), 'pfb2_l2h': ('-DOKC_PFB=2', '-DOKC_L2H=1')}
def tiny_l2h(defs, extra):
    if '-DOKC_L2H=1' in extra: defs = tuple(d for d in defs if not d.startswith('-DOKC_L2H'))
    return defs + extra
def cap(d):
    S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for bs in (1, 8):
    for seed in (419, 7919, 15241, 3, 77):
        d = R.generate_input(bs, 'ep8', seed)
        ids = d['topk_ids'].long(); distinct = int(ids[(ids >= 0) & (ids < 32)].unique().numel())
        graphs = {}
        for name, extra in VARIANTS.items():
            S.TINY_DEFINES = tiny_l2h(T0, extra); S.TINY2_DEFINES = tiny_l2h(T2, extra)
            g, o = cap(d)
            ok, msg = R.check_implementation(d, o, w)
            if not ok: raise RuntimeError(f'{name} bs{bs} seed{seed} failed: {msg}')
            graphs[name] = g
        S.TINY_DEFINES, S.TINY2_DEFINES = T0, T2
        ts = {k: [] for k in graphs}
        names = list(graphs)
        for rep in range(30):
            order = names if rep % 2 == 0 else names[::-1]
            for k in order: ts[k].append(t1(graphs[k]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(bs=bs, seed=seed, distinct=distinct, **{k: round(v, 2) for k, v in med.items()},
              **{f'gain_{k}': round(med['base'] / v, 3) for k, v in med.items() if k != 'base'})), flush=True)
print('DONE', flush=True)
