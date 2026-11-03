"""SMOE (M<=8 dynamic-claim kernel) vs the parent kernel_fn: correctness + paired cold-L2 graph timing."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
print('START', flush=True)
t0 = time.time()
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
print('prepare_s', round(time.time() - t0, 1), 'smoe' in st, flush=True)
VAR = [tuple(int(a) for a in v.split(':')) for v in os.environ.get('VAR', '6:100:0:1').split(',')]
flush = torch.empty(512 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)

def f_old(data):
    return S._smoe_parent_kernel(data, st)

def mk_new(ns, sp, xx, ncp):
    def f(data):
        x = data['hidden_states'].contiguous()
        ids = data['topk_ids'].to(torch.int32).contiguous(); tw = data['topk_weights'].float().contiguous()
        return S._smoe_call(st['smoe'], x, ids, tw, st['lo'], st['cnt'], st['grid'], ns, sp, xx, ncp)
    return f

def capture(fn, data):
    fn(data); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = fn(data)
    g.replay(); torch.cuda.synchronize()
    return g, o

def timed(g):
    flush.sum()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000

eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg):
    pass
empty = statistics.median([timed(eg) for _ in range(30)])
print('empty_graph_us', round(empty, 2), flush=True)

fns = {'old': f_old}
for ns, sp, xx, ncp in VAR:
    fns[f'n{ns}s{sp}x{xx}c{ncp}'] = mk_new(ns, sp, xx, ncp)
cases = [(1, 5041226219440759413, 'normal'), (8, 5041622730710591082, 'normal')]
for bs, seed, dist in cases:
    data = R.generate_input(bs, 'ep8', seed, dist=dist)
    with torch.no_grad():
        ref = R.ref_kernel(data, w)
    nexp = len([e for e in torch.unique(data['topk_ids']).tolist() if 0 <= e < 32])
    graphs = {}
    bad = False
    for k, fn in fns.items():
        graphs[k] = capture(fn, data)
        for rep in range(3):
            graphs[k][0].replay()
        torch.cuda.synchronize()
        if 'x1' in k or 'x2' in k: continue
        ok, msg = compare(graphs[k][1], ref, R.REL_L2_TOL)
        print('CHECK', k, bs, seed, dist, ok, msg, flush=True)
        bad |= not ok
    keys = list(fns)
    T = {k: [] for k in keys}
    for i in range(30):
        order = keys if i % 2 == 0 else keys[::-1]
        for k in order:
            T[k].append(timed(graphs[k][0]))
    med = {k: round(statistics.median(v) - empty, 2) for k, v in T.items()}
    print('ROW', json.dumps({'bs': bs, 'seed': seed, 'dist': dist, 'experts': nexp, 'us_minus_empty': med,
                             'gain': {k: round(med['old'] / med[k], 3) for k in keys if k != 'old'}}), flush=True)
    del graphs
print('DONE', flush=True)
