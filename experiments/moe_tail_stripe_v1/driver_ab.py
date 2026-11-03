"""Paired cold-L2 graph timing: baseline (kernels/muse_moe_w4a16/submission.py == moe_keell0renz_compare_v1) vs the
candidate (/w/run/sub.py), both through kernel_fn; correctness of both against the FP32 reference."""
import importlib.util, json, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
OLD = load('old_sub', '/w/kernels/muse_moe_w4a16/submission.py')
NEW = load('new_sub', '/w/run/sub.py')
print('START', flush=True)
t0 = time.time()
w = R.generate_weights(R.WEIGHTS_SEED)
st_old = OLD.prepare(w); print('prep old', round(time.time() - t0), flush=True)
st_new = NEW.prepare(w); print('prep new', round(time.time() - t0), flush=True)
flush = torch.empty(512 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
fns = {'old': lambda d: OLD.kernel_fn(d, st_old), 'new': lambda d: NEW.kernel_fn(d, st_new)}
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
# correctness-only cases (the judge's tests and nearby sizes), then timed cases
checks = [(1, 1, 'normal'), (2, 2, 'normal'), (3, 3, 'normal'), (4, 4, 'normal'), (4, 41, 'skewed'), (5, 5, 'normal'),
          (6, 6, 'normal'), (7, 7, 'normal'), (8, 8, 'skewed'), (1, 9, 'skewed'), (16, 16, 'normal'), (12, 12, 'normal')]
timed_cases = [(1, 5041226219440759413, 'normal'), (1, 7919, 'normal'), (1, 419, 'normal'), (1, 11, 'normal'),
               (8, 5041622730710591082, 'normal'), (8, 419, 'normal'), (8, 14, 'normal'), (4, 13, 'normal')]
allok = True
for bs, seed, dist in checks + timed_cases:
    data = R.generate_input(bs, 'ep8', seed, dist=dist)
    with torch.no_grad():
        ref = R.ref_kernel(data, w)
    nexp = len([e for e in torch.unique(data['topk_ids']).tolist() if 0 <= e < 32])
    graphs = {}
    for k, fn in fns.items():
        graphs[k] = capture(fn, data)
        for rep in range(3):
            graphs[k][0].replay()
        torch.cuda.synchronize()
        ok, msg = compare(graphs[k][1], ref, R.REL_L2_TOL)
        print('CHECK', k, bs, seed, dist, nexp, ok, msg, flush=True)
        allok &= bool(ok) or k == 'old'
    # eager path too (fresh output each call)
    o2 = fns['new'](data); torch.cuda.synchronize()
    ok2, msg2 = compare(o2, ref, R.REL_L2_TOL); allok &= bool(ok2)
    if not ok2: print('CHECK eager new FAIL', bs, seed, msg2, flush=True)
    if (bs, seed, dist) in timed_cases:
        keys = list(fns); T = {k: [] for k in keys}
        for i in range(60):
            for k in (keys if i % 2 == 0 else keys[::-1]):
                T[k].append(timed(graphs[k][0]))
        med = {k: round(statistics.median(v) - empty, 2) for k, v in T.items()}
        pr = sorted(a / b for a, b in zip(T['old'], T['new']))
        print('ROW', json.dumps({'bs': bs, 'seed': seed, 'experts': nexp, 'us_minus_empty': med,
                                 'gain_med': round(med['old'] / med['new'], 4), 'paired_ratio_med_raw': round(pr[len(pr) // 2], 4)}), flush=True)
    del graphs
print('ALLOK', allok, flush=True)
print('DONE', flush=True)
