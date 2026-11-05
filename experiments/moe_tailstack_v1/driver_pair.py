"""Paired cold-L2 graph timing on H200: candidate (/w/run/sub.py as given) vs a same-code control (the same file with
OKC_TFIN=0 and the baseline bs8 build, i.e. the moe_keell0renz_compare_v1 kernels), each with its own prepare/scratch.
Correctness of both against the FP32 reference, including after the timed replays (parity flips every launch)."""
import importlib.util, json, math, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
NEW = load('new_sub', '/w/run/sub.py')
CTL = load('ctl_sub', '/w/run/sub.py')
def to_ctl(t):
    return tuple(d.replace('-DOKC_TFIN=1', '-DOKC_TFIN=0') for d in t)
CTL.TINY_DEFINES = to_ctl(CTL.TINY_DEFINES); CTL.TINY2_DEFINES = to_ctl(CTL.TINY2_DEFINES); CTL.MID_DEFINES = to_ctl(CTL.MID_DEFINES)
CTL.PS_SHAPES = {8: ('tiny2', json.loads(os.environ.get('CTL_PS8', '{}')), 't2_wsi')}
for k, v in (json.loads(os.environ['NEW_PS8']).items() if os.environ.get('NEW_PS8') else []):
    NEW.PS_SHAPES[8][1][k] = v
print('START', NEW.TINY_DEFINES, CTL.TINY_DEFINES, NEW.PS_SHAPES, CTL.PS_SHAPES, flush=True)
t0 = time.time()
w = R.generate_weights(R.WEIGHTS_SEED)
st_ctl = CTL.prepare(w); print('prep ctl', round(time.time() - t0), flush=True)
st_new = NEW.prepare(w); print('prep new', round(time.time() - t0), flush=True)
flush = torch.empty(512 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
fns = {'ctl': lambda d: CTL.kernel_fn(d, st_ctl), 'new': lambda d: NEW.kernel_fn(d, st_new)}
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
checks = [(4, 1, 'normal'), (33, 2, 'normal'), (257, 3, 'normal'), (128, 4, 'skewed'), (1, 5, 'normal'),
          (2, 6, 'normal'), (3, 7, 'normal'), (5, 8, 'normal'), (6, 9, 'normal'), (7, 10, 'normal'), (8, 11, 'skewed'),
          (9, 12, 'normal'), (12, 13, 'normal'), (15, 14, 'skewed'), (16, 15, 'skewed'), (4, 41, 'skewed'), (1, 9, 'skewed')]
REPS = int(os.environ.get('REPS', '60'))
timed_cases = [(1, 5041226219440759413, 'normal'), (1, 7919, 'normal'), (1, 419, 'normal'),
               (16, 101, 'normal'), (16, 202, 'normal'), (32, 303, 'normal'),
               (8, 5041622730710591082, 'normal'), (8, 419, 'normal'), (8, 14, 'normal'),
               (128, 404, 'normal'), (256, 505, 'normal')]
allok = True
rows = []
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
        if not ok or k == 'new': print('CHECK', k, bs, seed, dist, nexp, ok, msg, flush=True)
        allok &= bool(ok) or k == 'ctl'
    o2 = fns['new'](data); torch.cuda.synchronize()
    ok2, msg2 = compare(o2, ref, R.REL_L2_TOL); allok &= bool(ok2)
    if not ok2: print('CHECK eager new FAIL', bs, seed, msg2, flush=True)
    if (bs, seed, dist) in timed_cases:
        keys = list(fns); T = {k: [] for k in keys}
        for i in range(REPS):
            for k in (keys if i % 2 == 0 else keys[::-1]):
                T[k].append(timed(graphs[k][0]))
        torch.cuda.synchronize()
        ok3, msg3 = compare(graphs['new'][1], ref, R.REL_L2_TOL); allok &= bool(ok3)
        if not ok3: print('CHECK after-timing new FAIL', bs, seed, msg3, flush=True)
        med = {k: round(statistics.median(v) - empty, 2) for k, v in T.items()}
        pr = sorted((a - empty) / (b - empty) for a, b in zip(T['ctl'], T['new']))
        row = {'bs': bs, 'seed': seed, 'experts': nexp, 'us': med, 'gain_med': round(med['ctl'] / med['new'], 4),
               'paired_med': round(pr[len(pr) // 2], 4)}
        rows.append(row)
        print('ROW', json.dumps(row), flush=True)
    del graphs
per = {}
for r in rows: per.setdefault(r['bs'], []).append(r['paired_med'])
g = {bs: math.exp(statistics.mean(math.log(x) for x in v)) for bs, v in per.items()}
print('PER_BS', json.dumps({k: round(v, 4) for k, v in g.items()}), 'GEOMEAN', round(math.exp(statistics.mean(math.log(v) for v in g.values())), 4), flush=True)
print('ALLOK', allok, flush=True)
print('DONE', flush=True)
