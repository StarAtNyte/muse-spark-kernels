"""Paired cold-L2 graph timing of build variants of the candidate (/w/run/sub.py) against the baseline kernel_fn
(kernels/muse_moe_w4a16/submission.py == moe_keell0renz_compare_v1). Variants swap -DOKC_SEG2 / -DOKC_DF in the TINY /
bs8 (t2_wsi) define tuples and launch through the same okc_run with the same state; correctness vs the FP32 reference."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
OLD = load('old_sub', '/w/kernels/muse_moe_w4a16/submission.py')
NEW = load('new_sub', '/w/run/sub.py')
w = R.generate_weights(R.WEIGHTS_SEED)
st_old = OLD.prepare(w); st = NEW.prepare(w)
VARS = json.loads(os.environ.get('VARS', '{"s0d0": [0, 0], "s1d0": [1, 0]}'))
def defs(bs, seg, df):
    base = tuple(st['ps'][bs]) if bs in st['ps'] else tuple(NEW.TINY_DEFINES if bs <= NEW.TINY_MAX_M else NEW.TINY2_DEFINES) + st['sfxs']
    out = tuple(d for d in base if not d.startswith(('-DOKC_SEG2=', '-DOKC_DF=')))
    return out + (f'-DOKC_SEG2={seg}', f'-DOKC_DF={df}')
def mk(bs, seg, df):
    ext = NEW._ext(defs(bs, seg, df))
    def f(d):
        x = d['hidden_states'].contiguous(); ids = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
        out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
        ext.okc_run(st['w'], st['scratch'], x, ids, tw, out, st['lo'], st['cnt'], st['grid'])
        return out
    return f
flush = torch.empty(512 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
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
cases = [(1, 5041226219440759413, 'normal', True), (1, 7919, 'normal', True), (8, 5041622730710591082, 'normal', True)]
allok = {k: True for k in VARS}
for bs, seed, dist, timeit in cases:
    data = R.generate_input(bs, 'ep8', seed, dist=dist)
    with torch.no_grad():
        ref = R.ref_kernel(data, w)
    nexp = len([e for e in torch.unique(data['topk_ids']).tolist() if 0 <= e < 32])
    fns = {'old': lambda d: OLD.kernel_fn(d, st_old)}
    for k, (sg, df) in VARS.items():
        fns[k] = mk(bs, sg, df)
    # control: the baseline module's own build, launched with the candidate's state (separates code from placement)
    od = tuple(st_old['ps'][bs]) if bs in st_old['ps'] else tuple(OLD.TINY_DEFINES if bs <= OLD.TINY_MAX_M else OLD.TINY2_DEFINES) + st_old['sfxs']
    oext = OLD._ext(od)
    def f_ostate(d, oext=oext):
        x = d['hidden_states'].contiguous(); ids = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
        out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
        oext.okc_run(st['w'], st['scratch'], x, ids, tw, out, st['lo'], st['cnt'], st['grid'])
        return out
    def f_oown(d, oext=oext):
        x = d['hidden_states'].contiguous(); ids = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
        out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
        oext.okc_run(st_old['w'], st_old['scratch'], x, ids, tw, out, st_old['lo'], st_old['cnt'], st_old['grid'])
        return out
    fns['oldK_newSt'] = f_ostate
    fns['oldK_direct'] = f_oown
    graphs = {}
    for k, fn in fns.items():
        graphs[k] = capture(fn, data)
        for rep in range(4):
            graphs[k][0].replay()
        torch.cuda.synchronize()
        ok, msg = compare(graphs[k][1], ref, R.REL_L2_TOL)
        if k in allok: allok[k] &= bool(ok)
        if not ok or k == 'old': print('CHECK', k, bs, seed, dist, nexp, ok, msg, flush=True)
    for k in VARS:  # eager, fresh output (DF sentinel restore across launches)
        o = fns[k](data); torch.cuda.synchronize(); ok, msg = compare(o, ref, R.REL_L2_TOL); allok[k] &= bool(ok)
        if not ok: print('CHECK eager FAIL', k, bs, seed, msg, flush=True)
    if timeit:
        keys = list(fns); T = {k: [] for k in keys}
        for i in range(50):
            for k in (keys if i % 2 == 0 else keys[::-1]):
                T[k].append(timed(graphs[k][0]))
        med = {k: round(statistics.median(v) - empty, 2) for k, v in T.items()}
        print('ROW', json.dumps({'bs': bs, 'seed': seed, 'experts': nexp, 'us': med,
                                 'gain': {k: round(med['old'] / med[k], 4) for k in keys if k != 'old'}}), flush=True)
    del graphs
print('ALLOK', allok, flush=True)
print('DONE', flush=True)
