"""decoder_small_v2 round 1: (a) per-kernel timeline of the parent graph at T=1 / T=8 (CUPTI via torch.profiler, cold L2),
(b) standalone cold-L2 times of the T=1 qkv / unpooler-slice kernels vs a 12-bit cgemv, (c) in-layer paired A/B at T=1."""
import importlib.util, json, statistics, sys, threading, time, os
from unittest.mock import MagicMock
import torch
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_decode_layer'); sys.path.insert(0, '/w/run')
import reference as R
import cg_ext as CG
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
box = {}
def build():
    try: CG.cg_ext(); box['ok'] = 1; log('cg ext built')
    except Exception as ex: box['err'] = repr(ex)[:2000]; log('cg build FAILED', box['err'])
th = threading.Thread(target=build); th.start()
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
th.join(); log('PREPARED')
assert 'ok' in box, box
for k, v in w.items() if isinstance(w, dict) else []:
    if hasattr(v, 'shape'): print('W', k, tuple(v.shape), v.dtype, v.numel() * v.element_size())
PK = {'qkv2': CG.cg_compress(st['qkv'], 2), 'un2': CG.cg_compress(st['unpool_slice'], 2)}
for k, v in PK.items(): log('pack', k, None if v is None else (v['pmax'], tuple(v['L'].shape)))
MODE = {'qkv': None, 'un': None}   # qkv: (nth,) ; un: (nth, pdl)
_orig_gemm = S._bfp_gemm
def _gemm(stt, name, x, m, n, k):
    if name == 'qkv' and m == 1 and MODE['qkv'] is not None:
        return CG.cg_run(PK['qkv2'], x, n, MODE['qkv'][0], 0)
    return _orig_gemm(stt, name, x, m, n, k)
S._bfp_gemm = _gemm
_orig_dense = S.dense
def _dense(x, wt, T, state=None, name=None):
    if name == 'un' and T == 1 and MODE['un'] is not None and wt.shape[0] == 1024:
        return CG.cg_run(PK['un2'], x, 1024, MODE['un'][0], MODE['un'][1]), 1
    return _orig_dense(x, wt, T, state, name)
S.dense = _dense
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d):
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
def timeline(g, n=12):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            flush.add_(1); torch.cuda.synchronize(); time.sleep(0.002); g.replay(); torch.cuda.synchronize(); time.sleep(0.002)
    path = '/w/run/tr.json'; prof.export_chrome_trace(path)
    ev = [e for e in json.load(open(path))['traceEvents'] if e.get('cat') == 'kernel']
    ev.sort(key=lambda e: e['ts'])
    groups, cur = [], []
    for e in ev:
        if cur and e['ts'] - (cur[-1]['ts'] + cur[-1]['dur']) > 500: groups.append(cur); cur = []
        cur.append(e)
    groups.append(cur)
    groups = [gp for gp in groups if len(gp) > 1]
    rows = {}
    for gp in groups:
        t0 = gp[0]['ts']
        for i, e in enumerate(gp):
            rows.setdefault(i, []).append((e['name'][:48], e['ts'] - t0, e['ts'] - t0 + e['dur']))
    out = []
    for i in sorted(rows):
        r = rows[i]
        out.append((r[0][0], round(statistics.median(x[1] for x in r), 2), round(statistics.median(x[2] for x in r), 2)))
    return out
CASES = {'d1s': dict(batch=1, ctx=80000, layer='sliding', page=1, dist='soft'),
         'd1f': dict(batch=1, ctx=80000, layer='full', page=1, dist='soft'),
         'v1s': dict(batch=1, ctx=80000, layer='sliding', page=1, q_len=8, dist='soft'),
         'v1f': dict(batch=1, ctx=80000, layer='full', page=1, q_len=8, dist='soft')}
def mk(name, seed):
    c = dict(CASES[name], seed=seed); ls = R.case_fields(c, w)['local_start']; return R.generate_input(**c, local_start=ls)
# (a) timelines
for nm in CASES:
    d = mk(nm, 9100)
    g, o = cap(d)
    ok, msg = R.check_implementation(d, R.collect(d, o), w)
    ids = o[2]; ls = int(d['local_start'].item()) if hasattr(d['local_start'], 'item') else d['local_start']
    loc = ((ids >= ls) & (ids < ls + 32))
    tl = timeline(g)
    print('TL', nm, json.dumps(dict(ok=bool(ok), routes=int(loc.sum()), experts=int(torch.unique(ids[loc]).numel()),
          graph_us=round(statistics.median(t1(g) for _ in range(15)), 2), kernels=tl)), flush=True)
    del g, o; torch.cuda.empty_cache()
log('timelines done')
# (b) standalone kernels, cold L2, graph of one kernel
x1 = torch.randn(1, 8192, device='cuda').to(torch.bfloat16); m1 = torch.randn(1, 4096, device='cuda').to(torch.bfloat16)
ref_q = (x1.float() @ st['qkv'].float().t()); ref_u = (m1.float() @ st['unpool_slice'].float().t())
def one(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = fn()
    g.replay(); torch.cuda.synchronize()
    return round(statistics.median(t1(g) for _ in range(30)), 2), o
def relerr(o, r): return float((o.float() - r).norm() / r.norm())
for lbl, fn, r in [('qkv sk2', lambda: _orig_gemm(st, 'qkv', x1, 1, 2304, 8192), ref_q),
                   ('qkv cg2x256', lambda: CG.cg_run(PK['qkv2'], x1, 2304, 256, 0), ref_q),
                   ('qkv cg2x512', lambda: CG.cg_run(PK['qkv2'], x1, 2304, 512, 0), ref_q),
                   ('un sk2', lambda: _orig_dense(m1, st['unpool_slice'], 1, st, 'un')[0], ref_u),
                   ('un cg2x128', lambda: CG.cg_run(PK['un2'], m1, 1024, 128, 0), ref_u),
                   ('un cg2x256', lambda: CG.cg_run(PK['un2'], m1, 1024, 256, 0), ref_u)]:
    try:
        us, o = one(fn); print('SA', lbl, us, 'relerr', relerr(o, r), flush=True)
    except Exception as ex: print('SA', lbl, 'ERR', repr(ex)[:300], flush=True)
# (c) in-layer A/B at T=1
VARS = {'base': (None, None), 'q256': ((256,), None), 'q512': ((512,), None), 'u128p': (None, (128, 1)),
        'q256u128p': ((256,), (128, 1))}
for nm, seed in (('d1s', 9201), ('d1f', 9202), ('d1s', 9203), ('d1f', 9204)):
    if time.time() - T0 > 780: log('time guard'); break
    d = mk(nm, seed); gs = {}; chk = {}
    for v, (q, u) in VARS.items():
        MODE['qkv'], MODE['un'] = q, u
        g, o = cap(d); gs[v] = g
        ok, msg = R.check_implementation(d, R.collect(d, o), w); chk[v] = bool(ok)
    MODE['qkv'] = MODE['un'] = None
    ts = {v: [] for v in VARS}; names = list(VARS)
    for rep in range(24):
        order = names if rep % 2 == 0 else names[::-1]
        for v in order: ts[v].append(t1(gs[v]))
    med = {v: statistics.median(ts[v]) for v in VARS}
    print('AB', nm, seed, json.dumps({v: dict(us=round(med[v], 2), ratio=round(med['base'] / med[v], 4), ok=chk[v]) for v in VARS}), flush=True)
    del gs; torch.cuda.empty_cache()
log('DONE')
