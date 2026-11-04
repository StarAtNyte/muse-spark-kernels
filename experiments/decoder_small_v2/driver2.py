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
import pf_ext as PF
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
box = {}
def build():
    try: PF.pf_ext(); box['ok'] = 1; log('pf ext built')
    except Exception as ex: box['err'] = repr(ex)[:2000]; log('cg build FAILED', box['err'])
th = threading.Thread(target=build); th.start()
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
th.join(); log('PREPARED')
assert 'ok' in box, box
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

PFC = {'cfg': None, 'pending': False, 'n': 0}
SIDE = torch.cuda.Stream()
def _unpool_stream(T):
    return st['bfp']['sk2'][S.bfp__sk2_name('unpooler', T)][1]
class _Proxy:
    def __init__(self, e): self._e = e
    def __getattr__(self, k): return getattr(self._e, k)
    def okc_run(self, *a):
        xm = a[2]; T = xm.shape[0]
        if PFC['cfg'] is not None and T <= 8:
            cur = torch.cuda.current_stream(); SIDE.wait_stream(cur)
            with torch.cuda.stream(SIDE):
                PF.prefetch_l2(_unpool_stream(T), **PFC['cfg'])
            PFC['pending'] = True; PFC['n'] += 1
        return self._e.okc_run(*a)
_orig_moe = S._moe
def _moe(M=None, state=None):
    return _Proxy(_orig_moe(M, state))
S._moe = _moe
_orig_dense = S.dense
def _dense(x, wt, T, state=None, name=None):
    r = _orig_dense(x, wt, T, state, name)
    if name == 'un' and PFC['pending']:
        torch.cuda.current_stream().wait_stream(SIDE); PFC['pending'] = False
    return r
S.dense = _dense
CASES = {'d1s': dict(batch=1, ctx=80000, layer='sliding', page=1, dist='soft'),
         'd1f': dict(batch=1, ctx=80000, layer='full', page=1, dist='soft'),
         'v1s': dict(batch=1, ctx=80000, layer='sliding', page=1, q_len=8, dist='soft'),
         'v1f': dict(batch=1, ctx=80000, layer='full', page=1, q_len=8, dist='soft')}
def mk(name, seed):
    c = dict(CASES[name], seed=seed); ls = R.case_fields(c, w)['local_start']; return R.generate_input(**c, local_start=ls)

log('unpooler stream bytes T1', _unpool_stream(1).numel() * _unpool_stream(1).element_size(), 'T8', _unpool_stream(8).numel() * _unpool_stream(8).element_size())
VARS = {'base': None, 'pf0h': dict(delay_ns=0, hint=1), 'pf0n': dict(delay_ns=0, hint=0),
        'pf8h': dict(delay_ns=8000, hint=1), 'pf15h': dict(delay_ns=15000, hint=1)}
for nm in ('d1s', 'v1s'):
    d = mk(nm, 9300)
    for v in ('base', 'pf0h'):
        PFC['cfg'] = VARS[v]; g, o = cap(d); PFC['cfg'] = None
        print('TL', nm, v, json.dumps(timeline(g)), flush=True)
        del g
allr = {}
for nm, seed in (('d1s', 9301), ('d1f', 9302), ('v1s', 9303), ('v1f', 9304), ('d1s', 9305), ('d1f', 9306), ('v1s', 9307), ('v1f', 9308)):
    if time.time() - T0 > 800: log('time guard'); break
    d = mk(nm, seed); gs = {}; chk = {}
    for v, c in VARS.items():
        PFC['cfg'] = c; n0 = PFC['n']
        g, o = cap(d); gs[v] = g
        PFC['cfg'] = None
        ok, msg = R.check_implementation(d, R.collect(d, o), w); chk[v] = (bool(ok), PFC['n'] - n0)
    ts = {v: [] for v in VARS}; names = list(VARS)
    for rep in range(30):
        order = names if rep % 2 == 0 else names[::-1]
        for v in order: ts[v].append(t1(gs[v]))
    med = {v: statistics.median(ts[v]) for v in VARS}
    pr = {v: statistics.median(b / x for b, x in zip(ts['base'], ts[v])) for v in VARS}
    for v in VARS: allr.setdefault((nm[:2], v), []).append(pr[v])
    print('AB', nm, seed, json.dumps({v: dict(us=round(med[v], 2), pratio=round(pr[v], 4), ok=chk[v]) for v in VARS}), flush=True)
    del gs; torch.cuda.empty_cache()
for k, v in allr.items(): print('GM', k, round(statistics.geometric_mean(v), 4))
log('DONE')
