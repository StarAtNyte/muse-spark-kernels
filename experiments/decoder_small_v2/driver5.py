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
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)

w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
log('PREPARED', 'ma_ext keys', list(st['ma_ext'].keys()))
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d):
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000


MAP = {}   # T -> build key
_orig_moe = S._moe
def _moe(M=None, state=None):
    if state is not None and M in MAP:
        return state['ma_ext'][MAP[M]]
    return _orig_moe(M, state)
S._moe = _moe
def cap(d):
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
CASES = {'d1s': dict(batch=1, ctx=80000, layer='sliding', page=1, dist='soft'),
         'd1f': dict(batch=1, ctx=80000, layer='full', page=1, dist='soft'),
         'v1s': dict(batch=1, ctx=80000, layer='sliding', page=1, q_len=8, dist='soft'),
         'v1f': dict(batch=1, ctx=80000, layer='full', page=1, q_len=8, dist='soft')}
def mk(name, seed):
    c = dict(CASES[name], seed=seed); ls = R.case_fields(c, w)['local_start']; return R.generate_input(**c, local_start=ls)

VARS = {'base': {}, 'c1': {1: 'mid'}, 'c18': {1: 'mid', 8: 'mid'}}
DEF = dict(S.MOE_SMALL_BUILD); log('file default', DEF)
allr = {}
for rnd in range(3):
    for nm in ('d1s', 'd1f', 'v1s', 'v1f'):
        seed = 9500 + 10 * rnd + list(CASES).index(nm)
        d = mk(nm, seed); gs = {}; chk = {}
        vs = list(VARS)
        for v in vs:
            S.MOE_SMALL_BUILD = dict(VARS[v]); g, o = cap(d); S.MOE_SMALL_BUILD = dict(DEF); gs[v] = g
            ok, msg = R.check_implementation(d, R.collect(d, o), w); chk[v] = bool(ok)
        ts = {v: [] for v in vs}
        for rep in range(30):
            for v in (vs if rep % 2 == 0 else vs[::-1]): ts[v].append(t1(gs[v]))
        pr = {v: statistics.median(b / x for b, x in zip(ts['base'], ts[v])) for v in vs}
        for v in vs: allr.setdefault((nm, v), []).append(pr[v])
        print('AB', nm, seed, json.dumps({v: dict(us=round(statistics.median(ts[v]), 2), pratio=round(pr[v], 4), ok=chk[v]) for v in vs}), flush=True)
        del gs; torch.cuda.empty_cache()
for k, v in allr.items(): print('GM', k, round(statistics.geometric_mean(v), 4), [round(x, 4) for x in v])
# the 13 TASK.md tests with the T1 variants
TESTS = [dict(batch=1, ctx=10, layer='sliding'), dict(batch=1, ctx=3000, layer='full'), dict(batch=8, ctx=2047, layer='sliding'),
         dict(batch=9, ctx=5000, layer='sliding', spread='mixed', page=1), dict(batch=3, ctx=6000, layer='full', q_len=8, dist='outliers'),
         dict(batch=5, ctx=5000, layer='sliding', q_len=8, spread='mixed', page=1), dict(batch=33, ctx=20000, layer='full', spread='mixed', page=1),
         dict(batch=2, ctx=300000, layer='sliding', q_len=8), dict(batch=1, ctx=1048576, layer='full', q_len=8),
         dict(batch=4, ctx=7000, layer='sliding', q_len=8, page=1, kv_scale=0.015625), dict(batch=1, ctx=80000, layer='full', dist='soft', page=1),
         dict(batch=4, ctx=6000, layer='sliding', q_len=8, dist='soft'), dict(batch=16, ctx=20000, layer='full', spread='mixed', dist='soft', page=1)]
for v in ('file_default',):
    S.MOE_SMALL_BUILD = dict(DEF); res = []
    for i, c in enumerate(TESTS):
        for seed in (11 + i, 777 + i):
            try:
                c2 = dict(c, seed=seed); ls = R.case_fields(c2, w)['local_start']; d = R.generate_input(**c2, local_start=ls)
                o = S.kernel_fn(d, st); torch.cuda.synchronize()
                ok, msg = R.check_implementation(d, R.collect(d, o), w); res.append(bool(ok))
                if not ok: print('FAIL', v, c2, str(msg)[:200], flush=True)
                del d, o
            except Exception as ex:
                res.append(False); print('EXC', v, c, repr(ex)[:200], flush=True)
        torch.cuda.empty_cache()
    print('TESTS', v, sum(res), '/', len(res), flush=True)
SCORED = [dict(batch=b, ctx=80000, layer=l, page=1, dist='soft', q_len=q, **({} if b == 1 else dict(spread='agentic')))
          for q in (1, 8) for b in (1, 16, 32) for l in ('sliding', 'full')]
res = []
for c in SCORED:
    c2 = dict(c, seed=4242); ls = R.case_fields(c2, w)['local_start']; d = R.generate_input(**c2, local_start=ls)
    g, o = cap(d); ok, msg = R.check_implementation(d, R.collect(d, o), w); res.append(bool(ok))
    if not ok: print('FAIL scored', c, str(msg)[:200], flush=True)
    del g, d, o; torch.cuda.empty_cache()
print('SCORED graph-checked', sum(res), '/', len(res), flush=True)
log('DONE')
