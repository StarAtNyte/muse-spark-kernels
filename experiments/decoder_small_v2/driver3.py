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

def gtime(fn, n=30):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize()
    return round(statistics.median(t1(g) for _ in range(n)), 2)
dummy = torch.zeros(1, device='cuda')
log('empty-ish graph (1 tiny kernel)', gtime(lambda: dummy.add_(1)))
for E in (1, 2, 3, 4, 8):
    buf = torch.empty(int(E * 28.31e6 // 4), device='cuda')
    log('stream ref torch.sum bytes', E * 28.31e6, 'us', gtime(lambda: buf.sum()))
    del buf
builds = {'tiny': S._moe(1, st), 'mid': S._moe(8, st), 'main': S._moe(128, st)}
for k, e in builds.items():
    e.okc_set_pdl(0)
def moe_case(T, routes):   # routes: list per token of local expert ids
    xm = (torch.randn(T, 4096, device='cuda') * 0.5).to(torch.bfloat16)
    lids = torch.full((T, 8), -1, dtype=torch.int32, device='cuda')
    wts = torch.zeros(T, 8, device='cuda')
    for t, r in enumerate(routes):
        for j, e in enumerate(r):
            lids[t, j] = e; wts[t, j] = 0.125
    out = torch.empty(T, 4096, device='cuda', dtype=torch.bfloat16)
    return xm, lids, wts, out
res = {}
for T, routes, tag in [(1, [[3]], 'T1E1'), (1, [[3, 17]], 'T1E2'), (1, [[3, 17, 29]], 'T1E3'), (1, [[3, 17, 29, 8]], 'T1E4'),
                       (1, [[0, 1, 2, 3, 4, 5, 6, 7]], 'T1E8'),
                       (8, [[3, 17], [3], [3, 9], [17, 21], [3], [5, 9], [3, 17], [2]], 'T8E7r13'),
                       (8, [[3]] * 8, 'T8E1r8'), (8, [[3, 17]] * 8, 'T8E2r16')]:
    xm, lids, wts, out = moe_case(T, routes)
    row = {}
    for k, e in builds.items():
        try:
            row[k] = gtime(lambda: e.okc_run(st['ma_w'], st['ma_scratch'], xm, lids, wts, out, 0, 32, st['nsm']))
            row[k + '_sum'] = float(out.float().abs().sum())
        except Exception as ex:
            row[k] = 'ERR ' + repr(ex)[:120]
    print('MOE', tag, json.dumps(row), flush=True)
# small-kernel latencies
x1 = torch.randn(1, 4096, device='cuda').to(torch.bfloat16)
log('unpooler sk2 T1', gtime(lambda: S.dense(x1, st['unpool_slice'], 1, st, 'un')))
x8 = torch.randn(8, 4096, device='cuda').to(torch.bfloat16)
log('unpooler sk2 T8', gtime(lambda: S.dense(x8, st['unpool_slice'], 8, st, 'un')))
log('DONE')
