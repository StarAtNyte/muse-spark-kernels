"""gle8 paired screen: ONE candidate file (/w/run/sub.py), one prepare; variants toggled by module globals at capture.
ctrl = GLE8_ENABLED False (the parent's T = 8 b1 path), c1 / c2 = gle path at T = 8 with GLE8_NCH 1 / 2.
Correctness on the 13 TASK.md tests (every variant at T = 8 tests), then rotated cold-L2 graph replays."""
import importlib.util, json, os, statistics, sys, time
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
t0 = time.time()
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
w = R.generate_weights(R.WEIGHTS_SEED)
st = C.prepare(w)
print('PREPARED', round(time.time() - t0, 1), flush=True)
VARS = {'ctrl': dict(GLE8_ENABLED=False, GLE8_NCH=1), 'c1': dict(GLE8_ENABLED=True, GLE8_NCH=1),
        'c2': dict(GLE8_ENABLED=True, GLE8_NCH=2), 'c8': dict(GLE8_ENABLED=True, GLE8_NCH=8)}
def setv(v):
    for k, val in VARS[v].items(): setattr(C, k, val)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d, v):
    setv(v)
    o = C.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = C.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
def mk(cc):
    ls = R.case_fields(cc, w)['local_start']
    return R.generate_input(**cc, local_start=ls)
# which path T = 8 takes (instrument the gle entry)
calls = {'gle': 0}
_orig = C._gle_after_attention
def _spy(*a, **k):
    calls['gle'] += 1; return _orig(*a, **k)
C._gle_after_attention = _spy

from torch.profiler import profile, ProfilerActivity
for layer in ('sliding',):
    cc = dict(batch=1, q_len=8, layer=layer, ctx=80000, page=1, dist='soft', spread='fixed', seed=4001)
    d = mk(cc)
    for v in ('ctrl', 'c1', 'c8'):
        g, o = cap(d, v)
        for _ in range(3): t1(g)
        tt = statistics.median([t1(g) for _ in range(15)])
        flush.add_(1); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            g.replay(); torch.cuda.synchronize()
        evs = [e for e in prof.events() if e.device_type.name == 'CUDA' and 'flush' not in e.name]
        evs.sort(key=lambda e: e.time_range.start)
        t0e = evs[0].time_range.start
        print('PROF', v, layer, 'median_us', round(tt, 1), flush=True)
        for e in evs:
            print('  K %8.2f %7.2f  %s' % (e.time_range.start - t0e, e.time_range.end - e.time_range.start, e.name[:90]), flush=True)
print('DONE', flush=True)
