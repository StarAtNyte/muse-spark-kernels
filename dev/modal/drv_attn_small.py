"""Scored attention on the latency-bound shapes, timed exactly like the gather-floor probe (cold L2, CUDA graph,
event-timed median) so the two are directly comparable."""
import importlib.util, json, statistics, sys
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
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def med(fn, n=30):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(n):
        flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
z = torch.zeros(4, device='cuda')
print('EMPTY_US', round(med(lambda: z.add_(1)), 2), flush=True)
for c in (dict(batch=1, ctx=80000, layer='sliding', page=1, dist='soft', heads='tp8'),
          dict(batch=1, ctx=80000, layer='full', page=1, dist='coherent', heads='tp8'),
          dict(batch=1, ctx=80000, layer='sliding', page=1, dist='soft', heads='tp8', q_len=8),
          dict(batch=1, ctx=80000, layer='full', page=1, dist='coherent', heads='tp8', q_len=8)):
    d = R.generate_input(**c, seed=5)
    print('ROW', json.dumps(dict(case=c, us=round(med(lambda: S.kernel_fn(d)), 2))), flush=True)
print('DONE', flush=True)
