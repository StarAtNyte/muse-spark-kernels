"""Projection: qkv M256 and unpooler M128 through wide2 (candidate kernel_fn) vs torch.mm (what the scored stack's
SGLang fallback runs: F.linear). Reference-checked; paired cold-L2 graphs."""
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
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
W = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(W)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(fn):
    o = fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = fn()
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for m, layer in ((256, 'qkv_gate'), (128, 'unpooler')):
    for seed in (31, 32):
        d = R.generate_input(m, layer, seed)
        n = d['config']['n']; wt = W[layer][:n]
        gc, oc = cap(lambda: S.kernel_fn(d, st))
        if not torch.is_tensor(oc): print('ROW', json.dumps(dict(m=m, layer=layer, error='candidate not routed')), flush=True); continue
        gb, ob = cap(lambda: torch.mm(d['x'], wt.t()))
        ok, msg = R.check_implementation(d, oc, W)
        tb, tc = [], []
        for rep in range(30):
            if rep % 2: tc.append(t1(gc)); tb.append(t1(gb))
            else: tb.append(t1(gb)); tc.append(t1(gc))
        print('ROW', json.dumps(dict(m=m, layer=layer, seed=seed, ok=ok, cublas=round(statistics.median(tb), 2),
              wide2=round(statistics.median(tc), 2), gain=round(statistics.median(tb) / statistics.median(tc), 4), msg=msg[:80])), flush=True)
print('DONE', flush=True)
