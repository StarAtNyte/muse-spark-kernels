"""Decoder MoE floor diagnostic: per decode/verify shape, local routes, distinct experts, layer time,
isolated moe_fused time (cold L2) and pure-read floor of the active experts' weights."""
import importlib.util, json, statistics, sys, os
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
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

rec = []
orig_moe = S._moe
class Proxy:
    def __init__(self, e): self.e = e
    def __getattr__(self, k): return getattr(self.e, k)
    def okc_run(self, *a):
        rec.append((self.e, a))
        return self.e.okc_run(*a)
S._moe = lambda M=None: Proxy(orig_moe(M))

w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
print('READY', flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def med(g, n=30):
    ts = []
    for _ in range(n):
        flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
EXP_BYTES = (w['w13'][0].numel() * w['w13'].element_size() + w['w13_sf'][0].numel() * w['w13_sf'].element_size()
             + w['w2'][0].numel() * w['w2'].element_size() + w['w2_sf'][0].numel() * w['w2_sf'].element_size())
print('EXPERT_MB', EXP_BYTES / 1e6, flush=True)
cases = [dict(batch=b, ctx=80000, layer=l, page=1, dist='soft', q_len=q, spread=('agentic' if b > 1 else 'fixed'))
         for q in (1, 8) for b in (1, 16, 32) for l in ('sliding', 'full')]
for i, c in enumerate(cases):
    c['seed'] = 1000 + i
    ls = R.case_fields(c, w)['local_start']
    d = R.generate_input(**c, local_start=ls)
    rec.clear()
    out = S.kernel_fn(d, st)
    ids = out[2].long()
    loc = ids[(ids >= ls) & (ids < ls + 32)] - ls
    routes, distinct = int(loc.numel()), int(loc.unique().numel())
    counts = torch.bincount(loc, minlength=32).tolist()
    g = cap(lambda: S.kernel_fn(d, st)); layer_us = med(g)
    moe_us = None
    if rec:
        e, a = rec[-1]
        gm = cap(lambda: e.okc_run(*a)); moe_us = med(gm)
    floor_us = distinct * EXP_BYTES / 4.8e6
    print('ROW', json.dumps(dict(case=f"{'verify' if c['q_len']==8 else 'decode'} {c['batch']} {c['layer']}", routes=routes,
          distinct=distinct, hottest=max(counts), layer_us=round(layer_us, 1), moe_us=moe_us and round(moe_us, 1),
          moe_floor48_us=round(floor_us, 1), moe_frac=moe_us and round(floor_us / moe_us, 2))), flush=True)
print('DONE', flush=True)
