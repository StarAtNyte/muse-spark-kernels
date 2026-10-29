"""Whole-decoder critical-path ablation on H200: replay the scored layer graph with one component's launches removed
(its outputs replaced by a cached tensor from a real call, so later kernels still run on valid shapes). The time that
disappears bounds what fusing or speeding up that component can save. Outputs of ablated variants are wrong by design."""
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
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
print('READY', flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')

def cached(fn):
    cache = {}
    def sig(v):
        return (tuple(v.shape), v.dtype) if torch.is_tensor(v) else (v if isinstance(v, (int, str, bool, type(None))) else type(v).__name__)
    def wrap(*a, **k):
        key = tuple(sig(v) for v in a) + tuple((n, sig(v)) for n, v in sorted(k.items()))
        if key not in cache:
            cache[key] = fn(*a, **k)
        return cache[key]
    return wrap
class Skip:
    def __getitem__(self, grid): return lambda *a, **k: None
class MoeSkip:
    def __init__(self, e): self.e = e
    def okc_run(self, *a): return None
    def okc_set_pdl(self, *a): return None
    def __getattr__(self, k): return getattr(self.e, k)
ORIG = {n: getattr(S, n) for n in ('_boundary1', '_boundary2', '_route_split', '_route', '_moe', '_decode_attention7',
                                    'dense', '_bfp_gemm', 'bfp_fused_pooler_router')}
def apply(var):
    for n, f in ORIG.items(): setattr(S, n, f)
    if var in ('no_glue', 'only_moe'):
        for n in ('_boundary1', '_boundary2', '_route_split', '_route'): setattr(S, n, Skip())
    if var in ('no_moe',):
        S._moe = lambda M=None: MoeSkip(ORIG['_moe'](M))
    if var in ('no_attn', 'only_moe'):
        S._decode_attention7 = cached(ORIG['_decode_attention7'])
    if var in ('no_dense', 'only_moe'):
        S.dense = cached(ORIG['dense']); S._bfp_gemm = cached(ORIG['_bfp_gemm'])
        S.bfp_fused_pooler_router = cached(ORIG['bfp_fused_pooler_router'])
def cap(d):
    S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
VARS = ['base', 'no_glue', 'no_moe', 'no_attn', 'no_dense', 'only_moe']
cases = [dict(batch=b, ctx=80000, layer=l, page=1, dist='soft', q_len=q, spread=('agentic' if b > 1 else 'fixed'))
         for (b, q, l) in ((1, 1, 'sliding'), (16, 1, 'sliding'), (32, 1, 'sliding'), (1, 8, 'sliding'), (32, 1, 'full'), (16, 8, 'sliding'))]
for i, c in enumerate(cases):
    c['seed'] = 2000 + i
    ls = R.case_fields(c, w)['local_start']
    d = R.generate_input(**c, local_start=ls)
    graphs = {}
    for v in VARS:
        apply(v); graphs[v] = cap(d)
    apply('base')
    ts = {v: [] for v in VARS}
    for rep in range(24):
        for v in (VARS if rep % 2 == 0 else VARS[::-1]): ts[v].append(t1(graphs[v]))
    med = {v: statistics.median(x) for v, x in ts.items()}
    name = f"{'verify' if c['q_len'] == 8 else 'decode'} {c['batch']} {c['layer']}"
    print('ROW', json.dumps(dict(case=name, **{v: round(x, 1) for v, x in med.items()},
          **{f'saves_{v}': round(med['base'] - x, 1) for v, x in med.items() if v != 'base'})), flush=True)
print('DONE', flush=True)
