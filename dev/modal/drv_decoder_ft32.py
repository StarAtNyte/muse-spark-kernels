"""Decoder whole-layer screen: WGS MoE with a 32-token fat-job tile (FT_T=32) for 17..32 tokens vs the scored build.
Reference-checked (check_implementation on the collected outputs); paired cold-L2 graph timing."""
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
assert S.MOE_CUDA_SRC.count('#define OKC_FT_T 64') == 1
S._MA_ALT_SRCS['ft32'] = S.MOE_CUDA_SRC.replace('#define OKC_FT_T 64', '#define OKC_FT_T 32')
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
orig_moe = S._moe
ft32 = S._moe_build(tuple(S._main_defines(S._MA_SFX)) + ('@ft32',))
def moe_ft32(M=None):
    return ft32 if (M is not None and 16 < M <= 32) else orig_moe(M)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d):
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for layer in ('sliding', 'full'):
    for seed in (3001, 3002, 3003):
        c = dict(batch=32, ctx=80000, layer=layer, page=1, dist='soft', spread='agentic', seed=seed)
        ls = R.case_fields(c, w)['local_start']
        d = R.generate_input(**c, local_start=ls)
        graphs = {}
        for name, fn in (('base', orig_moe), ('ft32', moe_ft32)):
            S._moe = fn
            g, o = cap(d)
            ok, msg = R.check_implementation(d, R.collect(d, o), w)
            if not ok: raise RuntimeError(f'{name} {layer} {seed}: {msg}')
            graphs[name] = g
        S._moe = orig_moe
        ts = {k: [] for k in graphs}
        for rep in range(30):
            for k in (('base', 'ft32') if rep % 2 == 0 else ('ft32', 'base')): ts[k].append(t1(graphs[k]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(layer=layer, seed=seed, **{k: round(v, 2) for k, v in med.items()},
              gain=round(med['base'] / med['ft32'], 4))), flush=True)
print('DONE', flush=True)
