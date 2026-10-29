"""Decoder T=32: WGS MoE knob variants (designated finisher, unit cost tables) via alt sources, whole layer,
reference-checked, paired cold-L2 graph replays."""
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
src = S.MOE_CUDA_SRC
A = '#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207'
Bq = '#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174'
D = '#define OKC_WGSDFIN 2'
for s_ in (A, Bq, D): assert src.count(s_) == 1, s_
def scale(line, f):
    head, vals = line.split('OKC_WGS')[0], line.split(' ', 2)[2]
    nums = [int(round(int(v) * f)) for v in vals.split(',')]
    return line.split(' ', 2)[0] + ' ' + line.split(' ', 2)[1] + ' ' + ', '.join(map(str, nums))
VAR = {'dfin0': src.replace(D, '#define OKC_WGSDFIN 0'), 'dfin1': src.replace(D, '#define OKC_WGSDFIN 1'),
       'dfin3': src.replace(D, '#define OKC_WGSDFIN 3'),
       'a085': src.replace(A, scale(A, 0.85)), 'a115': src.replace(A, scale(A, 1.15)),
       'b085': src.replace(Bq, scale(Bq, 0.85)), 'b115': src.replace(Bq, scale(Bq, 1.15))}
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
orig = S._moe
exts = {}
for k, v in VAR.items():
    S._MA_ALT_SRCS[k] = v
    try: exts[k] = S._moe_build(tuple(S._main_defines(S._MA_SFX)) + ('@' + k,))
    except Exception as e: print('BUILD_FAIL', k, str(e)[-200:], flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d):
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for layer in ('sliding', 'full'):
    for seed in (8001, 8002):
        c = dict(batch=32, ctx=80000, layer=layer, page=1, dist='soft', spread='agentic', seed=seed)
        ls = R.case_fields(c, w)['local_start']
        d = R.generate_input(**c, local_start=ls)
        graphs = {}
        for name, ext in [('base', None)] + list(exts.items()):
            S._moe = orig if ext is None else (lambda M=None, e=ext: e if (M is not None and 16 < M <= 32) else orig(M))
            g, o = cap(d)
            ok, msg = R.check_implementation(d, R.collect(d, o), w)
            if not ok: print('CHECK_FAIL', name, layer, seed, msg, flush=True); continue
            graphs[name] = g
        S._moe = orig
        names = list(graphs); ts = {k: [] for k in names}
        for rep in range(24):
            for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(layer=layer, seed=seed, base=round(med['base'], 1),
              **{f'g_{k}': round(med['base'] / v, 4) for k, v in med.items() if k != 'base'})), flush=True)
print('DONE', flush=True)
