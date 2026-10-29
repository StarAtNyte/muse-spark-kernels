"""Decoder: which MoE build per token count (MID vs WGS at 16 and 32 tokens), whole layer, reference-checked,
paired cold-L2 graph replays."""
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
orig = S._moe
mid, main = S._moe_mid(), orig(10 ** 6)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d):
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for bsz in (16, 32):
    for layer in ('sliding', 'full'):
        for seed in (7001, 7002):
            c = dict(batch=bsz, ctx=80000, layer=layer, page=1, dist='soft', spread='agentic', seed=seed + bsz)
            ls = R.case_fields(c, w)['local_start']
            d = R.generate_input(**c, local_start=ls)
            graphs = {}
            for name, ext in (('scored', None), ('mid', mid), ('wgs', main)):
                S._moe = orig if ext is None else (lambda M=None, e=ext: e if M == bsz else orig(M))
                g, o = cap(d)
                ok, msg = R.check_implementation(d, R.collect(d, o), w)
                if not ok: raise RuntimeError(f'{name} {c}: {msg}')
                graphs[name] = g
            S._moe = orig
            names = list(graphs); ts = {k: [] for k in names}
            for rep in range(24):
                for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
            med = {k: statistics.median(v) for k, v in ts.items()}
            print('ROW', json.dumps(dict(T=bsz, layer=layer, seed=seed, **{k: round(v, 1) for k, v in med.items()},
                  **{f'gain_{k}': round(med['scored'] / v, 4) for k, v in med.items() if k != 'scored'})), flush=True)
print('DONE', flush=True)
