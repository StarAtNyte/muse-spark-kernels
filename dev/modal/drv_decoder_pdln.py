"""Decoder: sk2 weight stages issued before griddepcontrol.wait (BFP_PDL_N 2 = scored) vs 4/8/16, whole layer,
reference-checked, paired cold-L2 graph replays (variants captured from one module by switching the global)."""
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
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d):
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
V = (2, 4, 8, 16)
for i, (bsz, q, layer) in enumerate(((1, 1, 'sliding'), (1, 1, 'full'), (16, 1, 'sliding'), (32, 1, 'sliding'), (32, 1, 'full'), (1, 8, 'sliding'))):
    for seed in (6001, 6002):
        c = dict(batch=bsz, ctx=80000, layer=layer, page=1, dist='soft', q_len=q, spread=('agentic' if bsz > 1 else 'fixed'), seed=seed + 10 * i)
        ls = R.case_fields(c, w)['local_start']
        d = R.generate_input(**c, local_start=ls)
        graphs = {}
        for n in V:
            S.BFP_PDL_N = n
            g, o = cap(d)
            ok, msg = R.check_implementation(d, R.collect(d, o), w)
            if not ok: raise RuntimeError(f'pdl{n} {c}: {msg}')
            graphs[n] = g
        S.BFP_PDL_N = 2
        ts = {k: [] for k in V}
        for rep in range(24):
            for k in (V if rep % 2 == 0 else V[::-1]): ts[k].append(t1(graphs[k]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(case=f"{'verify' if q == 8 else 'decode'} {bsz} {layer}", seed=seed,
              **{f'pdl{k}': round(v, 1) for k, v in med.items()}, **{f'gain{k}': round(med[2] / v, 4) for k, v in med.items() if k != 2})), flush=True)
print('DONE', flush=True)
