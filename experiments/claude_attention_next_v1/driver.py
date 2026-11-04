"""Paired screen: parent (/w/run/base.py) vs candidate (/w/run/sub.py), verify (q_len 8) full+sliding batch 16/32 and decode 32 full.
Reference-checked every case, alternating order, cold-L2 graph replays, 3 seeds."""
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
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base', '/w/run/base.py'); C = load('cand', '/w/run/sub.py')
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, d):
    o = M.kernel_fn(d); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
cases = [(32, 8, 'full', 'coherent'), (16, 8, 'full', 'coherent'), (32, 8, 'sliding', 'soft'), (32, 1, 'full', 'coherent')]
for seed in (101, 2027, 4093):
    for bsz, q, layer, dist in cases:
        c = dict(batch=bsz, ctx=80000, heads='tp8', layer=layer, page=1, dist=dist, q_len=q, spread='agentic')
        d = R.generate_input(**c, seed=seed + bsz + q)
        res = {}
        for name, M in (('ours', B), ('cand', C)):
            try:
                g, o = cap(M, d); ok, msg = R.check_implementation(d, R.collect(d, o)); res[name] = (g, ok, msg)
            except Exception as e:
                res[name] = (None, False, 'run failed: ' + str(e)[:120])
        tag = f"{'verify' if q == 8 else 'decode'} {bsz} {layer} seed{seed}"
        if not all(v[0] is not None for v in res.values()):
            print('ROW', json.dumps(dict(case=tag, error={k: v[2] for k, v in res.items()})), flush=True); continue
        ts = {k: [] for k in res}
        for rep in range(20):
            for k in (('ours', 'cand') if rep % 2 == 0 else ('cand', 'ours')): ts[k].append(t1(res[k][0]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(case=tag, ours=round(med['ours'], 3), cand=round(med['cand'], 3),
              cand_gain=round(med['ours'] / med['cand'], 5), ok={k: v[1] for k, v in res.items()})), flush=True)
print('DONE', flush=True)
