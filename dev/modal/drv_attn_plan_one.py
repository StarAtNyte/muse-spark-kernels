"""Single-request full-context attention: sweep launch-plan overrides (splits, coop finalize, workspace copies),
reference-check each, paired cold-L2 graph timing vs the scored plan."""
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
def cap(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = fn()
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
import os
bsz, q, layer = int(os.environ['B']), int(os.environ['Q']), os.environ['LAYER']
dist = 'soft' if layer == 'sliding' else 'coherent'
d = R.generate_input(bsz, 80000, 'tp8', layer, 77 + bsz + q, q_len=q, dist=dist, page=1, spread='agentic')
variants = {'base': {}}
if layer == 'sliding':
    for ns in (2, 3, 4, 6, 8, 10, 12, 16):
        variants[f'ns{ns}_c1'] = dict(n_splits=ns, coop=1, nc=1)
else:
    for ns in (132, 198, 264, 330, 396, 528):
        variants[f'ns{ns}'] = dict(n_splits=ns)
graphs = {}
for name, ov in variants.items():
    try:
        g, o = cap(lambda: S._launch(d, **ov) if ov else S.kernel_fn(d))
        for _ in range(3): g.replay()
        torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, R.collect(d, o))
    except Exception as e:
        print('VAR_FAIL', name, str(e)[:100], flush=True); break
    if ok: graphs[name] = g
names = list(graphs); ts = {k: [] for k in names}
for rep in range(16):
    for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
med = sorted((statistics.median(v), k) for k, v in ts.items())
base = statistics.median(ts['base'])
print('ROW', json.dumps(dict(B=bsz, q=q, layer=layer, base=round(base, 2), n=len(names),
      best=[(round(t, 2), k, round(base / t, 3)) for t, k in med[:4]])), flush=True)
