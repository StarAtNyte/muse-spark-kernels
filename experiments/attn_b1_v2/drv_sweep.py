"""Round 1: sweep existing plan overrides of the readable parent (/w/run/sub.py) on the four batch-1 shapes.
Cold-L2 CUDA-graph replays, round-robin order rotated every rep, reference-checked, 2 seeds."""
import importlib.util, json, os, statistics, sys
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
M = load('base', '/w/run/sub.py')
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(fn):
    o = fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = fn()
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
SPEC = {
  'd1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=1),
  'd1f': dict(batch=1, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=1),
  'v1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=8),
  'v1f': dict(batch=1, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=8),
}
VAR = {
  'd1s': {'n16': dict(n_splits=16), 'n24': dict(n_splits=24), 'cs16': dict(cs=16, ncl=1), 'cs8': dict(cs=8, ncl=1),
          'cs8x2': dict(cs=8, ncl=2), 'coop': dict(coop=1), 'pdl': dict(pdl=True)},
  'v1s': {'n16': dict(n_splits=16, coop=1), 'n24': dict(n_splits=24, coop=1), 'cs16': dict(cs=16, ncl=1), 'cs8': dict(cs=8, ncl=1),
          'nocoop': dict(coop=0), 'n66': dict(n_splits=66, coop=1)},
  'd1f': {'n132': dict(nc=8, coop=1, n_splits=132), 'n200': dict(nc=8, coop=1, n_splits=200), 'nc4': dict(nc=4, coop=1, n_splits=252),
          'nc16': dict(nc=16, coop=1, n_splits=252), 'nocoop': dict(nc=8, coop=0, n_splits=252), 'n264': dict(nc=8, coop=1, n_splits=264)},
  'v1f': {'n132': dict(n_splits=132), 'n200': dict(n_splits=200), 'n264': dict(n_splits=264), 'nc4': dict(n_splits=252, nc=4),
          'nc16': dict(n_splits=252, nc=16), 'coop0': dict(n_splits=252, coop=0)},
}
REPS = int(os.environ.get('REPS', '24'))
for cn in os.environ.get('CASES', 'd1s,v1s,d1f,v1f').split(','):
    for seed in (11, 4242):
        d = R.generate_input(**SPEC[cn], seed=seed)
        gs, bad = {}, {}
        for vn, ov in [('base', {})] + list(VAR[cn].items()):
            ov = dict(ov); pdl = ov.pop('pdl', None)
            try:
                fn = (lambda: M.kernel_fn(d)) if vn == 'base' else (lambda ov=ov, pdl=pdl: M._launch(d, pdl=pdl, **ov))
                g, o = cap(fn); ok, msg = R.check_implementation(d, R.collect(d, o))
                if ok: gs[vn] = g
                else: bad[vn] = str(msg)[:80]
            except Exception as e:
                bad[vn] = 'ERR ' + str(e)[:100]
        names = list(gs); ts = {k: [] for k in names}
        for rep in range(REPS):
            r = rep % len(names); order = names[r:] + names[:r]
            if rep % 2: order = order[::-1]
            for k in order: ts[k].append(t1(gs[k]))
        med = {k: round(statistics.median(v), 3) for k, v in ts.items()}
        print('ROW', json.dumps(dict(case=cn, seed=seed, med=med, gain={k: round(med['base'] / v, 4) for k, v in med.items()}, bad=bad)), flush=True)
print('DONE', flush=True)
