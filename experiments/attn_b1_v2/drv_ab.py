"""Paired screen: parent (/w/modal/_b1lat/base.py, identical to parent.py) vs candidate (/w/run/sub.py) on the four
batch-1 shapes, plus optional candidate plan overrides (env OVR = {case: {name: override}}). Cold-L2 graph replays,
order rotated/reversed each rep, every graph reference-checked, 3 seeds."""
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
B = load('base', '/w/modal/_b1lat/base.py'); C = load('cand', '/w/run/sub.py')
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
  't1': dict(batch=1, ctx=10, heads='tp8', layer='sliding'),
  't9': dict(batch=1, ctx=1048576, heads='tp8', layer='full', q_len=8),
  't1e': dict(batch=1, ctx=6000, heads='tp8', layer='sliding', dist='edges', page=1),
  't1f': dict(batch=1, ctx=3000, heads='tp8', layer='full', kv_scale=0.015625),
  'tv8': dict(batch=1, ctx=3000, heads='tp8', layer='sliding', q_len=8, dist='edges'),
  'tv8k': dict(batch=1, ctx=4000, heads='tp8', layer='sliding', q_len=8, page=1, kv_scale=0.015625),
  'v32f': dict(batch=32, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=8, spread='agentic'),
  'd16s': dict(batch=16, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=1, spread='agentic'),
  'v16f': dict(batch=16, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=8, spread='agentic'),
}
OVR = json.loads(os.environ.get('OVR', '{}'))
REPS = int(os.environ.get('REPS', '24'))
for cn in os.environ.get('CASES', 'd1s,v1s,d1f,v1f').split(','):
    for seed in (11, 4242, 777)[:int(os.environ.get('SEEDS', '3'))]:
        d = R.generate_input(**SPEC[cn], seed=seed)
        gs, bad = {}, {}
        fns = [('base', lambda: B.kernel_fn(d)), ('cand', lambda: C.kernel_fn(d))]
        fns += [(k, (lambda ov=v: C._launch(d, **ov))) for k, v in OVR.get(cn, {}).items()]
        for vn, fn in fns:
            try:
                g, o = cap(fn); ok, msg = R.check_implementation(d, R.collect(d, o))
                if ok: gs[vn] = g
                else: bad[vn] = str(msg)[:100]
            except Exception as e:
                bad[vn] = 'ERR ' + str(e)[:150]
        names = list(gs); ts = {k: [] for k in names}
        for rep in range(REPS):
            r = rep % len(names); order = names[r:] + names[:r]
            if rep % 2: order = order[::-1]
            for k in order: ts[k].append(t1(gs[k]))
        med = {k: round(statistics.median(v), 3) for k, v in ts.items()}
        gain = {k: round(med['base'] / v, 4) for k, v in med.items() if 'base' in med}
        print('ROW', json.dumps(dict(case=cn, seed=seed, med=med, gain=gain, bad=bad)), flush=True)
print('DONE', flush=True)
