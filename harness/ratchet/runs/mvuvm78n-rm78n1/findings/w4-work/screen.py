"""Whole-decoder paired screen: scored source (/w/kernels) vs candidate (/w/run/sub.py) on given shapes.
Each output reference-checked; alternating cold-L2 graph replays. Shapes from env CASES (json list) or defaults."""
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
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base', '/w/kernels/muse_decode_layer/submission.py')
C = load('cand', '/w/run/sub.py')
w = R.generate_weights(R.WEIGHTS_SEED)
sb = B.prepare(w); sc = C.prepare(w)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, d, st):
    o = M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
cases = [dict(batch=b, q_len=q, layer=l) for b,q in [(1,1),(16,1),(32,1),(1,8),(16,8),(32,8)] for l in ['sliding','full']]
for i, c in enumerate(cases):
    for seed in (4001, 4002):
        cc = dict(ctx=80000, page=1, dist='soft', spread=('agentic' if c['batch'] > 1 else 'fixed'), seed=seed + 10 * i, **c)
        ls = R.case_fields(cc, w)['local_start']
        d = R.generate_input(**cc, local_start=ls)
        res = {}
        for name, M, st in (('base', B, sb), ('cand', C, sc)):
            try:
                g, o = cap(M, d, st)
                ok, msg = R.check_implementation(d, R.collect(d, o), w)
                res[name] = (g, ok, msg)
            except Exception as e:
                res[name] = (None, False, 'run failed: ' + str(e)[:100])
        if not all(r[0] is not None for r in res.values()):
            print('ROW', json.dumps(dict(case=c, seed=seed, error={k: v[2] for k, v in res.items()})), flush=True); continue
        ts = {k: [] for k in res}
        for rep in range(24):
            for k in (('base', 'cand') if rep % 2 == 0 else ('cand', 'base')): ts[k].append(t1(res[k][0]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(case=c, seed=seed, base=round(med['base'], 1), cand=round(med['cand'], 1),
              gain=round(med['base'] / med['cand'], 4), ok={k: v[1] for k, v in res.items()},
              msg=res['cand'][2][:90])), flush=True)
print('DONE', flush=True)
