"""Decoder whole-layer attention plan sweep for one shape (env B, Q, LAYER): _TUNED7 overrides captured as separate
graphs from one module, reference-checked, paired cold-L2 replays."""
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
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
bsz, q, layer = int(os.environ['B']), int(os.environ['Q']), os.environ['LAYER']
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
key = (bsz, q, layer == 'sliding', 1)
orig = S._TUNED7.get(key)
V = {'base': None}
for ns in ((2, 4, 6, 8, 12, 16, 24, 32) if bsz > 1 else (8, 16, 24, 32, 48, 64)):
    for coop in ((0, 1) if q == 1 else (1,)):
        V[f'ns{ns}_c{coop}'] = dict(n_splits=ns, coop=coop, nc=1)
agg = {k: [] for k in V}
for seed in (5001, 5002):
    c = dict(batch=bsz, ctx=80000, layer=layer, page=1, dist='soft', q_len=q, spread=('agentic' if bsz > 1 else 'fixed'), seed=seed)
    ls = R.case_fields(c, w)['local_start']
    d = R.generate_input(**c, local_start=ls)
    graphs = {}
    for name, ov in V.items():
        if ov is None:
            if orig is None: S._TUNED7.pop(key, None)
            else: S._TUNED7[key] = orig
        else:
            S._TUNED7[key] = ov
        try:
            g, o = cap(d)
            for _ in range(2): g.replay()
            torch.cuda.synchronize()
            ok, msg = R.check_implementation(d, R.collect(d, o), w)
        except Exception as e:
            print('VAR_FAIL', name, str(e)[:100], flush=True); break
        if ok: graphs[name] = g
    names = list(graphs); ts = {k: [] for k in names}
    for rep in range(12):
        for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
    for k, v in ts.items(): agg[k].append(statistics.median(v))
base = statistics.geometric_mean(agg['base'])
res = sorted((statistics.geometric_mean(v), k) for k, v in agg.items() if len(v) == 2)
print('ROW', json.dumps(dict(B=bsz, q=q, layer=layer, base=round(base, 1), tuned=str(orig),
      best=[(round(t, 1), k, round(base / t, 4)) for t, k in res[:5]])), flush=True)
