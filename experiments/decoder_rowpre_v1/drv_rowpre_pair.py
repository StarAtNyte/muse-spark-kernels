"""rowpre paired screen (from kvpre driver).\nkvpre paired screen (from gle8 driver): ONE candidate file (/w/run/sub.py), one prepare; variants toggled by module globals at capture.
ctrl = GLE8_ENABLED False (the parent's T = 8 b1 path), c1 / c2 = gle path at T = 8 with GLE8_NCH 1 / 2.
Correctness on the 13 TASK.md tests (every variant at T = 8 tests), then rotated cold-L2 graph replays."""
import importlib.util, json, os, statistics, sys, time
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
t0 = time.time()
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
w = R.generate_weights(R.WEIGHTS_SEED)
st = C.prepare(w)
print('PREPARED', round(time.time() - t0, 1), flush=True)
BASE = dict(ROWPRE_VNW=(4, 32), ROWPRE_SPRE=False, ROWPRE_BIG_PRE=False, ROWPRE_DUP=1, ROWPRE_CLUS=False, ROWPRE_CLUSBIG=False)
VARS = {'ctrl': {}, 'clus': dict(ROWPRE_CLUS=True), 'clusall': dict(ROWPRE_CLUS=True, ROWPRE_CLUSBIG=True),
        'cdup': dict(ROWPRE_CLUS=True, ROWPRE_CLUSBIG=True, ROWPRE_DUP=2)}
VARS = {k: {**BASE, **v} for k, v in VARS.items()}
def setv(v):
    for k, val in VARS[v].items(): setattr(C, k, val)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d, v):
    setv(v)
    o = C.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = C.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
def mk(cc):
    ls = R.case_fields(cc, w)['local_start']
    return R.generate_input(**cc, local_start=ls)
TESTS = [dict(batch=1, ctx=10, layer='sliding'), dict(batch=1, ctx=3000, layer='full'), dict(batch=8, ctx=2047, layer='sliding'),
         dict(batch=9, ctx=5000, layer='sliding', spread='mixed', page=1), dict(batch=3, ctx=6000, layer='full', q_len=8, dist='outliers'),
         dict(batch=5, ctx=5000, layer='sliding', q_len=8, spread='mixed', page=1), dict(batch=33, ctx=20000, layer='full', spread='mixed', page=1),
         dict(batch=2, ctx=300000, layer='sliding', q_len=8), dict(batch=1, ctx=1048576, layer='full', q_len=8),
         dict(batch=4, ctx=7000, layer='sliding', q_len=8, page=1, kv_scale=0.015625), dict(batch=1, ctx=80000, layer='full', dist='soft', page=1),
         dict(batch=4, ctx=6000, layer='sliding', q_len=8, dist='soft'), dict(batch=16, ctx=20000, layer='full', spread='mixed', dist='soft', page=1)]
for i, tc in enumerate(TESTS):
    for v in [k for k in VARS if k not in ('ctrl', 'cdup')]:
        cc = dict(seed=7000 + i, **tc)
        try:
            d = mk(cc)
            setv(v); o = C.kernel_fn(d, st); torch.cuda.synchronize()
            ok, msg = R.check_implementation(d, R.collect(d, o), w)
        except Exception as e:
            ok, msg = False, 'run failed: ' + repr(e)[:300]
        print('TEST', json.dumps(dict(i=i, v=v, ok=bool(ok), msg=str(msg)[:160])), flush=True)
        del d
        torch.cuda.empty_cache()
NAMES = tuple(VARS)
cases = []
for (B, L) in ((1, 1), (16, 1), (32, 1), (1, 8), (16, 8), (32, 8)):
    for layer in ('sliding', 'full'):
        cases.append(dict(batch=B, q_len=L, layer=layer))
REPS = int(os.environ.get('REPS', '30'))
for i, c in enumerate(cases):
    for seed in (4001, 4002):
        cc = dict(ctx=80000, page=1, dist='soft', seed=seed + 10 * i, **c)
        if c['batch'] > 1: cc['spread'] = 'agentic'
        d = mk(cc)
        res = {}
        for v in NAMES:
            try:
                g, o = cap(d, v)
                ok, msg = R.check_implementation(d, R.collect(d, o), w)
                res[v] = (g, ok, msg)
            except Exception as e:
                res[v] = (None, False, 'run failed: ' + repr(e)[:200])
        if not all(r[0] is not None for r in res.values()):
            print('ROW', json.dumps(dict(case=c, seed=seed, error={k: v[2] for k, v in res.items()})), flush=True); continue
        ts = {k: [] for k in res}
        for rep in range(REPS):
            order = NAMES if rep % 2 == 0 else NAMES[::-1]
            for k in order: ts[k].append(t1(res[k][0]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(case=c, seed=seed, **{k: round(m, 2) for k, m in med.items()},
              gain={k: round(med['ctrl'] / m, 4) for k, m in med.items() if k != 'ctrl'},
              ok={k: bool(v[1]) for k, v in res.items()}, msg={k: str(v[2])[:60] for k, v in res.items() if not v[1]})), flush=True)
        del res, d
        torch.cuda.empty_cache()
print('DONE', round(time.time() - t0, 1), flush=True)
