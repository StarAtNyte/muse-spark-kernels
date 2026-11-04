"""decoder_bf16_v1 round 1: paired in-process A/B (parent par.py vs candidate sub.py) on the 12 scored shapes, cold L2,
CUDA graphs, alternating replay order; then all 13 TASK.md tests x 2 seeds on both with the checker's error messages."""
import importlib.util, json, statistics, sys, time, os, math
from unittest.mock import MagicMock
import torch
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_decode_layer'); sys.path.insert(0, '/w/run')
import reference as R
REPS = int(os.environ.get('REPS', '30')); SEEDS = int(os.environ.get('SEEDS', '2'))
w = R.generate_weights(R.WEIGHTS_SEED)
def load(tag, path):
    sp = importlib.util.spec_from_file_location(tag, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    st = M.prepare(w); torch.cuda.synchronize(); log('PREPARED', tag); return M, st
MODS = {}
MODS['A'] = load('par', '/w/run/par.py')
MODS['B'] = load('cand', '/w/run/sub.py')
print('RQ picks B', sorted(MODS['B'][1].get('chen_rq', {})), flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(v, d):
    M, st = MODS[v]
    o = M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
SCORED = [dict(batch=b, ctx=80000, layer=l, page=1, dist='soft', q_len=q, **({} if b == 1 else dict(spread='agentic')))
          for q in (1, 8) for b in (1, 16, 32) for l in ('sliding', 'full')]
allr = {}
for rnd in range(SEEDS):
    for i, c in enumerate(SCORED):
        c2 = dict(c, seed=5100 + 37 * rnd + i); ls = R.case_fields(c2, w)['local_start']; d = R.generate_input(**c2, local_start=ls)
        gs, chk = {}, {}
        for v in ('A', 'B'):
            g, o = cap(v, d); gs[v] = g
            ok, msg = R.check_implementation(d, R.collect(d, o), w); chk[v] = (bool(ok), str(msg)[:160])
        ts = {'A': [], 'B': []}
        for rep in range(REPS):
            for v in (('A', 'B') if rep % 2 == 0 else ('B', 'A')): ts[v].append(t1(gs[v]))
        pr = statistics.median(a / b for a, b in zip(ts['A'], ts['B']))
        key = f"T{c['batch'] * c['q_len']}{c['layer'][0]}"
        allr.setdefault(key, []).append(pr)
        print('AB', key, json.dumps(dict(A=round(statistics.median(ts['A']), 2), B=round(statistics.median(ts['B']), 2), ratio=round(pr, 4),
                                          okA=chk['A'], okB=chk['B'])), flush=True)
        del gs, d; torch.cuda.empty_cache()
gm = []
for k, v in allr.items():
    g = statistics.geometric_mean(v); gm.append(g); print('GM', k, round(g, 4), [round(x, 4) for x in v])
print('GM all', round(statistics.geometric_mean(gm), 4), flush=True)
TESTS = [dict(batch=1, ctx=10, layer='sliding'), dict(batch=1, ctx=3000, layer='full'), dict(batch=8, ctx=2047, layer='sliding'),
         dict(batch=9, ctx=5000, layer='sliding', spread='mixed', page=1), dict(batch=3, ctx=6000, layer='full', q_len=8, dist='outliers'),
         dict(batch=5, ctx=5000, layer='sliding', q_len=8, spread='mixed', page=1), dict(batch=33, ctx=20000, layer='full', spread='mixed', page=1),
         dict(batch=2, ctx=300000, layer='sliding', q_len=8), dict(batch=1, ctx=1048576, layer='full', q_len=8),
         dict(batch=4, ctx=7000, layer='sliding', q_len=8, page=1, kv_scale=0.015625), dict(batch=1, ctx=80000, layer='full', dist='soft', page=1),
         dict(batch=4, ctx=6000, layer='sliding', q_len=8, dist='soft'), dict(batch=16, ctx=20000, layer='full', spread='mixed', dist='soft', page=1)]
for v in ('A', 'B'):
    M, st = MODS[v]; res = []
    for i, c in enumerate(TESTS):
        for seed in (11 + i, 777 + i):
            try:
                c2 = dict(c, seed=seed); ls = R.case_fields(c2, w)['local_start']; d = R.generate_input(**c2, local_start=ls)
                o = M.kernel_fn(d, st); torch.cuda.synchronize()
                ok, msg = R.check_implementation(d, R.collect(d, o), w); res.append(bool(ok))
                print('TEST', v, i, seed, bool(ok), str(msg)[:220], flush=True)
                del d, o
            except Exception as ex:
                res.append(False); print('EXC', v, c, repr(ex)[:300], flush=True)
        torch.cuda.empty_cache()
    print('TESTS', v, sum(res), '/', len(res), flush=True)
log('DONE')
