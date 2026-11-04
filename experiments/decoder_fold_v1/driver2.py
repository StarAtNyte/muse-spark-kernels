"""decoder_fold_v1 round 2: (A) with spin waits qkv_gate sk2 x multicast across 2/4-CTA clusters at T=16/32. One prepare; graphs per
variant; reference-checked; paired round-robin cold-L2 replays vs the parent kernel; per-kernel profile; TASK tests."""
import importlib.util, json, statistics, sys, time
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
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
tb = time.time(); S._mc_ext(4); log('mc4 built', round(time.time() - tb, 1))
tp = time.time(); st = S.prepare(w); torch.cuda.synchronize(); log('PREPARED', round(time.time() - tp, 1), 's')
DEF = dict(S.SK2_QKV_MC)
VARS = {'base': {}, 'mc2': {16: 2, 32: 2}, 'mc4': {16: 4, 32: 4}}
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d, v):
    S.SK2_QKV_MC.clear(); S.SK2_QKV_MC.update(v)
    try:
        o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): o = S.kernel_fn(d, st)
        g.replay(); torch.cuda.synchronize(); return g, o
    finally:
        S.SK2_QKV_MC.clear(); S.SK2_QKV_MC.update(DEF)
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
from torch.profiler import profile, ProfilerActivity
def prof(g, n=8):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            flush.add_(1); g.replay()
        torch.cuda.synchronize()
    agg = {}
    for e in p.events():
        if e.device_type.name == 'CUDA' and 'add' not in e.name and 'elementwise' not in e.name:
            agg.setdefault(e.name[:40], []).append(e.device_time)
    return {k: round(sum(v) / n, 2) for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:6]}
base = dict(ctx=80000, page=1, dist='soft')
CASES = [dict(batch=32, layer='sliding', spread='agentic', **base), dict(batch=16, layer='sliding', spread='agentic', **base),
         dict(batch=32, layer='full', spread='agentic', **base), dict(batch=16, layer='full', spread='agentic', **base),
]
allr = {}
for i, c in enumerate(CASES):
    for seed in ((9301 + i, 9401 + i) if c['batch'] > 1 and c.get('q_len', 1) == 1 else (9301 + i,)):
        c2 = dict(c, seed=seed); ls = R.case_fields(c2, w)['local_start']; d = R.generate_input(**c2, local_start=ls)
        gs, chk = {}, {}
        for nm, v in VARS.items():
            try:
                g, o = cap(d, v); okk, msg = R.check_implementation(d, R.collect(d, o), w)
                chk[nm] = (bool(okk), str(msg)[:60])
                if okk: gs[nm] = g
            except Exception as e:
                chk[nm] = (False, 'EXC ' + repr(e)[:300])
        names = list(gs); ts = {n: [] for n in names}
        for rep in range(40):
            for n in (names if rep % 2 == 0 else names[::-1]): ts[n].append(t1(gs[n]))
        rat = {n: round(statistics.median(a / b for a, b in zip(ts['base'], ts[n])), 4) for n in names if 'base' in ts}
        for n in rat: allr.setdefault((c['batch'] * c.get('q_len', 1), c['layer'], n), []).append(rat[n])
        print('ROW', json.dumps(dict(T=c['batch'] * c.get('q_len', 1), layer=c['layer'], seed=seed,
                                     med_us={n: round(statistics.median(x), 2) for n, x in ts.items()}, base_over_var=rat, ok=chk)), flush=True)
        if i < 2 and seed == 9301 + i:
            for n in names: print('PROF', c['batch'], c['layer'], n, json.dumps(prof(gs[n])), flush=True)
        del gs, d; torch.cuda.empty_cache()
for k, v in allr.items(): print('GM', k, round(statistics.geometric_mean(v), 4), v)
log('timing done')
TESTS = [dict(batch=1, ctx=10, layer='sliding'), dict(batch=1, ctx=3000, layer='full'), dict(batch=8, ctx=2047, layer='sliding'),
         dict(batch=9, ctx=5000, layer='sliding', spread='mixed', page=1), dict(batch=3, ctx=6000, layer='full', q_len=8, dist='outliers'),
         dict(batch=5, ctx=5000, layer='sliding', q_len=8, spread='mixed', page=1), dict(batch=33, ctx=20000, layer='full', spread='mixed', page=1),
         dict(batch=2, ctx=300000, layer='sliding', q_len=8), dict(batch=1, ctx=1048576, layer='full', q_len=8),
         dict(batch=4, ctx=7000, layer='sliding', q_len=8, page=1, kv_scale=0.015625), dict(batch=1, ctx=80000, layer='full', dist='soft', page=1),
         dict(batch=4, ctx=6000, layer='sliding', q_len=8, dist='soft'), dict(batch=16, ctx=20000, layer='full', spread='mixed', dist='soft', page=1)]
for vn in ('mc2', 'mc4'):
    S.SK2_QKV_MC.clear(); S.SK2_QKV_MC.update(VARS[vn]); res = []
    for i, c in enumerate(TESTS):
        if c['batch'] * c.get('q_len', 1) not in (16, 32):
            continue
        try:
            c2 = dict(c, seed=11 + i); ls = R.case_fields(c2, w)['local_start']; d = R.generate_input(**c2, local_start=ls)
            o = S.kernel_fn(d, st); torch.cuda.synchronize()
            ok, msg = R.check_implementation(d, R.collect(d, o), w); res.append(bool(ok))
            if not ok: print('FAIL', vn, c2, str(msg)[:200], flush=True)
            del d, o
        except Exception as ex:
            res.append(False); print('EXC', vn, c, repr(ex)[:300], flush=True)
        torch.cuda.empty_cache()
    print('TESTS', vn, sum(res), '/', len(res), flush=True)
log('DONE')
