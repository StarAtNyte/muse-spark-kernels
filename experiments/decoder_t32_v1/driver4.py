"""decoder_t32_v1 round 1: T=16/32 projection section. Parent picks vs alternative sk2 geometries / unfused pooler,
one prepare, graphs per variant, reference-checked, paired round-robin cold-L2 replays, per-kernel CUPTI profile."""
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
tp = time.time(); st = S.prepare(w); torch.cuda.synchronize(); log('PREPARED', round(time.time() - tp, 1), 's')
bst = st['bfp']
log('mm_pick', {str(k): v for k, v in st.get('mm_pick', {}).items()})
log('choice', {str(k): v for k, v in bst.get('choice', {}).items() if k[1] in (8, 16, 32)})
log('sk2 names', sorted(bst['sk2'].keys()))
log('PCX_CFG', {str(k): v for k, v in S.PCX_CFG.items()})
QB = {}   # T -> qkvbig config index
_bg = S._bfp_gemm
def _bfp_gemm(st_, name, x, m, n, k):
    if name == 'qkv' and m in QB:
        return S.tt_v14_qkv_run_qbig_candidate(x, st_['qkv_compact'], QB[m])
    return _bg(st_, name, x, m, n, k)
S._bfp_gemm = _bfp_gemm
log('qkvbig configs', st['qkv_compact'].get('configs'))
def apply(v):
    QB.clear(); QB.update(v)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d, v):
    apply(v)
    try:
        o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): o = S.kernel_fn(d, st)
        g.replay(); torch.cuda.synchronize(); return g, o
    finally:
        apply({})
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
            agg.setdefault(e.name[:46], []).append(e.device_time)
    return {k: round(sum(v) / n, 2) for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:12]}
V16 = {'base': {}, 'qb0': {16: 0}, 'qb1': {16: 1}, 'qb2': {16: 2}}
V32 = {'base': {}, 'qb0': {32: 0}, 'qb1': {32: 1}, 'qb2': {32: 2}}
base = dict(ctx=80000, page=1, dist='soft', spread='agentic')
CASES = [dict(batch=16, layer='sliding', seed=9201, **base), dict(batch=32, layer='sliding', seed=9202, **base),
         dict(batch=16, layer='full', seed=9203, **base), dict(batch=32, layer='full', seed=9204, **base)]
REPS = 30
for i, c in enumerate(CASES):
    T = c['batch']; vs = V16 if T == 16 else V32
    ls = R.case_fields(c, w)['local_start']; d = R.generate_input(**c, local_start=ls)
    gs, chk = {}, {}
    for nm, v in vs.items():
        try:
            g, o = cap(d, v); okk, msg = R.check_implementation(d, R.collect(d, o), w)
            chk[nm] = (bool(okk), str(msg)[:80])
            if okk: gs[nm] = g
        except Exception as e:
            chk[nm] = (False, 'EXC ' + repr(e)[:160])
    names = list(gs); ts = {n: [] for n in names}
    for rep in range(REPS):
        for n in (names if rep % 2 == 0 else names[::-1]): ts[n].append(t1(gs[n]))
    med = {n: round(statistics.median(x), 2) for n, x in ts.items()}
    rat = {n: round(statistics.median(a / b for a, b in zip(ts['base'], ts[n])), 4) for n in names if 'base' in ts}
    print('ROW', json.dumps(dict(T=T, layer=c['layer'], med_us=med, base_over_var=rat, ok=chk)), flush=True)
    if i < 2:
        print('NOTE', 'qkv kernel is the first in the graph')
        for n in names: print('PROF', T, c['layer'], n, json.dumps(prof(gs[n])), flush=True)
    del gs; torch.cuda.empty_cache()
log('DONE')
