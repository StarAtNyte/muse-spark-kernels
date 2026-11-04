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
PICK0 = dict(S.bfp_SK2_PICK); MMP0 = dict(st.get('mm_pick', {}))
def apply(v):
    S.bfp_SK2_PICK.clear(); S.bfp_SK2_PICK.update(PICK0); st['mm_pick'].clear(); st['mm_pick'].update(MMP0)
    for k, val in v.items():
        if k[0] == 'mm': st['mm_pick'][(k[1], k[2])] = val
        else: S.bfp_SK2_PICK[k] = val
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
V16 = {'base': {}, 'o16t': {('o', 16): 'o16t_r68'}, 'qkv16t': {('qkv_gate', 16): 'qkv16t_r200'},
       'un16t': {('unpooler', 16): 'unpooler16t_r72'},
       'all3': {('o', 16): 'o16t_r68', ('qkv_gate', 16): 'qkv16t_r200', ('unpooler', 16): 'unpooler16t_r72'},
       'unf_p16t': {('mm', 16, 'pr'): 'bfp', ('pooler', 16): 'pooler16t_r192'},
       'unf_p32': {('mm', 16, 'pr'): 'bfp', ('pooler', 16): 'pooler32_r144'}}
V32 = {'base': {}, 'qkvn2': {('qkv_gate', 32): 'qkv32n2_r208'}, 'un32b': {('unpooler', 32): 'unpooler32b_r200'},
       'un32c': {('unpooler', 32): 'unpooler32c_r144'}, 'un16t': {('unpooler', 32): 'unpooler16t_r72'},
       'o32t': {('o', 32): 'o32t_r100'}, 'o16t': {('o', 32): 'o16t_r68'},
       'unf_p32': {('mm', 32, 'pr'): 'bfp'}, 'unf_p32b': {('mm', 32, 'pr'): 'bfp', ('pooler', 32): 'pooler32b_r200'}}
base = dict(ctx=80000, page=1, dist='soft', spread='agentic')
CASES = [dict(batch=16, layer='sliding', seed=9201, **base), dict(batch=32, layer='sliding', seed=9202, **base),
         dict(batch=16, layer='full', seed=9203, **base), dict(batch=32, layer='full', seed=9204, **base)]
REPS = 24
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
        for n in names: print('PROF', T, c['layer'], n, json.dumps(prof(gs[n])), flush=True)
    del gs; torch.cuda.empty_cache()
log('DONE')
