"""decoder_t1grid_v1 round 1: tokr-8 sk2 config sweep (qkv / unpooler slice / pooler) inside the full decoder layer at
T=1 (decode-1) and T=8 (verify-1), cold L2, CUDA graphs, rotating replay order, paired ratios base/var (>1 = faster)."""
import importlib.util, json, statistics, sys, time, os
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
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
tp = time.time(); st = S.prepare(w); torch.cuda.synchronize(); log('PREPARED', round(time.time() - tp, 1), 's')
bst = st['bfp']
log('mm_pick', {str(k): v for k, v in st.get('mm_pick', {}).items() if k[0] in (1, 8)})
log('sk2 names', sorted(bst['sk2'].keys()))
log('PCX_CFG', {str(k): v for k, v in S.PCX_CFG.items()})
# which sk2 names actually run at T=1/8
SEEN = set()
_orig = S.bfp__run_sk2
def _spy(x, layer, m, n, k, state, dg=0, dbg=None):
    SEEN.add((layer, m, S.bfp__sk2_name(layer, m))); return _orig(x, layer, m, n, k, state, dg, dbg)
S.bfp__run_sk2 = _spy
PICK0 = dict(S.bfp_SK2_PICK)
def apply(v):
    S.bfp_SK2_PICK.clear(); S.bfp_SK2_PICK.update(PICK0)
    for k, val in v.items(): S.bfp_SK2_PICK[k] = val
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
        if e.device_type.name == 'CUDA' and 'elementwise' not in e.name and 'add_' not in e.name:
            agg.setdefault(e.name[:90], []).append(e.device_time)
    return {k: round(sum(v) / n, 2) for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))}
Q = ['qkv8n2_r176', 'qkv8w8_r176', 'qkv8c2_r100', 'qkv8r16_r100', 'qkv8r16w8_r100', 'qkv8r16b_r100', 'qkv8r8_r72', 'qkv8r32c2_r100']
U = ['slice8c2_r100', 'slice8u1_r192', 'slice8w4_r192', 'slice8n1_r192', 'slice8w16_r192']
P = ['pool8a_r100', 'pool8b_r72', 'pool8r8_r72', 'pool8n2_r72', 'pool8c2_r100', 'pooler']
V = {'base': {}, 'base2': {}}
V.update({q: {('qkv_gate', 8): q} for q in Q})
V.update({u: {('unpooler', 8): u} for u in U})
V.update({p: {('pooler', 8): p} for p in P})
base = dict(ctx=80000, page=1, dist='soft', batch=1)
CASES = [dict(q_len=1, layer='sliding', seed=9401, **base), dict(q_len=1, layer='full', seed=9402, **base),
         dict(q_len=8, layer='sliding', seed=9403, **base), dict(q_len=8, layer='full', seed=9404, **base),
         dict(q_len=1, layer='sliding', seed=9405, **base), dict(q_len=8, layer='sliding', seed=9406, **base)]
REPS = int(os.environ.get('REPS', '40'))
for i, c in enumerate(CASES):
    T = c['q_len']
    ls = R.case_fields(c, w)['local_start']; d = R.generate_input(**c, local_start=ls)
    gs, chk = {}, {}
    for nm, v in V.items():
        try:
            SEEN.clear()
            g, o = cap(d, v); okk, msg = R.check_implementation(d, R.collect(d, o), w)
            chk[nm] = bool(okk)
            if nm == 'base': log('SEEN', T, sorted(SEEN))
            if okk: gs[nm] = g
            else: print('FAILCHK', nm, str(msg)[:200], flush=True)
        except Exception as e:
            chk[nm] = 'EXC ' + repr(e)[:300]; print('EXC', nm, repr(e)[:300], flush=True)
    names = list(gs); ts = {n: [] for n in names}
    for rep in range(REPS):
        sh = rep % len(names); order = names[sh:] + names[:sh]
        if rep % 2: order = order[::-1]
        for n in order: ts[n].append(t1(gs[n]))
    med = {n: round(statistics.median(x), 2) for n, x in ts.items()}
    rat = {n: round(statistics.median(a / b for a, b in zip(ts['base'], ts[n])), 4) for n in names}
    print('ROW', json.dumps(dict(T=T, layer=c['layer'], med_us=med, base_over_var=rat, ok={k: v for k, v in chk.items() if v is not True})), flush=True)
    if i < 4:
        for n in names:
            pr = prof(gs[n])
            if n == 'base':
                print('PROF', T, c['layer'], n, json.dumps(pr), flush=True)
            else:
                print('PROFSK2', T, c['layer'], n, json.dumps({k: v for k, v in pr.items() if 'sk2' in k or 'proj' in k}), flush=True)
    del gs; torch.cuda.empty_cache()
log('DONE')
