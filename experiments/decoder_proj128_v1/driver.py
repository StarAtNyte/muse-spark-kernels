"""decoder_proj128_v1 paired A/B: parent routes (P128_ON = {}) vs proj-board FP32 routes, one prepare, graphs per variant,
reference-checked, ABBA-style round-robin cold-L2 graph replays."""
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
tp = time.time(); st = S.prepare(w); torch.cuda.synchronize()
log('PREPARED in', round(time.time() - tp, 1), 's; p128 plans', sorted(st['p128_lt'].keys()),
    'qkv_compact routes', st.get('qkv_compact', {}).get('routes'))
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
VAR = {128: {'ctrl': set(), 'cand': {('qkv', 128)}, 'cand+un': {('qkv', 128), ('un', 128)},
             'cand+pr': {('qkv', 128), ('pr', 128)}},
       256: {'ctrl': set(), 'cand': {('o', 256), ('un', 256)}, 'cand+pr': {('o', 256), ('un', 256), ('pr', 256)},
             'o_only': {('o', 256)}, 'un_only': {('un', 256)}},
       0: {'ctrl': set(), 'cand': set(S.P128_ON)}}
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d, on):
    S.P128_ON = set(on)
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
base = dict(ctx=80000, page=1, dist='soft')
CASES = [dict(batch=16, layer='sliding', q_len=8, spread='agentic', seed=9101, **base),
         dict(batch=32, layer='sliding', q_len=8, spread='agentic', seed=9102, **base),
         dict(batch=16, layer='full', q_len=8, spread='agentic', seed=9103, **base),
         dict(batch=32, layer='full', q_len=8, spread='agentic', seed=9104, **base),
         dict(batch=16, layer='sliding', spread='agentic', seed=9105, **base),
         dict(batch=32, layer='full', spread='agentic', seed=9106, **base),
         dict(batch=1, layer='sliding', seed=9107, **base)]
ratios = {}
for i, c in enumerate(CASES):
    if time.time() - T0 > 1000: log('time guard: stop before case', i); break
    T = c['batch'] * c.get('q_len', 1)
    vs = VAR.get(T, VAR[0])
    ls = R.case_fields(c, w)['local_start']; d = R.generate_input(**c, local_start=ls)
    gs, chk, outs = {}, {}, {}
    for nm, on in vs.items():
        try:
            g, o = cap(d, on); okk, msg = R.check_implementation(d, R.collect(d, o), w)
            gs[nm] = g; outs[nm] = o; chk[nm] = (bool(okk), str(msg)[:90])
        except Exception as e:
            chk[nm] = (False, 'EXC ' + repr(e)[:150])
    names = [n for n in gs if chk[n][0]]
    ts = {n: [] for n in names}
    for rep in range(REPS):
        order = names if rep % 2 == 0 else names[::-1]
        for n in order: ts[n].append(t1(gs[n]))
    med = {n: round(statistics.median(v), 2) for n, v in ts.items()}
    rat = {}
    if 'ctrl' in ts:
        for n in names:
            if n == 'ctrl': continue
            pr = [a / b for a, b in zip(ts['ctrl'], ts[n])]
            rat[n] = round(statistics.median(pr), 4)
            ratios.setdefault(n, []).append(rat[n])
    same = {n: all(torch.equal(a, b) for a, b in zip(outs['ctrl'], outs[n])) for n in outs if n != 'ctrl' and 'ctrl' in outs}
    print('ROW', json.dumps(dict(i=i, T=T, case={k: c[k] for k in ('batch', 'layer') if k in c} | {'q_len': c.get('q_len', 1)},
          med_us=med, ctrl_over_var_paired=rat, ok=chk, bitwise_same_as_ctrl=same)), flush=True)
    del gs, outs, d; torch.cuda.empty_cache()
S.P128_ON = {('qkv', 128), ('o', 256), ('un', 256)}
log('DONE')
