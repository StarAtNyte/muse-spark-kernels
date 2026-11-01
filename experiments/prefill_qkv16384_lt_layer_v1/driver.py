"""Whole-layer paired screen of long-prefill QKV cuBLASLt plan 24."""
import importlib.util, json, statistics, sys
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
sys.path.insert(0, '/w/kernels/muse_prefill_layer')
import reference as R
def load(name):
    sp = importlib.util.spec_from_file_location(name, '/w/run/sub.py'); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base')
C = B
B.LT_TOKENS=(16384,)
w = R.generate_weights(R.WEIGHTS_SEED)
sb = B.prepare(w); sb['attn_choice'] = dict(B.ATTN_FIXED)
sc = dict(sb); sc['lt_choice'] = dict(sb['lt_choice'])
weight=sb['qkv_gate']; n,k=weight.shape
pid=sb['lt'].lt_plan(16384,n,k,False,sb['lt_ws'].numel(),32)
a=B._lt_pin_index(sb['lt'],pid,(24,1,0,7)); assert a is not None and a>=0
sc['lt_choice'][(16384,'qkv')]=(pid,a)
print('PLAN',sb['lt_choice'].get((16384,'qkv')),a,flush=True)
print('SHARED_PLANS', flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, st, d):
    M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, out
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for i, c in enumerate((dict(batch=1, ctx=0, q_len=16384, layer='full'), dict(batch=1, ctx=32768, q_len=16384, layer='sliding'), dict(batch=1, ctx=65536, q_len=16384, layer='full'))):
    cc = dict(page=1, dist='soft', seed=9400 + i, **c)
    if hasattr(R, 'case_fields'): cc.update(R.case_fields(cc, w))
    d = R.generate_input(**cc)
    (gb, ob), (gc, oc) = cap(B, sb, d), cap(C, sc, d)
    for name, out in [('base', ob), ('candidate', oc)]:
        ok, msg = R.check_implementation(d, R.collect(d, out), w)
        print('CHECK', name, ok, str(msg)[:200], flush=True)
        if not ok: raise RuntimeError('reference check failed: ' + name)
    tb, tc = [], []
    for rep in range(12):
        if rep % 2: tc.append(t1(gc)); tb.append(t1(gb))
        else: tb.append(t1(gb)); tc.append(t1(gc))
    print('ROW', json.dumps(dict(case=c, base_us=round(statistics.median(tb), 1), candidate_us=round(statistics.median(tc), 1),
          gain=round(statistics.median(tb) / statistics.median(tc), 4))), flush=True)
print('DONE', flush=True)
