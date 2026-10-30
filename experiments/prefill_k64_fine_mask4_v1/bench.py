"""Compare parent and K64 candidate: whole-layer graph timing, paired cold L2; attention choice pinned."""
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
    path = '/w/kernels/muse_prefill_layer/submission.py' if name == 'base' else '/w/run/sub.py'
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base'); C = load('cand')
w = R.generate_weights(R.WEIGHTS_SEED)
sb = B.prepare(w); sb['attn_choice'] = dict(B.ATTN_FIXED)
sc = C.prepare(w); sc['attn_choice'] = dict(C.ATTN_FIXED)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, st, d):
    M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for i, c in enumerate((dict(batch=1, ctx=80000, q_len=753, layer='sliding'), dict(batch=4, ctx=80000, q_len=753, layer='full'),
                       dict(batch=1, ctx=0, q_len=16384, layer='full'), dict(batch=1, ctx=65536, q_len=16384, layer='full'))):
    cc = dict(page=1, dist='soft', seed=9400 + i, **c)
    if hasattr(R, 'case_fields'): cc.update(R.case_fields(cc, w))
    d = R.generate_input(**cc)
    gb, gc = cap(B, sb, d), cap(C, sc, d)
    tb, tc = [], []
    for rep in range(8):
        if rep % 2: tc.append(t1(gc)); tb.append(t1(gb))
        else: tb.append(t1(gb)); tc.append(t1(gc))
    print('ROW', json.dumps(dict(case=c, base_us=round(statistics.median(tb), 1), candidate_us=round(statistics.median(tc), 1),
          gain=round(statistics.median(tb) / statistics.median(tc), 4))), flush=True)
print('DONE', flush=True)
