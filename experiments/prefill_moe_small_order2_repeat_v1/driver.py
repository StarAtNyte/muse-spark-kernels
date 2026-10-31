"""Prefill: cost of the MoE scale HMUL2s in a tensor-pipe-saturated kernel. Ablation (outputs invalid by design):
convert8 without the four bf16 multiplies. Whole-layer graph timing, paired cold L2; attention choice pinned."""
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
B = load('base'); C = load('cand')
C.MOE_ORDER_SMALL = 2
print('MOE SMALL ORDER 2 SCREEN', flush=True)
w = R.generate_weights(R.WEIGHTS_SEED)
sb = B.prepare(w); sb['attn_choice'] = dict(B.ATTN_FIXED)
sc = dict(sb); sc['mod'] = C._build(C._fast_scales_ok(w)); sc['mod'].moe_init(sc['smem']); sc['mod'].attn_init(sc['attn_smem']); C.SIDE_STREAM = B.SIDE_STREAM
print('SHARED_PLANS', flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, st, d):
    M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, out
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for seed in (9400, 9407, 9417):
    c = dict(batch=1, ctx=80000, q_len=753, layer='sliding')
    cc = dict(page=1, dist='soft', seed=seed, **c)
    if hasattr(R, 'case_fields'): cc.update(R.case_fields(cc, w))
    d = R.generate_input(**cc)
    (gb, ob), (gc, oc) = cap(B, sb, d), cap(C, sc, d)
    for name, out in [('base', ob), ('candidate', oc)]:
        ok, msg = R.check_implementation(d, R.collect(d, out), w)
        print('CHECK', seed, name, ok, str(msg)[:200], flush=True)
        if not ok: raise RuntimeError('reference check failed: ' + name)
    tb, tc, ratios = [], [], []
    for rep in range(24):
        if rep % 2: y,x=t1(gc),t1(gb)
        else: x,y=t1(gb),t1(gc)
        tb.append(x);tc.append(y);ratios.append(x/y)
    print('ROW',json.dumps(dict(seed=seed,base_us=statistics.median(tb),candidate_us=statistics.median(tc),paired_ratio=statistics.median(ratios))),flush=True)
print('DONE',flush=True)
