"""Prefill full-attention exp-path screen: variant 43 (scored, EXPMODE 0) vs the same kernel with EXPMODE -1 (f16x2
MUFU pairs), 8, 16, 32 (FMA-pipe polynomial share). Whole layer, reference-checked, paired cold-L2 graph replays."""
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
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
NEW = {44: -1, 45: 8, 46: 16, 47: 32}
case43 = "    case 43: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, grid, dim3(512), (size_t)smem, st, a); break;"
attr43 = "  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);"
assert S.GLUE.count(case43) == 1 and S.GLUE.count(attr43) == 1
g = S.GLUE
g = g.replace(case43, case43 + ''.join(f"\n    case {v}: okc_launch(okc_attn::attn_fwd_kernel<3, {e}, false, true, false, true>, grid, dim3(512), (size_t)smem, st, a); break;" for v, e in NEW.items()))
g = g.replace(attr43, attr43 + ''.join(f"\n  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, {e}, false, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);" for e in NEW.values()))
S.GLUE = g
for v in NEW: S.ATTN_VARIANTS[v] = (3, 0)
mod = S._build(True)
st = {'mod': mod, 'attn_smem': S.ATTN_SMEM, 'no_pf': torch.empty((0,), dtype=torch.uint8, device='cuda')}
mod.attn_init(st['attn_smem'])
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(fn):
    o = fn(); torch.cuda.synchronize(); gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): o = fn()
    gr.replay(); torch.cuda.synchronize(); return gr, o
def t1(gr):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); gr.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for (B, L, ctx) in ((1, 753, 80000), (4, 753, 80000), (1, 16384, 0), (1, 16384, 32768), (1, 16384, 65536)):
    data, cfg, q8, qkv = S._synthetic_attn_case(B, L, ctx, 0, 1, 9100 + B + ctx // 1000)
    ns = S._attn_splits(cfg, data, 43, torch.cuda.get_device_properties(0).multi_processor_count)
    ref = S._attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=43).float().clone()  # scored kernel as reference
    graphs, errs = {}, {}
    for v in [43] + list(NEW):
        try:
            gr, o = cap(lambda v=v: S._attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=v))
            errs[v] = ((o.float() - ref).norm() / ref.norm()).item()
            graphs[v] = gr
        except Exception as e:
            print('RUN_FAIL', v, B, L, ctx, str(e)[:150], flush=True)
    names = list(graphs); ts = {k: [] for k in names}
    for rep in range(12):
        for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
    med = {k: statistics.median(v) for k, v in ts.items()}
    print('ROW', json.dumps(dict(B=B, L=L, ctx=ctx, nsplit=ns, base43_us=round(med[43], 1), err43=f"{errs[43]:.2e}",
          **{f'em{NEW[k]}': dict(gain=round(med[43] / v, 4), err=f"{errs[k]:.2e}") for k, v in med.items() if k != 43})), flush=True)
print('DONE', flush=True)
