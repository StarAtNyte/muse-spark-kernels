"""Decoder dense ops at T=128/256: current route (torch.mm, fp32 out; qkv bf16; pooler+router as one [4608, 8192]
matmul) vs the embedded scored projection stack (_proj_transfer). Isolated cold-L2 graph medians; error vs fp32."""
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
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
pt = st['projection_transfer']
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def med(fn, n=25):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(n):
        flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
W = {'qkv_gate': w['qkv_gate'], 'o': w['o'], 'pooler': w['pooler'], 'unpooler': w['unpooler'], 'router': w['router']}
for m in (128, 256):
    row = {'m': m}
    for layer, Wt in W.items():
        n, k = Wt.shape
        x = torch.randn(m, k, device='cuda').to(torch.bfloat16)
        ref = x.double() @ Wt.double().t()
        def proj(): return S._proj_transfer.kernel_fn({'x': x, 'config': {'m': m, 'n': n, 'k': k, 'layer': layer}}, pt)
        try:
            o = proj(); torch.cuda.synchronize()
            if not torch.is_tensor(o): row[layer] = 'sglang-route'; continue
            err = ((o.double() - ref).norm() / ref.norm()).item()
            row[layer] = dict(proj_us=round(med(proj), 2), err=f'{err:.1e}')
        except Exception as e:
            row[layer] = 'fail: ' + str(e)[:60]
            try: torch.zeros(1, device='cuda').add_(1); torch.cuda.synchronize()
            except Exception: pass
    xh = torch.randn(m, 8192, device='cuda').to(torch.bfloat16)
    xo = torch.randn(m, 1024, device='cuda').to(torch.bfloat16)
    xu = torch.randn(m, 4096, device='cuda').to(torch.bfloat16)
    row['cur_qkv_us'] = round(med(lambda: torch.mm(xh, st['qkv'].t())), 2)
    row['cur_o_us'] = round(med(lambda: torch.mm(xo, st['o'].t(), out_dtype=torch.float32)), 2)
    row['cur_pr_us'] = round(med(lambda: torch.mm(xh, st['pr'].t(), out_dtype=torch.float32)), 2)
    row['cur_unpool_us'] = round(med(lambda: torch.mm(xu, st['unpool'].t(), out_dtype=torch.float32)), 2)
    print('ROW', json.dumps(row), flush=True)
print('DONE', flush=True)
