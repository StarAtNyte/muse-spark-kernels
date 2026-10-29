"""Decoder dense ops at T=1/8/16/32: the decoder's own tuned route (as _after_attention / kernel_fn call it) vs the
embedded scored projection stack. Isolated cold-L2 graph medians (event-timed)."""
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
mp = st.get('mm_pick', {})
print('PICKS', {f'{k[0]}-{k[1]}': str(v) for k, v in mp.items() if isinstance(k, tuple) and k[0] in (1, 8, 16, 32)}, flush=True)
def proj(x, layer, T, n, k):
    return S._proj_transfer.kernel_fn({'x': x, 'config': {'m': T, 'n': n, 'k': k, 'layer': layer}}, pt)
for T in (1, 8, 16, 32):
    row = {'T': T}
    xh = torch.randn(T, 8192, device='cuda').to(torch.bfloat16)
    xo = torch.randn(T, 1024, device='cuda').to(torch.bfloat16)
    xu = torch.randn(T, 4096, device='cuda').to(torch.bfloat16)
    qpick = mp.get((T, 'qkv'), True)
    def cur_qkv():
        if qpick == 'bfp' and T <= S.BFP_MAX_T and st.get('bfp') is not None:
            return S._bfp_gemm(st, 'qkv', xh, T, st['qkv'].shape[0], 8192)
        if qpick is True or qpick == 'bfp': return torch.mm(xh, st['qkv'].t())
        return S.dense(xh, st['qkv'], T)
    row['qkv'] = (round(med(cur_qkv), 2), round(med(lambda: proj(xh, 'qkv_gate', T, 2304, 8192)), 2))
    row['o'] = (round(med(lambda: S.dense(xo, st['o'], T, st, 'o')), 2), round(med(lambda: proj(xo, 'o', T, 8192, 1024)), 2))
    wu = st['unpool_slice'] if T <= 8 else st['unpool']
    row['un'] = (round(med(lambda: S.dense(xu, wu, T, st, 'un')), 2), round(med(lambda: proj(xu, 'unpooler', T, wu.shape[0], 4096)), 2))
    ppick = mp.get((T, 'pr'), mp.get((S._bucket(T), 'pr'), False))
    def cur_pr():
        if ppick == 'bfpf' and T <= 32 and st.get('bfp') is not None: return S.bfp_fused_pooler_router(xh, st['bfp'])
        if ppick == 'bfp' and st.get('bfp') is not None:
            return (S._bfp_gemm(st, 'pool', xh, T, 4096, 8192), S._bfp_gemm(st, 'router', xh, T, 256, 8192))
        return S.dense(xh, st['pr'], T, st, 'pr')
    row['pr_cur'] = round(med(cur_pr), 2)
    row['pool_proj'] = round(med(lambda: proj(xh, 'pooler', T, 4096, 8192)), 2)
    row['router_proj'] = round(med(lambda: proj(xh, 'router', T, 256, 8192)), 2)
    row['pool_router_proj_seq'] = round(med(lambda: (proj(xh, 'pooler', T, 4096, 8192), proj(xh, 'router', T, 256, 8192))), 2)
    row['picks'] = dict(qkv=str(qpick), pr=str(ppick))
    print('ROW', json.dumps(row), flush=True)
print('DONE', flush=True)
