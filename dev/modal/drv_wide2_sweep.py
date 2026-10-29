"""Sweep the projection file's wide2 wgmma kernel (bn, split, stages, hint) on M128/M256 shapes vs cuBLAS (torch.mm);
paired cold-L2 CUDA-graph medians; every config checked against torch.mm."""
import importlib.util, itertools, json, statistics, sys
import torch
from unittest.mock import MagicMock
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
import reference as R
WT = R.generate_weights(R.WEIGHTS_SEED) if hasattr(R, 'WEIGHTS_SEED') else R.generate_weights(1234)
ST = S.prepare(WT)
ext = S.wide2_build()
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
torch.manual_seed(0)
SH = {'qkv_gate': (2304, 8192), 'o': (8192, 1024), 'pooler': (4096, 8192), 'unpooler': (8192, 4096)}
def gtime(fn, n=25):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(n):
        flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
for m, layer in ((256, 'qkv_gate'), (256, 'o'), (256, 'unpooler'), (128, 'o'), (128, 'unpooler'), (256, 'pooler')):
    n, k = SH[layer]
    d = R.generate_input(m, layer, 77)
    x = d['x'].contiguous(); w = WT[layer].contiguous()
    ref = torch.mm(x, w.t())
    t_cublas = gtime(lambda: S.kernel_fn(d, ST))  # the submission's own route for this shape
    res = []
    for bn, split, stages, hint in itertools.product((64, 128, 256), (1, 2, 3, 4, 6, 8), (3, 4, 5, 6, 8), (0, 1)):
        y = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
        try:
            rc = ext.launch(x, w, y, bn, split, stages, hint, 0)
            torch.zeros(1, device='cuda').add_(1); torch.cuda.synchronize()  # surfaces (and clears) a launch error
        except Exception as e:
            try: torch.zeros(1, device='cuda').add_(1); torch.cuda.synchronize()
            except Exception: pass
            continue
        if rc: continue
        err = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
        if err > 1e-2: continue
        t = gtime(lambda: ext.launch(x, w, y, bn, split, stages, hint, 0))
        res.append((t, bn, split, stages, hint, err))
    res.sort()
    cur = S.CANDIDATE_CONFIGS.get((m, layer))
    print('ROW', json.dumps(dict(m=m, layer=layer, cublas_us=round(t_cublas, 2), n_ok=len(res),
          best=[dict(us=round(r[0], 2), bn=r[1], split=r[2], stages=r[3], hint=r[4]) for r in res[:4]],
          gain_vs_cublas=round(t_cublas / res[0][0], 3) if res else None, candidate_cfg=cur)), flush=True)
print('DONE', flush=True)
