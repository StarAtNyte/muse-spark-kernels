"""ncu target: scored attention, 1-request full decode and verify (cold L2 before each launch)."""
import importlib.util, sys
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
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
for q in (1, 8):
    d = R.generate_input(1, 80000, 'tp8', 'full', 5, q_len=q, dist='coherent', page=1)
    for _ in range(3): S.kernel_fn(d)
    torch.cuda.synchronize()
    flush.add_(1); torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f'q{q}')
    S.kernel_fn(d); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print('DONE', flush=True)
