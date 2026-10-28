"""Profile the current full-context verify attention kernel on H200."""
import importlib.util, sys
from unittest.mock import MagicMock
import torch

class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec
        return m
    def exec_module(self, module): pass

sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
data = R.generate_input(32, 80000, 'tp8', 'full', 4319, q_len=8,
                        spread='agentic', dist='coherent', page=1)
out = sub.kernel_fn(data); torch.cuda.synchronize()
ok, msg = R.check_implementation(data, R.collect(data, out))
print('CHECK', ok, msg, flush=True)
assert ok, msg
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph): out = sub.kernel_fn(data)
graph.replay(); torch.cuda.synchronize()
flush = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.int32, device='cuda')
flush.add_(1); torch.cuda.synchronize()
torch.cuda.profiler.start(); graph.replay(); torch.cuda.synchronize(); torch.cuda.profiler.stop()
print('NCU_CAPTURE_DONE', out.shape, torch.cuda.get_device_name(), flush=True)
