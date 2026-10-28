"""Nsight probe of the validated full-context decode path on Modal H200."""
import importlib.util
import sys
from unittest.mock import MagicMock

import torch


class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = MagicMock()
        module.__path__ = []
        module.__name__ = spec.name
        module.__spec__ = spec
        return module

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R

spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
sys.path.insert(0,'/w/modal')
from compile_split import build
sub._WG_MOD=build('muse_fp8p_joint_profile',sub._WG_CPP,sub._WG_CUDA,['wg5_run_t'],['-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo'])
data = R.generate_input(32, 80000, 'tp8', 'full', 4319, spread='agentic',
                        dist='coherent', page=1, q_len=8)
sub.kernel_fn(data)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    result = sub.kernel_fn(data)
torch.cuda.synchronize()
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
flush.add_(1)
torch.cuda.synchronize()
torch.cuda.profiler.start()
graph.replay()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print('PROFILE_FINISHED', result.shape, torch.cuda.get_device_name())
