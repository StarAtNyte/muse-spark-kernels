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
for batch,ctx,layer,page,scale,dist in ((8,10,'sliding',64,1.,'edges'),(17,8192,'full',64,1.,'coherent'),(33,9000,'full',64,1/64,'soft'),(64,5000,'sliding',1,1.,'soft')):
 data=R.generate_input(batch,ctx,'tp8',layer,173+batch,page=page,kv_scale=scale,dist=dist)
 out=sub.kernel_fn(data)
 ok,message=R.check_implementation(data,R.collect(data,out))
 print('NEARBY_CHECK',batch,ctx,layer,page,scale,ok,message,flush=True)
 assert ok,message
 del data,out
data=R.generate_input(32,80000,'tp8','full',4319,spread='agentic',dist='coherent',page=1)
sub.kernel_fn(data);torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):result=sub.kernel_fn(data)
flush=torch.zeros(256*1024*1024//4,device='cuda');flush.add_(1);torch.cuda.synchronize()
torch.cuda.profiler.start();graph.replay();torch.cuda.synchronize();torch.cuda.profiler.stop()
print('TRANSPOSE_PROFILE_FINISHED',result.shape,flush=True)
