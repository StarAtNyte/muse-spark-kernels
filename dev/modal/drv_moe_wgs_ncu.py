"""Capture the scored WGS MoE kernel for Nsight Compute."""
import importlib.util, sys
from unittest.mock import MagicMock
import torch

class StubSGLang:
    def find_spec(self,name,path=None,target=None):
        if name.split(".")[0] in ("sglang","sgl_kernel"):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self,module): pass

sys.meta_path.insert(0,StubSGLang())
sys.path.insert(0,"/w/kernels/muse_moe_w4a16")
import reference as R
sp=importlib.util.spec_from_file_location("sub","/w/run/sub.py")
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.TINY_DEFINES=();m.TINY2_DEFINES=();m.MID_DEFINES=()
weights=R.generate_weights(5252);state=m.prepare(weights)
data=R.generate_input(32,"ep8",20491)
out=m.kernel_fn(data,state);torch.cuda.synchronize()
ok,msg=R.check_implementation(data,out,weights)
print("CHECK",ok,msg,flush=True);assert ok,msg
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g): out=m.kernel_fn(data,state)
g.replay();torch.cuda.synchronize()
torch.cuda.profiler.start();g.replay();torch.cuda.synchronize();torch.cuda.profiler.stop()
print("NCU_CAPTURE_DONE",out.shape,flush=True)
