"""Paired H200 screen for the q_len-1 flat virtual-boundary padding."""
import importlib.util,json,statistics,sys
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
sys.meta_path.insert(0,StubSGLang());sys.path.insert(0,"/w/kernels/muse_attention_decode")
import reference as R
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=load("base","/w/kernels/muse_attention_decode/submission.py")
cand=load("cand","/w/run/sub.py")
flush=torch.empty(256*1024*1024//4,dtype=torch.int32,device="cuda")
def graph(m,data):
    out=m._launch(data);torch.cuda.synchronize();ok,msg=R.check_implementation(data,R.collect(data,out))
    if not ok:raise AssertionError((m.__name__,msg))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=m._launch(data)
    g.replay();torch.cuda.synchronize();return g
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for seed in (539,1427,2753):
    data=R.generate_input(32,80000,"tp8","full",seed,q_len=1,spread="agentic",dist="coherent",page=1)
    gb=graph(base,data);gc=graph(cand,data);bt=[];ct=[];rr=[]
    for i in range(24):
        if i&1:c,b=elapsed(gc),elapsed(gb)
        else:b,c=elapsed(gb),elapsed(gc)
        bt.append(b);ct.append(c);rr.append(b/c)
    print("RESULT",json.dumps({"seed":seed,"base_us":statistics.median(bt),"candidate_us":statistics.median(ct),"speedup":statistics.median(rr)}),flush=True)
    del data,gb,gc
print("ATTN_SEGK_SCREEN_DONE",torch.cuda.get_device_name(),flush=True)
