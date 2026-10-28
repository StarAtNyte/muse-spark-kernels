"""Local flat-grid CTA-count sweep for full-context decode B=32, q_len=1."""
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
sp=importlib.util.spec_from_file_location("sub","/w/run/sub.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
flush=torch.empty(256*1024*1024//4,dtype=torch.int32,device="cuda")
def graph(data,plan):
    call=lambda:m._launch(data,**plan)
    out=call();torch.cuda.synchronize();ok,msg=R.check_implementation(data,R.collect(data,out))
    if not ok:raise AssertionError((plan,msg))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=call()
    g.replay();torch.cuda.synchronize();return g
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
def paired(a,b):
    at=[];bt=[];rr=[]
    for i in range(24):
        if i&1:y,x=elapsed(b),elapsed(a)
        else:x,y=elapsed(a),elapsed(b)
        at.append(x);bt.append(y);rr.append(x/y)
    return statistics.median(at),statistics.median(bt),statistics.median(rr)
for seed in (539,1427,2753):
    data=R.generate_input(32,80000,"tp8","full",seed,q_len=1,spread="agentic",dist="coherent",page=1)
    base=m._plan(32,1,1,0,data["page_table"].shape[1],1,data["qkv"].device)
    gb=graph(data,base);print("BASE_PLAN",json.dumps(base),flush=True)
    for splits in (240,288,320):
        plan={**base,"n_splits":splits,"flat":1,"cs":0,"ncl":1,"nc":1,"coop":0}
        gc=graph(data,plan);b,c,s=paired(gb,gc)
        print("RESULT",json.dumps({"seed":seed,"splits":splits,"base_us":b,"candidate_us":c,"speedup":s}),flush=True);del gc
    del data,gb
print("ATTN_FULL_DECODE32_FINE_DONE",torch.cuda.get_device_name(),flush=True)
