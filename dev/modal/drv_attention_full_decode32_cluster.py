"""Screen Hopper DSMEM cluster reductions for full decode B=32."""
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
    out=m._launch(data,**plan);torch.cuda.synchronize();ok,msg=R.check_implementation(data,R.collect(data,out))
    if not ok:raise AssertionError((plan,msg))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=m._launch(data,**plan)
    g.replay();torch.cuda.synchronize();return g
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
def paired(a,b):
    at=[];bt=[];rr=[]
    for i in range(20):
        if i&1:y,x=elapsed(b),elapsed(a)
        else:x,y=elapsed(a),elapsed(b)
        at.append(x);bt.append(y);rr.append(x/y)
    return statistics.median(at),statistics.median(bt),statistics.median(rr)
plans=[
 ("cluster8_splits32",dict(n_splits=32,flat=0,cs=8,ncl=4,nc=4,coop=0)),
 ("cluster8_splits16",dict(n_splits=16,flat=0,cs=8,ncl=2,nc=2,coop=0)),
 ("cluster16_splits32",dict(n_splits=32,flat=0,cs=16,ncl=2,nc=2,coop=0)),
 ("cluster4_splits16",dict(n_splits=16,flat=0,cs=4,ncl=4,nc=4,coop=0)),
]
for seed in (539,1427):
    data=R.generate_input(32,80000,"tp8","full",seed,q_len=1,spread="agentic",dist="coherent",page=1)
    base=m._plan(32,1,1,0,data["page_table"].shape[1],1,data["qkv"].device)
    gb=graph(data,base);print("BASE_PLAN",json.dumps(base),flush=True)
    for name,plan in plans:
      try:
        gc=graph(data,plan);b,c,s=paired(gb,gc)
        print("RESULT",json.dumps({"seed":seed,"name":name,"base_us":b,"candidate_us":c,"speedup":s}),flush=True);del gc
      except Exception as e:print("FAIL",json.dumps({"seed":seed,"name":name,"error":str(e)[:300]}),flush=True)
    del data,gb
print("ATTN_CLUSTER_SCREEN_DONE",torch.cuda.get_device_name(),flush=True)
