"""Paired H200 screen: flat versus per-request full-context decode, B=32."""
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
sys.meta_path.insert(0,StubSGLang())
sys.path.insert(0,"/w/kernels/muse_attention_decode")
import reference as R
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=load("base","/w/kernels/muse_attention_decode/submission.py")
cand=load("cand","/w/run/sub.py")
flush=torch.empty(256*1024*1024//4,dtype=torch.int32,device="cuda")
def graph(m,data,plan=None):
    call=lambda:m._launch(data,**(plan or {}))
    out=call();torch.cuda.synchronize();ok,msg=R.check_implementation(data,R.collect(data,out))
    if not ok:raise AssertionError((m.__name__,plan,msg))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=call()
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
def paired(gb,gc):
    bt=[];ct=[];rat=[]
    for i in range(20):
        if i&1:c,b=elapsed(gc),elapsed(gb)
        else:b,c=elapsed(gb),elapsed(gc)
        bt.append(b);ct.append(c);rat.append(b/c)
    return statistics.median(bt),statistics.median(ct),statistics.median(rat)
for seed in (539,1427):
    data=R.generate_input(32,80000,"tp8","full",seed,q_len=1,spread="agentic",dist="coherent",page=1)
    gb,ob=graph(base,data)
    plans=[("perreq8",cand._plan(32,1,1,0,data["page_table"].shape[1],1,data["qkv"].device)),
           ("perreq16",{"n_splits":16,"flat":0,"cs":0,"ncl":1,"nc":1,"coop":0}),
           ("perreq4",{"n_splits":4,"flat":0,"cs":0,"ncl":1,"nc":1,"coop":0})]
    print("BASE_PLAN",json.dumps(base._plan(32,1,1,0,data["page_table"].shape[1],1,data["qkv"].device)),flush=True)
    for name,plan in plans:
        gc,oc=graph(cand,data,plan)
        b,c,r=paired(gb,gc)
        print("RESULT",json.dumps({"seed":seed,"name":name,"plan":plan,"base_us":b,"candidate_us":c,"speedup":r}),flush=True)
        del gc,oc
    del data,gb,ob
print("ATTN_FULL_DECODE32_SCREEN_DONE",torch.cuda.get_device_name(),flush=True)
