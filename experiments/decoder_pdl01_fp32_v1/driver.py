import importlib.util, json, statistics, sys, torch
from unittest.mock import MagicMock
class StubSGLang:
 def find_spec(self,name,path=None,target=None):
  if name.split(".")[0] in ("sglang","sgl_kernel"):
   import importlib.machinery; return importlib.machinery.ModuleSpec(name,self,is_package=True)
 def create_module(self,spec):
  m=MagicMock(); m.__path__=[]; m.__name__=spec.name; m.__spec__=spec; return m
 def exec_module(self,m): pass
sys.meta_path.insert(0,StubSGLang())
sys.path.insert(0,"/w/kernels/muse_decode_layer")
import reference as R
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
S=load("candidate", "/w/run/sub.py")
w=R.generate_weights(R.WEIGHTS_SEED)
st=S.prepare(w)
flush=torch.zeros(256*1024*1024//4,device="cuda")
def cap(M,d,st):
 o=M.kernel_fn(d,st); torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g): o=M.kernel_fn(d,st)
 g.replay(); torch.cuda.synchronize(); return g,o
def t(g):
 flush.sum(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b)*1000
cases=[dict(batch=b,q_len=q,layer="sliding") for b in (1,16,32) for q in (1,8)]
for c in cases:
 for seed in (5101,5102):
  cc=dict(ctx=80000,page=1,dist="soft",spread=("agentic" if c["batch"]>1 else "fixed"),seed=seed,**c); ls=R.case_fields(cc,w)["local_start"]; d=R.generate_input(**cc,local_start=ls); runs={}
  for name,n in (("p2",2),("p1",1),("p0",0)):
   S.BFP_PDL_N=n
   try:
    g,o=cap(S,d,st); ok,msg=R.check_implementation(d,R.collect(d,o),w); runs[name]=(g,ok,msg)
   except Exception as e: runs[name]=(None,False,str(e))
  if any(v[0] is None for v in runs.values()): print("ROW",json.dumps(dict(case=c,seed=seed,error={n:str(v[2])[:120] for n,v in runs.items()})),flush=True); continue
  ts={n:[] for n in runs}
  for rep in range(12):
   for n in (("p2","p1","p0") if rep%2==0 else ("p0","p1","p2")): ts[n].append(t(runs[n][0]))
  med={n:statistics.median(v) for n,v in ts.items()}
  print("ROW",json.dumps(dict(case=c,seed=seed,**{n:round(v,2) for n,v in med.items()},gain1=round(med["p2"]/med["p1"],4),gain0=round(med["p2"]/med["p0"],4),correct={n:v[1] for n,v in runs.items()})),flush=True)
print("DONE",flush=True)
