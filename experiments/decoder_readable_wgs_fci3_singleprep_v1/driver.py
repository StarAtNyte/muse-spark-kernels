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
# Build and prepare only the baseline; the candidate changes this one CUDA define.
alt_src=S.MOE_CUDA_SRC
needle="#define OKC_WGSFCI 3  // test one earlier deferred flush on the readable decoder"
assert alt_src.count(needle)==1
S.MOE_CUDA_SRC=alt_src.replace(needle,"#define OKC_WGSFCI 4")
w=R.generate_weights(R.WEIGHTS_SEED)
st=S.prepare(w)
# Prime and preserve the exact b4 module that the baseline dispatch uses.
base=S._moe(32, st)
S.MOE_CUDA_SRC=alt_src
S._MA_EXTS={}
sfx=tuple(st["ma_sfx"])
assert tuple(S._main_defines(sfx))==tuple(S.MOE_DEFINES)
alt=S._RouterExt(S._ma_ext(tuple(S.MOE_DEFINES)+("-DOKC_B4FIN=1",)+sfx))
if S.MOE_HAS_PDL: alt.okc_set_pdl(0)
flush=torch.zeros(256*1024*1024//4,device="cuda")
def cap(M,d,st):
 o=M.kernel_fn(d,st); torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g): o=M.kernel_fn(d,st)
 g.replay(); torch.cuda.synchronize(); return g,o
def t(g):
 flush.sum(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b)*1000
cases=[dict(batch=32,q_len=1,layer="sliding")]
for i,c in enumerate(cases):
 for seed in (5101,5102):
  cc=dict(ctx=80000,page=1,dist="soft",spread="agentic",seed=seed,**c); ls=R.case_fields(cc,w)["local_start"]; d=R.generate_input(**cc,local_start=ls); runs={}
  for name,ext in (("base",base),("cand",alt)):
   st["ma_ext"]["b4"]=ext
   try:
    g,o=cap(S,d,st); ok,msg=R.check_implementation(d,R.collect(d,o),w); runs[name]=(g,ok,msg)
   except Exception as e: runs[name]=(None,False,str(e))
  if any(v[0] is None for v in runs.values()): print("ROW",json.dumps(dict(seed=seed,error={n:str(v[2])[:150] for n,v in runs.items()})),flush=True); continue
  ts={n:[] for n in runs}
  for rep in range(12):
   for n in (("base","cand") if rep%2==0 else ("cand","base")): ts[n].append(t(runs[n][0]))
  med={n:statistics.median(v) for n,v in ts.items()}
  print("ROW",json.dumps(dict(seed=seed,base=round(med["base"],2),cand=round(med["cand"],2),gain=round(med["base"]/med["cand"],4),correct={n:v[1] for n,v in runs.items()})),flush=True)
print("DONE",flush=True)
