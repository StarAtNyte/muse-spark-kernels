import importlib.util, sys, statistics, json
from unittest.mock import MagicMock
import torch
class StubSGLang:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('sglang','sgl_kernel'):
   import importlib.machinery
   return importlib.machinery.ModuleSpec(name,self,is_package=True)
 def create_module(self,spec):
  m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
 def exec_module(self,module): pass
sys.meta_path.insert(0,StubSGLang());sys.path.insert(0,'/w/kernels/muse_attention_decode')
import reference as R

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('base','/w/kernels/muse_attention_decode/submission.py')
cand=load('cand','/w/run/sub.py')
def capture(mod,data):
 mod.kernel_fn(data);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g): out=mod.kernel_fn(data)
 torch.cuda.synchronize();return g,out
def elapsed(g):
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 start.record();g.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000
for batch in (16,32):
 for seed in (4319,7319):
  data=R.generate_input(batch,80000,'tp8','full',seed,spread='agentic',dist='coherent',page=1,q_len=8)
  gb,ob=capture(base,data);gc,oc=capture(cand,data)
  ok,msg=R.check_implementation(data,R.collect(data,oc))
  if not ok: raise AssertionError((batch,seed,msg))
  flush=torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda')
  ratios=[];bt=[];ct=[]
  for i in range(14):
   flush.fill_(i)
   if i%2: c,b=elapsed(gc),elapsed(gb)
   else: b,c=elapsed(gb),elapsed(gc)
   ratios.append(b/c);bt.append(b);ct.append(c)
  print('RESULT',json.dumps(dict(batch=batch,seed=seed,correct=True,ratio=statistics.median(ratios),base_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
  del gb,gc,ob,oc,data
print('SCREEN_PASS',torch.cuda.get_device_name(),flush=True)
