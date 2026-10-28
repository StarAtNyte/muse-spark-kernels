"""Paired H200 graph comparison of vjeux public attention and canonical, with reference checks."""
import importlib.util, json, statistics, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('sglang','sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self,module):pass
sys.meta_path.insert(0,StubSGLang())
sys.path.insert(0,'/w/kernels/muse_attention_decode')
import reference as R

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('/w/kernels/muse_attention_decode/submission.py','base')
cand=load('/w/run/sub.py','candidate')
flush=torch.empty(256*1024*1024//4,device='cuda',dtype=torch.int32)
def graph(module,data):
    out=module.kernel_fn(data);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=module.kernel_fn(data)
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for batch in (1,16,32):
  for layer,dist in (('sliding','soft'),('full','coherent')):
    for seed in (539,1427):
      data=R.generate_input(batch,80000,'tp8',layer,seed,spread='agentic',dist=dist,page=1)
      gb,ob=graph(base,data);gc,oc=graph(cand,data)
      ok,msg=R.check_implementation(data,R.collect(data,oc))
      if not ok:raise AssertionError(('candidate reference check',batch,layer,seed,msg))
      # Also check the current source, independently, with freshly generated state.
      data2=R.generate_input(batch,80000,'tp8',layer,seed,spread='agentic',dist=dist,page=1)
      _,ob2=graph(base,data2);ok,msg=R.check_implementation(data2,R.collect(data2,ob2))
      if not ok:raise AssertionError(('base reference check',batch,layer,seed,msg))
      ratios=[];bt=[];ct=[]
      for i in range(16):
        if i&1:c,b=elapsed(gc),elapsed(gb)
        else:b,c=elapsed(gb),elapsed(gc)
        ratios.append(b/c);bt.append(b);ct.append(c)
      print('RESULT',json.dumps(dict(batch=batch,layer=layer,seed=seed,base_us=statistics.median(bt),candidate_us=statistics.median(ct),speedup=statistics.median(ratios))),flush=True)
      del data,data2,gb,gc,ob,oc,ob2
print('SOTA_COMPARE_PASSED',torch.cuda.get_device_name(),flush=True)
