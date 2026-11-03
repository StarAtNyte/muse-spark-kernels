"""Two-shape H200 A/B screen for the SWK transfer; not a judge score."""
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
    def exec_module(self,module): pass
sys.meta_path.insert(0,StubSGLang());sys.path.insert(0,'/w/kernels/muse_attention_decode')
import reference as R

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('/w/kernels/muse_attention_decode/submission.py','base')
cand=load('/w/run/sub.py','candidate')
flush=torch.empty(256*1024*1024//4,device='cuda',dtype=torch.int32)
def capture(mod,data):
    out=mod.kernel_fn(data);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=mod.kernel_fn(data)
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for batch in (16,32):
  per_seed=[]
  for seed in (539,1427):
    db=R.generate_input(batch,80000,'tp8','sliding',seed,spread='agentic',dist='soft',page=1)
    dc=R.generate_input(batch,80000,'tp8','sliding',seed,spread='agentic',dist='soft',page=1)
    gb,ob=capture(base,db);gc,oc=capture(cand,dc)
    ok,msg=R.check_implementation(dc,R.collect(dc,oc))
    if not ok:raise AssertionError(('candidate reference check',batch,seed,msg))
    ok,msg=R.check_implementation(db,R.collect(db,ob))
    if not ok:raise AssertionError(('base reference check',batch,seed,msg))
    bt=[];ct=[];rat=[]
    for i in range(16):
      if i&1: c,b=elapsed(gc),elapsed(gb)
      else: b,c=elapsed(gb),elapsed(gc)
      bt.append(b);ct.append(c);rat.append(b/c)
    row={'batch':batch,'seed':seed,'base_us':statistics.median(bt),'candidate_us':statistics.median(ct),'speedup':statistics.median(rat)}
    print('RESULT',json.dumps(row),flush=True);per_seed.append(row['speedup'])
    del db,dc,gb,gc,ob,oc
  print('BATCH_SUMMARY',json.dumps({'batch':batch,'median_speedup':statistics.median(per_seed)}),flush=True)
print('SWK_MODAL_SCREEN_PASSED',torch.cuda.get_device_name(),flush=True)
