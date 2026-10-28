"""Bounded H200 launch-plan sweep for vjeux's slowest scored case: full verify B32."""
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
sys.meta_path.insert(0,StubSGLang());sys.path.insert(0,'/w/kernels/muse_attention_decode')
import reference as R
spec=importlib.util.spec_from_file_location('cand','/w/run/sub.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
flush=torch.empty(256*1024*1024//4,device='cuda',dtype=torch.int32)
data=R.generate_input(32,80000,'tp8','full',56321,q_len=8,spread='agentic',dist='coherent',page=1)
def cap(fn):
    out=fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
def paired(a,b,reps=20):
    at=[];bt=[];rat=[]
    for i in range(reps):
        if i&1:y,x=elapsed(b),elapsed(a)
        else:x,y=elapsed(a),elapsed(b)
        at.append(x);bt.append(y);rat.append(x/y)
    return statistics.median(rat),statistics.median(at),statistics.median(bt)
base_plan=m._plan(32,1,8,0,data['page_table'].shape[1],1,data['qkv'].device)
configs=[('default',base_plan,False),('flat1_s132',{**base_plan,'n_splits':132,'flat':1,'cs':0,'ncl':1,'nc':1,'coop':1},False),('flat1_s396',{**base_plan,'n_splits':396,'flat':1,'cs':0,'ncl':1,'nc':1,'coop':0},False),('flat1_s528',{**base_plan,'n_splits':528,'flat':1,'cs':0,'ncl':1,'nc':1,'coop':0},False),('perreq_s8',{**base_plan,'n_splits':8,'flat':0,'cs':0,'ncl':1,'nc':1,'coop':1},False),('perreq_s16',{**base_plan,'n_splits':16,'flat':0,'cs':0,'ncl':1,'nc':1,'coop':0},False),('perreq_s8_nocoop',{**base_plan,'n_splits':8,'flat':0,'cs':0,'ncl':1,'nc':1,'coop':0},False),('default_pdl',base_plan,True)]
base,bo=cap(lambda:m._launch(data,**base_plan))
ok,msg=R.check_implementation(data,R.collect(data,bo))
if not ok:raise AssertionError(('default reference',msg))
print('BASE',json.dumps(base_plan),flush=True)
for name,plan,pdl in configs[1:]:
  try:
    run=lambda:m._launch(data,pdl=pdl,**plan)
    g,o=cap(run);ok,msg=R.check_implementation(data,R.collect(data,o))
    if not ok:raise AssertionError(msg)
    ratio,b,c=paired(base,g)
    print('RESULT',json.dumps({'name':name,'plan':plan,'pdl':pdl,'speedup':ratio,'base_us':b,'candidate_us':c}),flush=True)
    del g,o
  except Exception as e:print('FAIL',name,type(e).__name__,str(e)[:300],flush=True)
print('ATTENTION_PLAN_SCREEN_DONE',torch.cuda.get_device_name(),flush=True)
