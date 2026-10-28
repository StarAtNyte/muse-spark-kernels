import importlib.util, json, os, statistics, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('sglang','sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self,m): pass
sys.meta_path.insert(0,StubSGLang())
sys.path.insert(0,'/w/kernels/muse_moe_w4a16')
import reference as R

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('base','/w/kernels/muse_moe_w4a16/submission.py')
cand=load('cand','/w/run/sub.py')
R.N_EXPERTS=32
weights=R.generate_weights(93819)
R.N_EXPERTS=256
# Keep the reference-shaped local bank and use each source's own preparation path.
bs=base.prepare(weights); cs=cand.prepare(weights)
print('READY',torch.cuda.get_device_name(),flush=True)
flush=torch.zeros(256*1024*1024//4,device='cuda')
def cap(mod,st,data):
    mod.kernel_fn(data,st);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=mod.kernel_fn(data,st)
    g.replay();torch.cuda.synchronize();return g,out
def time(g):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for seed in (419,7919,15241):
    data_b=R.generate_input(32,'ep8',seed)
    data_c=R.generate_input(32,'ep8',seed)
    gb,ob=cap(base,bs,data_b);gc,oc=cap(cand,cs,data_c)
    err=((oc.float()-ob.float()).norm()/ob.float().norm().clamp_min(1e-20)).item()
    exact=torch.equal(oc,ob)
    print('CHECK_AGAINST_VALIDATED_WINNER',seed,'bit_equal',exact,'relative_error',err,flush=True)
    if err>0.001: raise RuntimeError('candidate differs from validated winner')
    bt=[];ct=[]
    for r in range(20):
        if r%2: ct.append(time(gc));bt.append(time(gb))
        else: bt.append(time(gb));ct.append(time(gc))
    print('RESULT',json.dumps({'seed':seed,'base_us':statistics.median(bt),'candidate_us':statistics.median(ct),'ratio':statistics.median(bt)/statistics.median(ct)}),flush=True)
print('BM4_SCREEN_PASSED',flush=True)
