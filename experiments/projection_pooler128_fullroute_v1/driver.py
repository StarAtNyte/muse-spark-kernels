"""Screen alternate frozen H200 cuBLASLt descriptors against final v18 routes."""
import importlib.util, json, statistics, sys
import torch
from unittest.mock import MagicMock
class _F:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('sglang','sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self,module): pass
sys.meta_path.insert(0,_F())
sys.path.insert(0,'/w/kernels/muse_bf16_proj')
import reference as R
spec=importlib.util.spec_from_file_location('candidate','/w/run/sub.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
weights=R.generate_weights(1234)
state=m.prepare(weights)
flush=torch.empty(256*1024*1024//4,device='cuda',dtype=torch.int32)

def graph(fn):
    fn();torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    g.replay();torch.cuda.synchronize()
    return g,out

def time_us(g):
    flush.add_(1)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize()
    return a.elapsed_time(b)*1000

def paired(a,b,reps=16):
    av=[];bv=[]
    for i in range(reps):
        if i&1: bv.append(time_us(b));av.append(time_us(a))
        else: av.append(time_us(a));bv.append(time_us(b))
    return statistics.median(av),statistics.median(bv)

print('ENV',torch.__version__,torch.cuda.get_device_name(),flush=True)
key=(128,'pooler')
plan,base_idx=state['new_lt']['configs'][key]
cfg=[66,15,1,0,0,0,35,0,2]
alt_idx=plan.restore(cfg)
assert alt_idx>=0 and plan.metadata()[alt_idx][:9]==cfg
for seed in (6121,6122,6123,6124):
    data=R.generate_input(128,'pooler',seed)
    graphs={}; outputs={}
    for name,index in [('base',base_idx),('candidate',alt_idx)]:
        state['new_lt']['configs'][key]=(plan,index)
        graph_,out=graph(lambda:m.kernel_fn(data,state))
        ok,msg=R.check_implementation(data,out,weights)
        print('CHECK',seed,name,ok,str(msg)[:150],flush=True)
        if not ok:raise RuntimeError('reference check failed')
        graphs[name]=graph_; outputs[name]=out
    av,bv=paired(graphs['base'],graphs['candidate'],24)
    print('RESULT',json.dumps(dict(seed=seed,base_us=av,candidate_us=bv,ratio=av/bv)),flush=True)
    state['new_lt']['configs'][key]=(plan,base_idx)
print('FULL_ROUTE_DONE',flush=True)
