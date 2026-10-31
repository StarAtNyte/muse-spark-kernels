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
old='q_launch_t<1,2,1,1,5,0,128,0,2304,8192,1,4,8,4,7,1,0>'
new='q_launch_t<1,4,1,1,5,0,128,0,2304,8192,1,4,8,4,7,1,0>'
assert m.b2q_tt_v10_m32x__Q_CUDA.count(old)==1
m.b2q_tt_v10_m32x__Q_CUDA=m.b2q_tt_v10_m32x__Q_CUDA.replace(old,new)
ext=m.b2q_tt_v10_m32x__q_build(); p=state['bundle2']['qkv']; assert p and p['idx']==0
for seed in (6121,6122):
    data=R.generate_input(32,'qkv_gate',seed); x=data['x']; n=data['config']['n']
    def cand():
        out=torch.empty((32,n),device=x.device,dtype=x.dtype)
        rc=m.b2q_tt_v10_m32x__q_launch_raw(ext,0,p,x,out,n,p['cfg'])
        if rc: raise RuntimeError(f'candidate launch rc={rc}')
        return out
    graphs={};outs={}
    for name,fn in [('base',lambda:m.kernel_fn(data,state)),('candidate',cand)]:
        g,out=graph(fn);ok,msg=R.check_implementation(data,out,weights)
        print('CHECK',seed,name,ok,str(msg)[:180],flush=True)
        if not ok:raise RuntimeError('reference check failed')
        graphs[name]=g;outs[name]=out
    a,b=paired(graphs['base'],graphs['candidate'],24)
    print('RESULT',json.dumps(dict(seed=seed,base_us=a,candidate_us=b,ratio=a/b)),flush=True)
print('DONE',flush=True)
