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
baseinst,basecfg=m.V18_SMALL[('pooler',1)];cfg=(1,16,8,1)
inst=baseinst[:10]+cfg+baseinst[14:]
ext=m.v18s_build([inst],'pooler1_geometry_confirm')
packed=m.v18s_pack_format(weights['pooler'],1)
if packed is None: raise RuntimeError('lossless packing failed')
sp=m.v18s_stream(packed,inst[2],*cfg)
if sp is None: raise RuntimeError('candidate stream packing failed')
sp=dict(sp,inst_id=0,cfg=cfg,rows=weights['pooler'].shape[0])
assert state['v18']['plans'][('pooler',1)]['cfg']==basecfg
alt=dict(state);v18=dict(state['v18']);v18['plans']=dict(v18['plans']);v18['plans'][('pooler',1)]=sp;v18['ext']=ext;alt['v18']=v18
print('PLAN',cfg,'BASE',basecfg,flush=True)
import random
for seed in (419,7919,22073,49151):
    data=R.generate_input(1,'pooler',seed);graphs={};outs={}
    for label,st in (('base1',state),('base2',state),('candidate',alt)):
        g,out=graph(lambda st=st:m.kernel_fn(data,st));ok,msg=R.check_implementation(data,out,weights)
        print('CHECK',seed,label,ok,str(msg)[:160],flush=True)
        if not ok: raise RuntimeError((label,msg))
        graphs[label]=g;outs[label]=out
    vals={k:[] for k in graphs};rat={k:[] for k in graphs if k!='base1'};rng=random.Random(seed)
    for rep in range(32):
        order=list(graphs);rng.shuffle(order);row={k:time_us(graphs[k]) for k in order}
        for k,v in row.items():vals[k].append(v)
        for k in rat:rat[k].append(row['base1']/row[k])
    print('RESULT',json.dumps({'seed':seed,'medians_us':{k:statistics.median(v) for k,v in vals.items()},'paired_ratios':{k:statistics.median(v) for k,v in rat.items()},'samples':rat}),flush=True)
print('DONE',flush=True)
