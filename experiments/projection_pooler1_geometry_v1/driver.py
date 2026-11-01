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
baseinst,basecfg=m.V18_SMALL[('pooler',1)]
cfgs=[(2,8,8,2),(2,8,16,1),(1,16,8,1),(2,16,4,2)]
instances=[baseinst[:4]+(int(cfg[3]>1),)+baseinst[5:10]+tuple(cfg)+baseinst[14:] for cfg in cfgs]
ext=m.v18s_build(instances,'pooler1_geometry_probe')
packed=m.v18s_pack_format(weights['pooler'],1)
if packed is None: raise RuntimeError('lossless packing failed')
plans={}
for i,cfg in enumerate(cfgs):
    try:
        sp=m.v18s_stream(packed,instances[i][2],*cfg)
        if sp is not None: plans[str(cfg)]=(i,cfg,sp)
        else: print('PLAN_UNAVAILABLE',cfg,flush=True)
    except Exception as e: print('PLAN_ERROR',cfg,repr(e),flush=True)
print('PLANS',list(plans),flush=True)
for seed in (419,7919):
    data=R.generate_input(1,'pooler',seed);x=data['x'];n=data['config']['n']
    def candidate(plan):
        idx,cfg,sp=plan;out=torch.empty((1,n),device=x.device,dtype=torch.bfloat16)
        rc=m.v18s_launch_raw(ext,idx,sp,x,out,n,cfg)
        if rc: raise RuntimeError(f'launch failed {rc}')
        return out
    graphs={};outs={}
    for name,fn in [('base',lambda:m.kernel_fn(data,state))]+[(key,lambda p=p:candidate(p)) for key,p in plans.items()]:
        try:
            g,out=graph(fn);ok,msg=R.check_implementation(data,out,weights)
            print('CHECK',seed,name,ok,str(msg)[:180],flush=True)
            if not ok:
                if name=='base': raise RuntimeError((name,msg))
                continue
            graphs[name]=g;outs[name]=out
        except Exception as e:
            print('RUN_FAIL',seed,name,repr(e),flush=True)
            if name=='base': raise
    import random
    vals={k:[] for k in graphs};rat={k:[] for k in graphs if k!='base'};rng=random.Random(seed)
    for rep in range(24):
        order=list(graphs);rng.shuffle(order);row={k:time_us(graphs[k]) for k in order}
        for k,v in row.items():vals[k].append(v)
        for k in rat:rat[k].append(row['base']/row[k])
    print('RESULT',json.dumps({'seed':seed,'medians_us':{k:statistics.median(v) for k,v in vals.items()},'paired_ratios':{k:statistics.median(v) for k,v in rat.items()},'samples':rat}),flush=True)
print('DONE',flush=True)
