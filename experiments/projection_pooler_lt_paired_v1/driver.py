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
for batch,layer in ((128,'pooler'),(256,'pooler')):
    data=R.generate_input(batch,layer,9000+batch)
    x=data['x']; n=data['config']['n']; key=(x.shape[0],layer)
    baseline,base_out=graph(lambda:m.kernel_fn(data,state))
    ok,msg=R.check_implementation(data,base_out,weights)
    if not ok: raise AssertionError(f'baseline failed {key}: {msg}')
    baseline_us=time_us(baseline)
    ltstate=m.tt_lt__lt_state(weights)
    entries=[m.tt_lt__LT_H200_FIXED[key], *m.tt_lt__LT_SEARCH_SEEDS.get(key,[])]
    configs=list(dict.fromkeys((layout, epilogue, tuple(cfg)) for layout, epilogue, cfg in entries))
    outcomes=[]
    plans=[]
    for layout,epilogue,cfg in configs:
        plan=m.tt_lt__lt_plan(ltstate,layer,x.shape[0],layout,epilogue)
        plans.append(plan)
        index=plan.restore(list(cfg))
        if index<0 or plan.metadata()[index][:9]!=list(cfg):
            print('SKIP',key,cfg,'not available',flush=True);continue
        out=torch.empty((x.shape[0],n),device=x.device,dtype=torch.bfloat16)
        def invoke():
            rc=plan.run(x,out,index)
            if rc: raise RuntimeError(f'plan.run={rc} for {cfg}')
            return out
        g,candidate_out=graph(invoke)
        ok,msg=R.check_implementation(data,candidate_out,weights)
        if not ok: raise AssertionError(f'candidate failed {key} {cfg}: {msg}')
        base_t,cand_t=paired(baseline,g,12)
        rec={'batch':batch,'layer':layer,'input_rows':x.shape[0],'descriptor':list(cfg),'layout':layout,'epilogue':epilogue,
             'baseline_us':base_t,'candidate_us':cand_t,'speedup':base_t/cand_t}
        outcomes.append(rec);print('RESULT',json.dumps(rec),flush=True)
        del g,candidate_out,out
    print('BEST',json.dumps({'key':key,'baseline_us':baseline_us,
          'results':sorted(outcomes,key=lambda r:r['candidate_us'])}),flush=True)
    del baseline,base_out,plans,plan,ltstate
print('LT_STACK_SCREEN_DONE',flush=True)
