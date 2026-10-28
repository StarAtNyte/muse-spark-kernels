"""Focused BF16 cuBLASLt heuristic screening with independent input confirmation."""
import importlib.util,json,statistics,sys,time
import torch
from torch.utils.cpp_extension import load_inline
sys.path.insert(0,'/w/kernels/muse_bf16_proj')
import reference as R

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
probe=load('/w/run/sub.py','probe')
parent=load('/w/kernels/muse_bf16_proj/submission.py','parent')
module=load_inline('muse_lt_probe',cpp_sources=probe.CPP,functions=['lt_plan','lt_run'],
    with_cuda=True,extra_ldflags=['-lcublasLt'],extra_cflags=['-O3'],verbose=False)
weights=R.generate_weights(98127)
state=parent._prepare_v5(weights)
workspace=torch.empty(8*1024*1024,device='cuda',dtype=torch.uint8)
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(fn):
    fn();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): output=fn()
    graph.replay();torch.cuda.synchronize();return graph,output

def elapsed(g):
    flush.add_(1)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();g.replay();end.record();end.synchronize()
    return start.elapsed_time(end)*1000

def pairs(base,candidate,reps):
    ratios,bt,ct=[],[],[]
    for rep in range(reps):
        if rep%2: c,b=elapsed(candidate),elapsed(base)
        else: b,c=elapsed(base),elapsed(candidate)
        ratios.append(b/c);bt.append(b);ct.append(c)
    return statistics.median(ratios),statistics.median(bt),statistics.median(ct)

for batch,layer in ((16,'qkv_gate'),(32,'pooler'),(32,'unpooler'),(128,'pooler')):
    data=R.generate_input(batch,layer,9137+batch)
    n=data['config']['n'];k=data['x'].shape[1]
    plan,count=module.lt_plan(batch,n,k,workspace.numel())
    baseline,ref=capture(lambda:parent._kernel_v5(data,state))
    def run(index):
        y=torch.empty((batch,n),device='cuda',dtype=torch.bfloat16)
        module.lt_run(plan,index,weights[layer],data['x'],y,workspace)
        return y
    results=[]
    for index in range(min(count,16)):
        try:
            graph,out=capture(lambda:run(index))
            ok,message=R.check_implementation(data,out,weights)
            if not ok: raise AssertionError(message)
            ratio,b,c=pairs(baseline,graph,8)
            row=dict(batch=batch,layer=layer,algorithm=index,speedup=ratio,baseline_us=b,candidate_us=c)
            results.append(row);print('SCREEN',json.dumps(row),flush=True)
            del graph,out
        except Exception as exc: print('FAIL',batch,layer,index,str(exc)[:250],flush=True)
    if not results: continue
    best=max(results,key=lambda r:r['speedup'])['algorithm']
    del baseline,ref
    for seed in (173,2387,7919):
        data=R.generate_input(batch,layer,seed)
        baseline,ref=capture(lambda:parent._kernel_v5(data,state))
        graph,out=capture(lambda:run(best))
        ok,message=R.check_implementation(data,out,weights)
        if not ok: raise AssertionError(message)
        ratio,b,c=pairs(baseline,graph,16)
        print('CONFIRM',json.dumps(dict(batch=batch,layer=layer,seed=seed,algorithm=best,
              speedup=ratio,baseline_us=b,candidate_us=c)),flush=True)
        del baseline,ref,graph,out
