"""Native FP32 activations and weights; no BF16 conversion for the small router."""
import torch,importlib.util,json,statistics,sys
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=False
sp=importlib.util.spec_from_file_location('r_bundle','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
sys.path.insert(0,'/w/modal');from compile_split import build
mod=build('muse_decoder_fp32_router',m.CPP,m.CUDA,['fp32_router_run'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(fn):
    fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):o=fn()
    g.replay();torch.cuda.synchronize();return g,o
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for seed in (5011,49157):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    w=torch.randn(256,8192,generator=gen,device='cuda')*.045
    for bs in (1,2,4,8):
        x=torch.randn(bs,8192,generator=gen,device='cuda')
        ref=(x.double()@w.double().T).float()
        gb,ob=capture(lambda:F.linear(x,w))
        for cfg in range(5 if bs==1 else 4):
            g,o=capture(lambda:mod.fp32_router_run(w,x,cfg))
            err=((o-ref).norm()/ref.norm()).item();assert torch.isfinite(o).all() and err<1.e-5,(bs,cfg,err)
            print('CHECK',bs,seed,cfg,'PASS',err,flush=True)
            bt,ct,ratios=[],[],[]
            for rep in range(24):
                if rep%2:c,b=elapsed(g),elapsed(gb)
                else:b,c=elapsed(gb),elapsed(g)
                bt.append(b);ct.append(c);ratios.append(b/c)
            print('RESULT',json.dumps(dict(batch=bs,seed=seed,cfg=cfg,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
            del g,o
        del gb,ob,x,ref
print('DECODER_FP32_ROUTER_SCREEN_PASSED',flush=True)
