"""BF16 normalization and route metadata, paired cold-L2 graphs."""
import torch,triton,importlib.util,sys,json,statistics
import torch.nn.functional as F
spec=importlib.util.spec_from_file_location('norm_bundle','/w/run/sub.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
open('/w/run/norm_impl.py','w').write('import torch\nimport triton\nimport triton.language as tl\n'+m.NORM)
spec=importlib.util.spec_from_file_location('norm_impl','/w/run/norm_impl.py');norm=importlib.util.module_from_spec(spec);spec.loader.exec_module(norm)
sys.path.insert(0,'/w/modal');from compile_split import build
mod=build('muse_norm_route_probe','',m.CUDA,['wm_set_pdl_t','wm_route','wm_norm_route'],['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a'])
mod.wm_set_pdl_t(1)
flush=torch.zeros(256*1024*1024//4,device='cuda')
nols=torch.empty(0,device='cuda',dtype=torch.int32)
def capture(x,w,ids,name):
    bs=x.shape[0];lmax=(bs+7)//8*8;grp=8 if bs<=8 else 16 if bs<=16 else 32
    cnt=torch.empty(32,device='cuda',dtype=torch.int32);lst=torch.empty(32*lmax,device='cuda',dtype=torch.int32)
    act=torch.empty(512,device='cuda',dtype=torch.int32);nact=torch.empty(1,device='cuda',dtype=torch.int32)
    lids=torch.empty(bs*8,device='cuda',dtype=torch.int32);woff=torch.empty(128,device='cuda',dtype=torch.int32)
    y=torch.empty_like(x)
    def run():
        nonlocal y
        if name=='fused':mod.wm_norm_route(ids,nols,0,lmax,cnt,lst,act,nact,lids,woff,grp,x,w,y)
        else:
            y=F.rms_norm(x,(4096,),w,1.e-5) if name=='torch' else norm._pre_rms(x,w,int(name[-1]))
            mod.wm_route(ids,lmax,cnt,lst,act,nact,lids,woff,grp)
        return y
    run();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=run()
    g.replay();torch.cuda.synchronize();return g,out,(cnt,lst,lids)
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
def metadata_same(a,b,bs):
    ca,la,ia=a;cb,lb,ib=b
    assert torch.equal(ca,cb) and torch.equal(ia,ib)
    counts=ca.tolist();lmax=(bs+7)//8*8
    for e,n in enumerate(counts):assert torch.equal(la[e*lmax:e*lmax+n].sort().values,lb[e*lmax:e*lmax+n].sort().values)
for seed in (173,7919):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    w=(torch.randn(4096,generator=gen,device='cuda')*.1).bfloat16()+1
    for bs in (1,4,8,16,32,128,256):
        x=torch.randn(bs,4096,generator=gen,device='cuda').bfloat16()
        ids=torch.stack([torch.randperm(256,generator=gen,device='cuda')[:8] for _ in range(bs)]).int()
        gb,ob,mb=capture(x,w,ids,'torch')
        ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1.e-5)*w.float()).bfloat16()
        for name in ('triton4','triton8')+(('fused',) if bs<=8 else ()):
            g,o,meta=capture(x,w,ids,name);metadata_same(mb,meta,bs)
            err=((o.float()-ob.float()).norm()/ob.float().norm()).item()
            trueerr=((o.float()-ref.float()).norm()/ref.float().norm()).item()
            assert err<.004 and trueerr<.004,(name,bs,err,trueerr)
            print('CHECK',name,bs,seed,'PASS','torch_error',err,'fp32_error',trueerr,flush=True)
            bt,ct,ratios=[],[],[]
            for rep in range(24):
                if rep%2:c,b=elapsed(g),elapsed(gb)
                else:b,c=elapsed(gb),elapsed(g)
                bt.append(b);ct.append(c);ratios.append(b/c)
            print('RESULT',json.dumps(dict(variant=name,batch=bs,seed=seed,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
            del g,o,meta
        del gb,ob,mb,x,ids
print('DECODER_NORM_SCREEN_PASSED',flush=True)
