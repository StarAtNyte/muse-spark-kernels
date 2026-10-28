"""DSM correctness/latency probe; not a claim about fused attention speed."""
import importlib.util,torch,triton,triton.language as tl,json,statistics,sys
sp=importlib.util.spec_from_file_location('probe','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
sys.path.insert(0,'/w/modal');from compile_split import build
mod=build('muse_dsm_merge_probe',m.CPP,m.CUDA,['cluster_merge'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
@triton.jit
def merge(P,M,S,G,O):
 group,row=tl.program_id(0),tl.program_id(1);sp=tl.arange(0,8);d=tl.arange(0,64)
 mx=tl.load(M+(group*8+sp)*64+row);high=tl.max(mx,0)
 w=tl.where(mx==-float('inf'),0.,tl.exp2(mx-high))
 den=tl.sum(tl.load(S+(group*8+sp)*64+row)*w,0)
 num=tl.sum(tl.load(P+((group*8+sp[:,None])*64+row)*64+d[None,:])*w[:,None],0)
 inv=tl.rsqrt(tl.sum(num*num,0)/64.+tl.where(den>0,1.e-5*den*den,1.))
 gate=tl.load(G+(group*64+row)*64+d).to(tl.float32)
 tl.store(O+(group*64+row)*64+d,num*inv*tl.sigmoid(gate))
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(fn):
 fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):o=fn()
 g.replay();torch.cuda.synchronize();return g,o
def elapsed(g):
 flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for groups in (2,8,32):
 for seed in (113,7919):
  torch.manual_seed(seed);mx=torch.randn(groups,8,64,device='cuda')*3
  sums=torch.rand(groups,8,64,device='cuda')*256+1
  part=torch.randn(groups,8,64,64,device='cuda')*sums[...,None]*(1.e-6 if seed==7919 else 1.)
  gate=torch.randn(groups,64,64,device='cuda',dtype=torch.bfloat16)
  mx[:,1]=-torch.inf;sums[:,1]=0;part[:,1]=0
  mx[:,:,0]=-torch.inf;sums[:,:,0]=0;part[:,:,0]=0
  weights=torch.exp2(mx-torch.where(torch.isfinite(mx.amax(1)),mx.amax(1),0)[:,None,:]);weights=torch.where(torch.isfinite(mx),weights,0)
  den=(sums*weights).sum(1);num=(part*weights[...,None]).sum(1)
  att=num/den.clamp_min(1.e-30)[...,None]
  ref=(att*torch.rsqrt(att.square().mean(-1,keepdim=True)+1.e-5)*torch.sigmoid(gate.float())).bfloat16()
  gc,oc=capture(lambda:mod.cluster_merge(part,mx,sums,gate))
  out=torch.empty_like(gate)
  def direct():merge[(groups,64)](part,mx,sums,gate,out,num_warps=4);return out
  gg,og=capture(direct)
  ec=((oc.float()-ref.float()).norm()/ref.float().norm()).item();eg=((og.float()-ref.float()).norm()/ref.float().norm()).item()
  print('CHECK',groups,seed,ec,eg,flush=True);assert torch.isfinite(oc).all().item() and ec<.003 and eg<.003,(ec,eg)
  ct,gt=[],[]
  for rep in range(24):
   if rep%2:ct.append(elapsed(gc));gt.append(elapsed(gg))
   else:gt.append(elapsed(gg));ct.append(elapsed(gc))
  print('RESULT',json.dumps(dict(groups=groups,seed=seed,cluster_us=statistics.median(ct),global_merge_us=statistics.median(gt),cluster_error=ec,global_error=eg)),flush=True)
  del gc,gg,oc,og,out,part,mx,sums,gate,ref
print('CLUSTER_MERGE_PROBE_PASSED',flush=True)
