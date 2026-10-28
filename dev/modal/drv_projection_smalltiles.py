"""Exact BF16 small-M tensor-core GEMMs, paired cold-cache graph screen."""
exec(open('/w/modal/drv_router.py').read().split("print('GPU'")[0])
import triton
import triton.language as tl
@triton.jit
def mm(X,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,T:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);sp=tl.program_id(2)
 kk=sp*(K//S)+tl.arange(0,BK);acc=tl.full((BM,BN),0,tl.float32)
 for i in range(K//S//BK):
  k=kk+i*BK;a=tl.load(X+m[:,None]*K+k[None,:],m[:,None]<M,0)
  if T:b=tl.load(W+k[:,None]*N+n[None,:],n[None,:]<N,0)
  else:b=tl.load(W+n[None,:]*K+k[:,None],n[None,:]<N,0)
  acc=tl.dot(a,b,acc)
 tl.store(P+sp*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))
for layer in ('qkv_gate','o'):
 w=weights[layer];wt=w.T.contiguous()
 configs=((16,32,64,4),(16,64,64,4),(16,64,64,8),(16,64,64,16),(16,128,64,8),(16,64,128,8)) if layer=='qkv_gate' else ((16,32,64,1),(16,64,64,1),(32,32,64,1),(16,32,128,1),(32,64,64,1),(16,64,128,1))
 for batch in (16,32):
  data=R.generate_input(batch,layer,7919);x=data['x'];n=data['config']['n'];k=x.shape[1]
  gb,_=capture(lambda:sub._kernel_v5(data,state))
  for trans in (False,True):
   for bm,bn,bk,split in configs:
    out=torch.empty((batch,n),dtype=torch.bfloat16,device='cuda');part=out if split==1 else torch.empty((split,batch,n),dtype=torch.float32,device='cuda')
    def run():
     mm[(triton.cdiv(batch,bm),triton.cdiv(n,bn),split)](x,wt if trans else w,part,batch,n,k,bm,bn,bk,split,trans,num_warps=4,num_stages=3)
     if split>1:sub._finish[(triton.cdiv(batch*n,512),)](part,out,batch*n,split,512)
     return out
    gc,oc=capture(run);ok,msg=R.check_implementation(data,oc,weights);assert ok,(layer,batch,trans,bm,bn,bk,split,msg)
    ratios,bt,ct=[],[],[]
    for rep in range(16):
     if rep%2:c,b=elapsed(gc),elapsed(gb)
     else:b,c=elapsed(gb),elapsed(gc)
     ratios.append(b/c);bt.append(b);ct.append(c)
    print('RESULT',json.dumps(dict(layer=layer,batch=batch,transposed=trans,bm=bm,bn=bn,bk=bk,split=split,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
    del gc,oc,part,out
  del gb,data
print('PROJECTION_SMALL_TILES_PASSED',flush=True)
