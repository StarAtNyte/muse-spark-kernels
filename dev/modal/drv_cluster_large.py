"""Large-batch screen against real FA3 plus a local Triton norm/gate proxy.

This is a screening comparison, not an exact reproduction of official SGLang
norm/gate timing. Full task correctness is checked independently.
"""
import sgl_kernel.flash_attn
exec(open('/w/modal/drv_cluster_attention.py').read().split('\nfor batch,ctx,page,scale,dist in ')[0])
import triton.language as tl
@triton.jit
def norm_gate(X,G,O,GS:tl.constexpr,H:tl.constexpr):
 row=tl.program_id(0);d=tl.arange(0,64)
 x=tl.load(X+row*64+d).to(tl.float32)
 g=tl.load(G+(row//H)*GS+(row%H)*64+d).to(tl.float32)
 y=x*tl.rsqrt(tl.sum(x*x,0)/64.+1.e-5)*tl.sigmoid(g)
 tl.store(O+row*64+d,y)
def local_norm(x,gate,weight,**kw):
 out=torch.empty_like(x);norm_gate[(x.numel()//64,)](x,gate,out,gate.stride(0),x.shape[1]//64,num_warps=4);return out
base.rms_norm_gate=local_norm
for batch in (16,32):
 for seed in (173,2387):
  data=R.generate_input(batch,80000,'tp8','sliding',seed,q_len=8,page=1,dist='coherent',spread='agentic')
  gc,oc=capture(candidate,data)
  ok,msg=R.check_implementation(data,R.collect(data,oc));print('CHECK',batch,seed,ok,msg,flush=True);assert ok,msg
  gb,ob=capture(base,data)
  ok,msg=R.check_implementation(data,R.collect(data,ob));print('BASE_CHECK',batch,seed,ok,msg,flush=True);assert ok,msg
  bt,ct,ratios=[],[],[]
  for rep in range(24):
   if rep%2:c,b=elapsed(gc),elapsed(gb)
   else:b,c=elapsed(gb),elapsed(gc)
   bt.append(b);ct.append(c);ratios.append(b/c)
  print('RESULT',json.dumps(dict(batch=batch,seed=seed,fa3_proxy_us=statistics.median(bt),cluster_us=statistics.median(ct),speedup=statistics.median(ratios))),flush=True)
  del data,gc,gb,oc,ob
print('CLUSTER_LARGE_SCREEN_PASSED',flush=True)
