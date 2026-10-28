import importlib.util,torch,json
from torch.utils.cpp_extension import load_inline
sp=importlib.util.spec_from_file_location('probe','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
mod=load_inline('muse_fp8p_native_probe_v2',m.CPP,cuda_sources=m.CUDA,functions=['pv8'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
for mode in ('identity','ones','integer','permutation'):
 torch.manual_seed(113)
 v=torch.randint(-4,5,(3,64,64),device='cuda').float().to(torch.float8_e4m3fn)
 if mode=='identity':p=torch.eye(64,device='cuda').expand(3,-1,-1).contiguous()
 elif mode=='ones':p=torch.ones((3,64,64),device='cuda')
 elif mode=='integer':p=torch.randint(0,5,(3,64,64),device='cuda').float()
 else:p=torch.eye(64,device='cuda')[:,torch.arange(64,device='cuda')*7%64].expand(3,-1,-1).contiguous()
 p=p.to(torch.float8_e4m3fn);o=mod.pv8(p,v);r=torch.bmm(p.float(),v.float())
 diff=(o-r).abs().max().item();print('EXACT',mode,diff,flush=True);assert diff==0,(mode,diff)
scale=torch.ones((),device='cuda',dtype=torch.float32)
for batch in (1,3,16):
 for seed in (113,7919,3613):
  torch.manual_seed(seed);p=(torch.rand(batch,64,64,device='cuda')*256).to(torch.float8_e4m3fn);v=torch.randn(batch,64,64,device='cuda').to(torch.float8_e4m3fn)
  o=mod.pv8(p,v);r=torch.bmm(p.float(),v.float());err=((o-r).norm()/r.norm()).item()
  fast=[]
  for i in range(batch):
   y=torch._scaled_mm(p[i],v[i].T.contiguous().T,scale_a=scale,scale_b=scale,out_dtype=torch.float32,use_fast_accum=True)
   fast.append(y[0] if isinstance(y,tuple) else y)
  fast=torch.stack(fast);tcerr=((o-fast).norm()/fast.norm()).item();ref_tc_err=((fast-r).norm()/r.norm()).item()
  print('RESULT',json.dumps(dict(batch=batch,seed=seed,fp32_error=err,pytorch_fp8_error=ref_tc_err,error_vs_fp8_tensorcore=tcerr,max_abs=(o-r).abs().max().item())),flush=True)
  assert torch.isfinite(o).all().item() and err<.001 and tcerr<.0002,(err,tcerr)
print('FP8P_NATIVE_PRIMITIVE_PASSED',flush=True)
