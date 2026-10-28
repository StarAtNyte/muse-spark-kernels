import importlib.util,torch,json
from torch.utils.cpp_extension import load_inline
sp=importlib.util.spec_from_file_location('probe','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
mod=load_inline('muse_fp8p_native_probe',m.CPP,cuda_sources=m.CUDA,functions=['pv8'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
for batch in (1,3,16):
 for seed in (113,7919,3613):
  torch.manual_seed(seed);p=(torch.rand(batch,64,64,device='cuda')*256).to(torch.float8_e4m3fn);v=torch.randn(batch,64,64,device='cuda').to(torch.float8_e4m3fn)
  o=mod.pv8(p,v);r=torch.bmm(p.float(),v.float());err=((o-r).norm()/r.norm()).item()
  print('RESULT',json.dumps(dict(batch=batch,seed=seed,error=err,max_abs=(o-r).abs().max().item())),flush=True);assert torch.isfinite(o).all().item() and err<2e-5,err
print('FP8P_NATIVE_PRIMITIVE_PASSED',flush=True)
