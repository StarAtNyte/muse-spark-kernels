import importlib.util,json,torch
from torch.utils.cpp_extension import load_inline
spec=importlib.util.spec_from_file_location('probe','/w/run/sub.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
mod=load_inline('transposed_head_probe',m.CPP,cuda_sources=m.CUDA,functions=['trans_probe'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
for batch,keys in ((1,16),(1,128),(3,16),(16,256)):
 for seed in (113,7919):
  torch.manual_seed(seed)
  q=torch.randn(batch,16,64,device='cuda').to(torch.float8_e4m3fn)
  k=torch.randn(batch,keys,2,64,device='cuda').to(torch.float8_e4m3fn)
  v=torch.randn_like(k.float()).to(torch.float8_e4m3fn)
  p=torch.randn(batch,16,keys,device='cuda',dtype=torch.float16)
  s,o=mod.trans_probe(q,k,p,v)
  sr=torch.cat([torch.bmm(q[:,:8].float(),k[:,:,0].float().transpose(1,2)),torch.bmm(q[:,8:].float(),k[:,:,1].float().transpose(1,2))],1)
  vr=torch.cat([torch.bmm(p[:,:8].float(),v[:,:,0].float()),torch.bmm(p[:,8:].float(),v[:,:,1].float())],1)
  errors=[((a-r).norm()/r.norm()).item() for a,r in ((s,sr),(o,vr))]
  print('RESULT',json.dumps(dict(batch=batch,keys=keys,seed=seed,errors=errors)),flush=True)
  assert all(e<2e-5 for e in errors),errors
print('TRANSPOSE_PRIMITIVES_PASSED',flush=True)
