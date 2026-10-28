"""Check exact two-head FP16 P×V sparse MMA packing."""
import importlib.util,json
import torch
from torch.utils.cpp_extension import load_inline
spec=importlib.util.spec_from_file_location('probe','/w/run/sub.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
mod=load_inline('muse_sparse_pv',probe.CPP,cuda_sources=probe.CUDA,functions=['sparse_pv'],
    extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
for batch,keys in ((1,16),(1,128),(3,16),(16,256)):
    for seed in (113,7919):
        torch.manual_seed(seed)
        p=torch.randn(batch,16,keys,device='cuda',dtype=torch.float16)
        v=torch.randn(batch,keys,2,64,device='cuda',dtype=torch.float16)
        out=mod.sparse_pv(p,v)
        ref=torch.cat([torch.bmm(p[:,:8].float(),v[:,:,0].float()),
                       torch.bmm(p[:,8:].float(),v[:,:,1].float())],1)
        err=((out-ref).norm()/ref.norm()).item()
        diff=(out-ref).abs().max().item()
        print('RESULT',json.dumps(dict(batch=batch,keys=keys,seed=seed,relative_error=err,max_abs_error=diff)),flush=True)
        if not torch.isfinite(out).all().item() or err>2e-5: raise AssertionError((err,diff))
print('SPARSE_PV_PASSED',flush=True)
