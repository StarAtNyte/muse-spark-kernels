"""Check exact two-head sparse MMA packing against dense products."""
import importlib.util,json
import torch
from torch.utils.cpp_extension import load_inline
spec=importlib.util.spec_from_file_location('probe','/w/run/sub.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
mod=load_inline('muse_sparse_qk',probe.CPP,cuda_sources=probe.CUDA,functions=['sparse_qk'],
    extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
for batch,keys in ((1,8),(1,128),(3,16),(16,256)):
    for seed in (113,7919):
        torch.manual_seed(seed)
        q=torch.randn(batch,16,64,device='cuda').to(torch.float8_e4m3fn)
        k=torch.randn(batch,keys,2,64,device='cuda').to(torch.float8_e4m3fn)
        out=mod.sparse_qk(q,k)
        ref=torch.cat([torch.bmm(q[:,:8].float(),k[:,:,0].float().transpose(1,2)),
                       torch.bmm(q[:,8:].float(),k[:,:,1].float().transpose(1,2))],1)
        err=((out-ref).norm()/ref.norm()).item()
        diff=(out-ref).abs().max().item()
        print('RESULT',json.dumps(dict(batch=batch,keys=keys,seed=seed,relative_error=err,max_abs_error=diff)),flush=True)
        if not torch.isfinite(out).all().item() or err>1e-5: raise AssertionError((err,diff))
print('SPARSE_QK_PASSED',flush=True)
