"""Isolate current-stream correction on the failed Forge boundary candidate."""
import json, torch
from pathlib import Path
from torch.utils.cpp_extension import load
src = Path('/w/run/sub.py').read_text()
assert src.count('boundary_kernel<<<T, TPB>>>(') == 1
src = '#include <ATen/cuda/CUDAContext.h>\n' + src.replace('boundary_kernel<<<T, TPB>>>(', 'boundary_kernel<<<T, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(')
src = src.replace('    return {sout_t, nout_t};', '    C10_CUDA_KERNEL_LAUNCH_CHECK();\n    return {sout_t, nout_t};')
p = Path('/tmp/fixed_boundary.cu'); p.write_text(src)
m = load(name='forge_boundary_stream_fixed', sources=[str(p)], extra_cuda_cflags=['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a'], verbose=False)
torch.manual_seed(419)
for n in (128, 753, 3012):
  with torch.cuda.stream(torch.cuda.Stream()):
    b=torch.randn((n,8192), device='cuda', dtype=torch.bfloat16)
    st=torch.randn((n,8192), device='cuda')
    gp=torch.randn(8192, device='cuda', dtype=torch.bfloat16)
    a=torch.randn(8192, device='cuda'); be=torch.randn(8192, device='cuda')
    gn=torch.randn(8192, device='cuda', dtype=torch.bfloat16)
    y,z=m.run(b,st,gp,a,be,gn)
    x=b.float()*(1+gp.float()); ref=a*st+be*x*torch.rsqrt((x*x).mean(dim=1,keepdim=True)+1e-8)
    rz=(ref*torch.rsqrt((ref*ref).mean(dim=1,keepdim=True)+1e-5)*(1+gn.float())).bfloat16()
    err=((y-ref).abs().max()/ref.abs().max()).item(); ze=((z.float()-rz.float()).abs().max()/rz.float().abs().max()).item()
    print('ROW',json.dumps({'n':n,'sout_normalized_max_error':err,'nout_normalized_max_error':ze,'finite':bool(torch.isfinite(y).all() and torch.isfinite(z).all())}),flush=True)
