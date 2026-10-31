"""Check whether loading an sm_100a-only extension poisons CUDA on an H200."""
import os, subprocess, sys
if len(sys.argv) == 1:
    for arch in ('90a', '100a'):
        p = subprocess.run([sys.executable, __file__, arch], capture_output=True, text=True, timeout=120)
        print('ARCH', arch, 'RETURN', p.returncode, '\n', p.stdout, '\n', p.stderr[-3000:], flush=True)
    raise SystemExit(0)
import torch
from torch.utils.cpp_extension import load_inline
arch = sys.argv[1]
print('DEVICE', torch.cuda.get_device_name(), 'CAP', torch.cuda.get_device_capability(), 'TORCH', torch.__version__, flush=True)
print('PRE', torch.randn(32, device='cuda').sum().item(), flush=True)
cu = '''#include <torch/extension.h>
__global__ void k(float* p) { p[threadIdx.x] += 1; }
void run(torch::Tensor t) { k<<<1,32>>>(t.data_ptr<float>()); auto e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e)); }
'''
m = load_inline('arch_probe_' + arch, cpp_sources='void run(torch::Tensor t);', cuda_sources=cu, functions=['run'], extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}'], verbose=False)
print('LOADED', m.__file__, flush=True)
print('ELF', subprocess.run(['cuobjdump', '--list-elf', m.__file__], capture_output=True, text=True).stdout, flush=True)
t = torch.randn(32, device='cuda'); torch.cuda.synchronize(); print('POST', t.sum().item(), flush=True)
try:
    m.run(t); torch.cuda.synchronize(); print('KERNEL PASS', flush=True)
except Exception as e:
    print('KERNEL FAIL', repr(e), flush=True)
