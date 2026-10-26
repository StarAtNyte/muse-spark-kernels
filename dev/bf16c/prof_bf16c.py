import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf16c_cuda as B
from torch.utils.cpp_extension import load_inline
mod = load_inline("bf16c_t", B.CPP_SRC, cuda_sources=B.CUDA_SRC + B.TORCH_GLUE, functions=["bf16c"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
N, K = int(os.environ.get("N", 4096)), int(os.environ.get("K", 8192)); m = int(os.environ.get("M", 1))
nt, nw, s = [int(v) for v in os.environ.get("CFG", "1,16,4").split(",")]
w = (0.02 * torch.randn(N, K, device="cuda")).to(torch.bfloat16)
wc, esc_off, esc, base, st = B.pack(w)
x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
out = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
for _ in range(3):
    flush.add_(1); torch.cuda.synchronize()
    mod.bf16c(wc, x, out, esc_off, esc, N, K, base, nt, nw, s)
torch.cuda.synchronize()
