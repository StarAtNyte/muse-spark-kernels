import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf16c_cuda as B
from torch.utils.cpp_extension import load_inline
t0 = time.time()
mod = load_inline("bf16c_t", B.CPP_SRC, cuda_sources=B.CUDA_SRC + B.TORCH_GLUE, functions=["bf16c"], extra_cuda_cflags=["-O3"], verbose=False)
print("compiled", time.time() - t0)
torch.manual_seed(0)
for (N, K) in ((2304, 8192), (8192, 1024), (4096, 8192)):
    w = (0.02 * torch.randn(N, K, device="cuda")).to(torch.bfloat16)
    wc, esc_off, esc, base, st = B.pack(w)
    print("pack", N, K, "base", base, st)
    for m in (1, 8, 16, 32):
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        ref = x.float() @ w.float().T
        for nt, nw, s in ((1, 8, 4), (2, 4, 4)):
            out = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
            try:
                mod.bf16c(wc, x, out, esc_off, esc, N, K, base, nt, nw, s)
            except Exception as ex:
                print("ERR", N, K, m, nt, nw, s, str(ex)[:80]); continue
            torch.cuda.synchronize()
            err = ((out.float() - ref).norm() / ref.norm()).item()
            print(f"N={N} K={K} m={m} cfg=({nt},{nw},{s}) relerr={err:.2e}")
