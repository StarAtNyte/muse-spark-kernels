import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf16c_cuda as B
from torch.utils.cpp_extension import load_inline
mod = load_inline("bf16c_t", B.CPP_SRC, cuda_sources=B.CUDA_SRC + B.TORCH_GLUE, functions=["bf16c"], extra_cuda_cflags=["-O3"], verbose=False)
flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
def timeit(fn, reps=15):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[len(ts) // 2]
CFGS = [(1,8,4),(1,8,8),(1,16,4),(1,16,6),(1,16,8),(1,32,3),(1,32,4),(1,32,5),(2,8,4),(2,16,4),(2,16,6)]
L = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "unpooler_slice": (1024, 4096)}
torch.manual_seed(0)
for name, (N, K) in L.items():
    w = (0.02 * torch.randn(N, K, device="cuda")).to(torch.bfloat16)
    wc, esc_off, esc, base, st = B.pack(w)
    for m in (1, 16, 32):
        if name == "unpooler_slice" and m > 8: continue
        if name == "unpooler" and m <= 8: continue
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        cub = timeit(lambda: torch.nn.functional.linear(x, w))
        out = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ w.float().T
        res = []
        for nt, nw, s in CFGS:
            if (K // 64) % nw != 0 or N % (16 * nt) != 0: continue
            try:
                mod.bf16c(wc, x, out, esc_off, esc, N, K, base, nt, nw, s)
                torch.cuda.synchronize()
                err = ((out.float() - ref).norm() / ref.norm()).item()
                if err > 3e-3: continue
                res.append((timeit(lambda: mod.bf16c(wc, x, out, esc_off, esc, N, K, base, nt, nw, s)), (nt, nw, s)))
            except Exception as ex:
                pass
        res.sort()
        print("BENCH %-15s m=%-3d cublas=%.1f  " % (name, m, cub) + " ".join("%.1f:%s" % (t, "-".join(map(str, c))) for t, c in res[:4]), flush=True)
