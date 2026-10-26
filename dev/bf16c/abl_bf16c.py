import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf16c_cuda as B
from torch.utils.cpp_extension import load_inline
flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
def timeit(fn, reps=15):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.sum(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[len(ts) // 2]
N, K, m = 4096, 8192, 1
w = (0.02 * torch.randn(N, K, device="cuda")).to(torch.bfloat16)
wc, esc_off, esc, base, st = B.pack(w)
x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
out = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
for name, flags in (("base", []), ("nox", ["-DNOX=1"]), ("now", ["-DNOW=1"]), ("nodec", ["-DNODEC=1"]), ("nox_nodec", ["-DNOX=1", "-DNODEC=1"]), ("now_nodec_nox", ["-DNOW=1", "-DNODEC=1", "-DNOX=1"])):
    mod = load_inline("bf16c_abl_" + name, B.CPP_SRC, cuda_sources=B.CUDA_SRC + B.TORCH_GLUE, functions=["bf16c"], extra_cuda_cflags=["-O3"] + flags, verbose=False)
    res = []
    for cfg in ((1, 16, 4), (1, 8, 4), (1, 32, 4)):
        res.append("%s=%.1f" % ("-".join(map(str, cfg)), timeit(lambda: mod.bf16c(wc, x, out, esc_off, esc, N, K, base, *cfg))))
    print("ABL", name, " ".join(res), flush=True)
