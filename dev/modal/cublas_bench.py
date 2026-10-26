import torch
flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
def timeit(fn, reps=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(4): fn()
    ts = []
    for _ in range(reps):
        flush.add_(1); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000 / 4)
    ts.sort(); return ts[len(ts) // 2]
L = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
for name, (n, k) in L.items():
    w = torch.randn(n, k, device='cuda').to(torch.bfloat16)
    row = []
    for m in (1, 8, 16, 32, 128, 256):
        x = torch.randn(m, k, device='cuda').to(torch.bfloat16)
        t = timeit(lambda: torch.nn.functional.linear(x, w))
        by = n * k * 2
        row.append("m%d=%.1fus(%.2fTB/s,%.0fTF)" % (m, t, by / t / 1e6, 2 * m * n * k / t / 1e6))
    print("CUBLAS", name, n, k, " ".join(row), flush=True)
