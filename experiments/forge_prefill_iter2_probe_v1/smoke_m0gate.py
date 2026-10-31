"""Smoke probe: verify boundary_kernel launches without crash for all 3 scored shapes.
Non-numerical contract — no output comparison, just launch + sync.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from leaderboard import load_cuda_entry

import torch

KERNEL_PATH = os.path.join(os.path.dirname(__file__), "..", "kernel.cu")
run = load_cuda_entry(KERNEL_PATH, "run")

H = 8192
device = "cuda"

for T in [128, 753, 3012]:
    branch = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    stream = torch.randn(T, H, dtype=torch.float32, device=device)
    gpost  = torch.randn(H, dtype=torch.bfloat16, device=device)
    alpha  = torch.ones(H, dtype=torch.float32, device=device)
    beta   = torch.ones(H, dtype=torch.float32, device=device)
    gpre   = torch.randn(H, dtype=torch.bfloat16, device=device)

    sout, nout = run(branch, stream, gpost, alpha, beta, gpre)
    torch.cuda.synchronize()

    assert sout.shape == (T, H) and sout.dtype == torch.float32, f"T={T} sout shape/dtype wrong"
    assert nout.shape == (T, H) and nout.dtype == torch.bfloat16, f"T={T} nout shape/dtype wrong"

print("PROBE PASS max_diff=n/a")
