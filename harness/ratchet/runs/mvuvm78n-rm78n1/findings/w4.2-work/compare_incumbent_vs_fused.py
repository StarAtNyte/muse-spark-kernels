"""Benchmark comparing Incumbent Triton Route Kernel vs Fused Cooperative Router + Dispatch Descriptor Builder
"""

import sys
import os
import json
import torch
import triton
import triton.language as tl

ROUTER_MULT = tl.constexpr(0.1767766952966369)

# Incumbent Triton split routing kernel (as extracted from kernels/muse_decode_layer/submission.py)
@triton.jit
def _incumbent_route_split(
    POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS,
    PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False
):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)
    lsv = tl.load(LS)
    
    logit = tl.load(LOGITS + t * 256 + e)
    sc = tl.sigmoid(logit * ROUTER_MULT)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)


# Incumbent prologue routine (simulating the MoE CTA entry prologue that parses IDS from global memory)
@triton.jit
def _incumbent_moe_prologue(
    IDS_PTR, LS_PTR, CNT_OUT_PTR, OFF_OUT_PTR, META_OUT_PTR,
    T: tl.constexpr, LOCAL_EXPERTS: tl.constexpr = 32, TOP_K: tl.constexpr = 8,
    SK_T: tl.constexpr = 8, FT_T: tl.constexpr = 64
):
    pid = tl.program_id(0)
    ls = tl.load(LS_PTR)
    le = tl.arange(0, LOCAL_EXPERTS)
    cnts = tl.zeros([LOCAL_EXPERTS], dtype=tl.int32)
    for s in range(T * TOP_K):
        raw_id = tl.load(IDS_PTR + s)
        rel_id = (raw_id - ls + 256) % 256
        is_match = (rel_id == le) & (rel_id < LOCAL_EXPERTS)
        cnts += tl.where(is_match, 1, 0)
        
    tl.store(CNT_OUT_PTR + le, cnts)
    inclusive = tl.cumsum(cnts, 0)
    tl.store(OFF_OUT_PTR + le, inclusive - cnts)
    
    total = tl.sum(cnts, 0)
    is_fat = cnts > SK_T
    is_skinny = (cnts > 0) & (cnts <= SK_T)
    nfj = tl.sum(tl.where(is_fat, (cnts + FT_T - 1) // FT_T, 0), 0)
    nsj = tl.sum(tl.where(is_skinny, 1, 0), 0)
    
    tl.store(META_OUT_PTR + 0, total)
    tl.store(META_OUT_PTR + 1, nfj)
    tl.store(META_OUT_PTR + 2, nsj)


sys.path.append(os.path.dirname(__file__))
from fused_router_triton import run_fused_moe_router

def benchmark_pair(T, reps=200, warmup=50):
    dev = "cuda"
    logits = torch.randn(T, 256, device=dev, dtype=torch.float32)
    bias = torch.randn(256, device=dev, dtype=torch.float32)
    ls = torch.tensor(32, dtype=torch.int32, device=dev)
    
    # Incumbent buffers
    ids = torch.empty((T, 8), dtype=torch.int32, device=dev)
    wts = torch.empty((T, 8), dtype=torch.float32, device=dev)
    lids = torch.empty((T, 8), dtype=torch.int32, device=dev)
    cnt = torch.zeros(32, dtype=torch.int32, device=dev)
    off = torch.zeros(32, dtype=torch.int32, device=dev)
    meta = torch.zeros(4, dtype=torch.int32, device=dev)
    dummy_p = torch.zeros(1, device=dev)
    
    def run_incumbent():
        _incumbent_route_split[(T,)](
            dummy_p, logits, bias, dummy_p, ids, wts, dummy_p, ls, lids,
            PERMUTED=False, USE_PDL=False, num_warps=4
        )
        _incumbent_moe_prologue[(1,)](
            ids, ls, cnt, off, meta, T=T, num_warps=2
        )

    def run_fused():
        run_fused_moe_router(logits, bias, ls, write_diagnostics=False)
        
    # Warmup
    for _ in range(warmup):
        run_incumbent()
        run_fused()
    torch.cuda.synchronize()
    
    # Time Incumbent
    s_inc = torch.cuda.Event(enable_timing=True)
    e_inc = torch.cuda.Event(enable_timing=True)
    s_inc.record()
    for _ in range(reps):
        run_incumbent()
    e_inc.record()
    
    # Time Fused
    s_fused = torch.cuda.Event(enable_timing=True)
    e_fused = torch.cuda.Event(enable_timing=True)
    s_fused.record()
    for _ in range(reps):
        run_fused()
    e_fused.record()
    
    torch.cuda.synchronize()
    
    t_inc_us = (s_inc.elapsed_time(e_inc) / reps) * 1000.0
    t_fused_us = (s_fused.elapsed_time(e_fused) / reps) * 1000.0
    
    return t_inc_us, t_fused_us

if __name__ == "__main__":
    shapes = [1, 8, 16, 32, 128, 256]
    summary = {}
    print(f"{'T':<6} | {'Incumbent (us)':<16} | {'Fused Zero-GMEM (us)':<22} | {'Speedup':<10}")
    print("-" * 62)
    for T in shapes:
        t_inc, t_fused = benchmark_pair(T)
        spd = t_inc / t_fused
        print(f"{T:<6} | {t_inc:<16.2f} | {t_fused:<22.2f} | {spd:<10.2f}x")
        summary[T] = {
            "incumbent_us": round(t_inc, 2),
            "fused_zero_gmem_us": round(t_fused, 2),
            "speedup": round(spd, 2)
        }
    out_file = os.path.join(os.path.dirname(__file__), "incumbent_comparison.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved comparison to {out_file}")
