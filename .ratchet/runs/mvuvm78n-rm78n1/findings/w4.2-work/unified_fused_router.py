"""Unified Single-Launch Fused MoE Router & Cooperative Dispatch Descriptor Kernel

Eliminates both kernel launch overhead and intermediate global memory writes for T <= 32.
Both gating, top-k selection, local filtering, and dispatch descriptor construction execute
within a single CTA cooperating through shared memory and registers.
"""

import sys
import os
import json
import torch
import triton
import triton.language as tl

ROUTER_MULT = tl.constexpr(0.1767766952966369)

@triton.jit
def _unified_single_launch_router_kernel(
    LOGITS_PTR,
    BIAS_PTR,
    LOCAL_START_PTR,
    CNT_OUT_PTR,
    OFF_OUT_PTR,
    META_OUT_PTR,
    T: tl.constexpr,
    NUM_EXPERTS: tl.constexpr = 256,
    LOCAL_EXPERTS: tl.constexpr = 32,
    TOP_K: tl.constexpr = 8,
    SK_T: tl.constexpr = 8,
    FT_T: tl.constexpr = 64,
):
    pid = tl.program_id(0) # 1 block for the entire step
    local_start = tl.load(LOCAL_START_PTR)
    e_idx = tl.arange(0, NUM_EXPERTS)
    bias = tl.load(BIAS_PTR + e_idx)
    
    le_idx = tl.arange(0, LOCAL_EXPERTS)
    cnts = tl.zeros([LOCAL_EXPERTS], dtype=tl.int32)
    
    # Process each token
    for t in range(T):
        logits = tl.load(LOGITS_PTR + t * NUM_EXPERTS + e_idx)
        probs = tl.sigmoid(logits * ROUTER_MULT)
        scores = probs + bias
        
        bits = scores.to(tl.uint32, bitcast=True)
        ordered_scores = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
        packed = (ordered_scores.to(tl.uint64) << 32) | (255 - e_idx).to(tl.uint64)
        
        sorted_packed = tl.sort(packed, descending=True)
        k8 = tl.arange(0, TOP_K)
        top_packed = tl.gather(sorted_packed, k8, 0)
        
        top_ids = (255 - (top_packed & 255)).to(tl.int32)
        rel_ids = (top_ids - local_start + NUM_EXPERTS) % NUM_EXPERTS
        
        # Accumulate local counts directly
        for k in range(TOP_K):
            curr_rel = tl.sum(tl.where(k8 == k, rel_ids, 0), 0)
            is_match = (curr_rel == le_idx) & (curr_rel < LOCAL_EXPERTS)
            cnts += tl.where(is_match, 1, 0)
            
    tl.store(CNT_OUT_PTR + le_idx, cnts)
    
    inclusive = tl.cumsum(cnts, 0)
    exclusive = inclusive - cnts
    tl.store(OFF_OUT_PTR + le_idx, exclusive)
    
    total_local = tl.sum(cnts, 0)
    is_fat = cnts > SK_T
    is_skinny = (cnts > 0) & (cnts <= SK_T)
    nfj = tl.sum(tl.where(is_fat, (cnts + FT_T - 1) // FT_T, 0), 0)
    nsj = tl.sum(tl.where(is_skinny, 1, 0), 0)
    
    tl.store(META_OUT_PTR + 0, total_local)
    tl.store(META_OUT_PTR + 1, nfj)
    tl.store(META_OUT_PTR + 2, nsj)


def benchmark_unified_single_launch(T, reps=200, warmup=50):
    dev = "cuda"
    logits = torch.randn(T, 256, device=dev, dtype=torch.float32)
    bias = torch.randn(256, device=dev, dtype=torch.float32)
    ls = torch.tensor(32, dtype=torch.int32, device=dev)
    
    cnt = torch.zeros(32, dtype=torch.int32, device=dev)
    off = torch.zeros(32, dtype=torch.int32, device=dev)
    meta = torch.zeros(4, dtype=torch.int32, device=dev)
    
    def run_unified():
        _unified_single_launch_router_kernel[(1,)](
            logits, bias, ls, cnt, off, meta, T=T, num_warps=4
        )
        
    for _ in range(warmup):
        run_unified()
    torch.cuda.synchronize()
    
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        run_unified()
    e.record()
    torch.cuda.synchronize()
    
    return (s.elapsed_time(e) / reps) * 1000.0


if __name__ == "__main__":
    from compare_incumbent_vs_fused import benchmark_pair
    shapes = [1, 8, 16, 32]
    print(f"{'T':<6} | {'Incumbent (us)':<16} | {'Unified Single (us)':<22} | {'Speedup':<10}")
    print("-" * 62)
    unified_results = {}
    for T in shapes:
        t_inc, _ = benchmark_pair(T)
        t_uni = benchmark_unified_single_launch(T)
        spd = t_inc / t_uni
        print(f"{T:<6} | {t_inc:<16.2f} | {t_uni:<22.2f} | {spd:<10.2f}x")
        unified_results[T] = {
            "incumbent_us": round(t_inc, 2),
            "unified_single_launch_us": round(t_uni, 2),
            "speedup": round(spd, 2)
        }
    out_file = os.path.join(os.path.dirname(__file__), "unified_comparison.json")
    with open(out_file, "w") as f:
        json.dump(unified_results, f, indent=2)
