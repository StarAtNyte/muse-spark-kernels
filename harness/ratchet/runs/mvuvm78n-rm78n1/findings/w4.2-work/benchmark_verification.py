"""Verification and Benchmark Suite for Fused MoE Router & Cooperative Top-K Selection

Validates correctness against PyTorch reference and measures latency across all
challenge batch shapes: T in {1, 8, 16, 32, 128, 256}.
"""

import sys
import os
import json
import time
import torch
import triton
import triton.language as tl

sys.path.append(os.path.dirname(__file__))
from fused_router_triton import run_fused_moe_router

ROUTER_MULT = 0.1767766952966369  # 1 / sqrt(32)

def pytorch_reference_router(logits, bias, local_start, num_experts=256, local_experts=32, top_k=8):
    """Exact PyTorch mathematical reference as specified in TASK.md."""
    T = logits.shape[0]
    probs = torch.sigmoid(logits * ROUTER_MULT)
    scores = probs + bias
    
    # Descending topk
    val, ids = scores.topk(top_k, dim=-1)
    
    # Gather unbiased probabilities for weight normalization
    top_probs = torch.gather(probs, -1, ids)
    weights = top_probs / (top_probs.sum(-1, keepdim=True) + 1e-15)
    
    # Local expert IDs
    ls = local_start.item() if isinstance(local_start, torch.Tensor) else local_start
    rel_ids = (ids - ls + num_experts) % num_experts
    is_local = rel_ids < local_experts
    local_ids = torch.where(is_local, rel_ids, -1)
    
    # Histogram of local routes
    loc_mask = local_ids >= 0
    cnt = torch.zeros(local_experts, dtype=torch.int32, device=logits.device)
    for e in range(local_experts):
        cnt[e] = (local_ids == e).sum()
        
    off = torch.zeros(local_experts, dtype=torch.int32, device=logits.device)
    curr = 0
    for e in range(local_experts):
        off[e] = curr
        curr += cnt[e].item()
        
    total_local = curr
    
    # Jobs calculation
    SK_T = 8
    FT_T = 64
    njF = 0
    njS = 0
    for e in range(local_experts):
        c = cnt[e].item()
        if c > SK_T:
            njF += (c + FT_T - 1) // FT_T
        elif c > 0:
            njS += 1
            
    return {
        "topk_ids": ids.to(torch.int32),
        "topk_weights": weights.to(torch.float32),
        "local_ids": local_ids.to(torch.int32),
        "expert_counts": cnt,
        "expert_offsets": off,
        "total_local_routes": total_local,
        "fat_jobs": njF,
        "skinny_jobs": njS,
    }


def benchmark_kernel(fn, reps=100, warmup=25):
    """Accurate CUDA event timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(reps):
        fn()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    return (elapsed_ms / reps) * 1000.0  # in microseconds (us)


def run_all_tests():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA device required.")
        return
        
    print(f"Testing on GPU: {torch.cuda.get_device_name(0)}")
    shapes = [1, 8, 16, 32, 128, 256]
    results = {
        "device": torch.cuda.get_device_name(0),
        "shapes": {},
        "overall_status": "PASSED"
    }
    
    for T in shapes:
        print(f"\n--- Evaluating Shape T = {T} ---")
        logits = torch.randn(T, 256, device=device, dtype=torch.float32)
        # Add skewed distribution to mimic realistic MoE routing
        popularity = torch.randn(256, device=device).pow(2) * 2.0
        logits += popularity.unsqueeze(0)
        
        bias = torch.randn(256, device=device, dtype=torch.float32) * 0.1
        local_start = torch.tensor(32, dtype=torch.int32, device=device)
        
        # 1. Reference
        ref = pytorch_reference_router(logits, bias, local_start)
        
        # 2. Fused Kernel
        out = run_fused_moe_router(logits, bias, local_start, write_diagnostics=True)
        
        # 3. Verifications
        # Check IDs
        id_diff = (ref["topk_ids"] != out["topk_ids"]).sum().item()
        
        # Check Weights (relative L2 and max abs diff)
        w_max_diff = (ref["topk_weights"] - out["topk_weights"]).abs().max().item()
        w_rel_err = torch.norm(ref["topk_weights"] - out["topk_weights"]) / (torch.norm(ref["topk_weights"]) + 1e-12)
        w_rel_err = w_rel_err.item()
        
        # Check Local IDs
        lid_diff = (ref["local_ids"] != out["local_ids"]).sum().item()
        
        # Check Counts and Offsets
        cnt_diff = (ref["expert_counts"] != out["expert_counts"]).sum().item()
        off_diff = (ref["expert_offsets"] != out["expert_offsets"]).sum().item()
        
        total_routes_match = (ref["total_local_routes"] == out["total_local_routes"])
        jobs_match = (ref["fat_jobs"] == out["fat_jobs"]) and (ref["skinny_jobs"] == out["skinny_jobs"])
        
        print(f"T={T}: ID mismatches: {id_diff}/{T*8}, Weight max diff: {w_max_diff:.2e}, Rel L2: {w_rel_err:.2e}")
        print(f"T={T}: Counts match: {cnt_diff == 0}, Offsets match: {off_diff == 0}, Jobs match: {jobs_match}")
        
        passed = (id_diff == 0) and (w_max_diff < 1e-5) and (cnt_diff == 0) and (off_diff == 0) and jobs_match
        if not passed:
            results["overall_status"] = "FAILED"
            print(f"FAILED on T={T}")
            
        # 4. Latency Benchmark
        time_ref = benchmark_kernel(lambda: pytorch_reference_router(logits, bias, local_start))
        time_fused = benchmark_kernel(lambda: run_fused_moe_router(logits, bias, local_start, write_diagnostics=False))
        time_fused_diag = benchmark_kernel(lambda: run_fused_moe_router(logits, bias, local_start, write_diagnostics=True))
        
        speedup = time_ref / time_fused
        print(f"T={T} Latency: PyTorch = {time_ref:.2f} us | Fused = {time_fused:.2f} us (Speedup = {speedup:.2f}x) | Fused+Diag = {time_fused_diag:.2f} us")
        
        results["shapes"][T] = {
            "passed": passed,
            "id_mismatches": id_diff,
            "weight_max_diff": w_max_diff,
            "weight_rel_err": w_rel_err,
            "total_local_routes": out["total_local_routes"],
            "fat_jobs": out["fat_jobs"],
            "skinny_jobs": out["skinny_jobs"],
            "latency_pytorch_us": round(time_ref, 2),
            "latency_fused_zero_gmem_us": round(time_fused, 2),
            "latency_fused_diag_us": round(time_fused_diag, 2),
            "speedup_vs_pytorch": round(speedup, 2)
        }
        
    out_path = os.path.join(os.path.dirname(__file__), "verification_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured verification results to {out_path}")

if __name__ == "__main__":
    run_all_tests()
