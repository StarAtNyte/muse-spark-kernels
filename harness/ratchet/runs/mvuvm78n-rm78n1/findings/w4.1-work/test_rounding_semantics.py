"""Investigate numerical sensitivity of whole decoder reference checks to BF16 intermediate rounding.

Compares:
1. Exact reference path: stream -> RMSNorm -> cast to BF16 -> projection GEMM/GEMV -> downstream
2. Unrounded FP32 algebraic path: stream -> RMSNorm in FP32 -> projection GEMM/GEMV without intermediate BF16 rounding
3. Fused prologue with explicit BF16 rounding in registers/SRAM before MMA
Evaluates error bounds across all 12 shapes against check_implementation() tolerance.
"""
import sys
import os
import torch

# Ensure path includes kernels/muse_decode_layer
ROOT = "/home/nitiz/Projects/Self/Challenges/muse-spark"
sys.path.insert(0, f"{ROOT}/kernels/muse_decode_layer")

import reference as R
from checks import compare, query_precision_bound, relative_l2

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")
    weights = R.generate_weights(R.WEIGHTS_SEED, device=device)
    
    shapes = [
        ("decode 1 sliding", dict(batch=1, q_len=1, layer="sliding", dist="soft", page=1, ctx=80000, spread="fixed")),
        ("decode 1 full", dict(batch=1, q_len=1, layer="full", dist="soft", page=1, ctx=80000, spread="fixed")),
        ("decode 16 sliding", dict(batch=16, q_len=1, layer="sliding", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("decode 16 full", dict(batch=16, q_len=1, layer="full", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("decode 32 sliding", dict(batch=32, q_len=1, layer="sliding", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("decode 32 full", dict(batch=32, q_len=1, layer="full", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("verify 1 sliding", dict(batch=1, q_len=8, layer="sliding", dist="soft", page=1, ctx=80000, spread="fixed")),
        ("verify 1 full", dict(batch=1, q_len=8, layer="full", dist="soft", page=1, ctx=80000, spread="fixed")),
        ("verify 16 sliding", dict(batch=16, q_len=8, layer="sliding", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("verify 16 full", dict(batch=16, q_len=8, layer="full", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("verify 32 sliding", dict(batch=32, q_len=8, layer="sliding", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("verify 32 full", dict(batch=32, q_len=8, layer="full", dist="soft", page=1, ctx=80000, spread="agentic")),
    ]

    results = []
    for name, shape in shapes:
        seed = 4001
        cc = dict(seed=seed, **shape)
        ls = R.case_fields(cc, weights)["local_start"]
        data = R.generate_input(**cc, local_start=ls, device=device)
        
        # 1. Standard reference front & back
        front_ref = R.ref_front(data, weights)
        ids_ref = front_ref["biased"].topk(R.TOP_K, dim=-1).indices
        hidden_ref, stream_ref = R.ref_back(front_ref, weights, ids_ref)
        
        # 2. Check front intermediate h:
        # In ref_front:
        # stream1 = alpha1 * data["stream"].float() + beta1 * _rms(a, POST_NORM_EPS)
        # h = _rms(stream1) * (1.0 + weights["pre_ffn_norm"].float())
        # Notice ref_front keeps h in float32.
        # But in SGLang and decoder kernels, h is cast to BF16 when written to memory!
        h_f32 = front_ref["h"]
        h_bf16 = h_f32.to(torch.bfloat16).float()
        rel_diff_h = relative_l2(h_bf16, h_f32)
        
        # Check projection difference with h_bf16 vs h_f32:
        pooler_w = weights["pooler"].float()
        p_from_f32 = h_f32 @ pooler_w.T
        p_from_bf16 = h_bf16 @ pooler_w.T
        rel_diff_p = relative_l2(p_from_bf16, p_from_f32)
        
        router_w = weights["router"].float()
        logits_f32 = (h_f32 @ router_w.T) * R.ROUTER_MULTIPLIER
        logits_bf16 = (h_bf16 @ router_w.T) * R.ROUTER_MULTIPLIER
        rel_diff_logits = relative_l2(logits_bf16, logits_f32)
        
        # Expert routing check with logits_bf16
        biased_bf16 = logits_bf16 + weights["router_bias"].float()
        ids_bf16 = biased_bf16.topk(R.TOP_K, dim=-1).indices
        ids_match = (ids_ref == ids_bf16).float().mean().item()
        
        # Check if routing check passes:
        routing_err = R.check_routing(ids_bf16, front_ref["biased"], front_ref["local_start"])
        
        print(f"Shape: {name:20s} | rel_diff(h_bf16, h_f32)={rel_diff_h:.4e} | p_diff={rel_diff_p:.4e} | logits_diff={rel_diff_logits:.4e} | ids_match={ids_match*100:.1f}% | routing_ok={routing_err is None}")
        results.append({
            "name": name,
            "rel_diff_h": rel_diff_h,
            "rel_diff_p": rel_diff_p,
            "rel_diff_logits": rel_diff_logits,
            "ids_match": ids_match,
            "routing_ok": routing_err is None,
            "routing_err": routing_err
        })
        
    return results

if __name__ == "__main__":
    run_test()
