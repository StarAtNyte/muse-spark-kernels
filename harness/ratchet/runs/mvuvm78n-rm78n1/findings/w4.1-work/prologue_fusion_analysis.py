"""Prologue Fusion vs Deferred Normalization: Comprehensive Analysis across all 12 H200 Evaluation Shapes.

Analyzes:
1. Mathematical semantics:
   - Exact SGLang reference: s1 -> RMSNorm in FP32 -> * (1+w) -> round to BF16 -> W @ h_bf16
   - Deferred algebraic (GLE_T): 2 GEMMs (W @ u + W @ v) -> post-scale by r2
   - Fused prologue: on-chip RMSNorm in FP32 -> cast to BF16 in SRAM -> 1 GEMM (W @ h_bf16)
2. Numerical fidelity across all 12 evaluation shapes.
3. HBM Memory Traffic (Bytes read/written) and FLOP count.
4. Latency and roofline bounds on H200 (4.8 TB/s, 989 TFLOP/s).
"""
import json
import math
import sys
import torch

ROOT = "/home/nitiz/Projects/Self/Challenges/muse-spark"
sys.path.insert(0, f"{ROOT}/kernels/muse_decode_layer")
import reference as R
from checks import relative_l2

def analyze_all_shapes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = R.generate_weights(R.WEIGHTS_SEED, device=device)
    
    # 12 official evaluation shapes
    shapes = [
        # (name, load, batch, q_len, layer, weight, floor_us)
        ("decode_1_sliding", "decode 1", 1, 1, "sliding", 46, 40.0),
        ("decode_1_full", "decode 1", 1, 1, "full", 16, 44.2),
        ("decode_16_sliding", "decode 16", 16, 1, "sliding", 46, 76.7),
        ("decode_16_full", "decode 16", 16, 1, "full", 16, 137.5),
        ("decode_32_sliding", "decode 32", 32, 1, "sliding", 46, 84.5),
        ("decode_32_full", "decode 32", 32, 1, "full", 16, 208.1),
        ("verify_1_sliding", "verify 1", 1, 8, "sliding", 46, 62.3),
        ("verify_1_full", "verify 1", 1, 8, "full", 16, 66.4),
        ("verify_16_sliding", "verify 16", 16, 8, "sliding", 46, 126.6),
        ("verify_16_full", "verify 16", 16, 8, "full", 16, 188.7),
        ("verify_32_sliding", "verify 32", 32, 8, "sliding", 46, 143.8),
        ("verify_32_full", "verify 32", 32, 8, "full", 16, 267.2),
    ]

    H = 8192      # hidden dim
    N_pooler = 4096
    N_router = 256
    HBM_BW = 4.8e12  # 4.8 TB/s on H200
    PEAK_FLOPS = 989e12 # 989 TFLOP/s BF16 on H200
    LAUNCH_OVERHEAD_US = 3.5  # Typical CUDA graph replay fixed boundary kernel launch overhead

    records = []

    for name, load, batch, q_len, layer, w_load, floor_us in shapes:
        T = batch * q_len
        spread = "agentic" if batch > 1 else "fixed"
        cc = dict(batch=batch, q_len=q_len, layer=layer, dist="soft", page=1, ctx=80000, spread=spread, seed=4001)
        ls = R.case_fields(cc, weights)["local_start"]
        data = R.generate_input(**cc, local_start=ls, device=device)
        
        # Ground truth reference
        front = R.ref_front(data, weights)
        stream1 = front["stream1"]  # FP32 [T, 8192]
        pre_w = (1.0 + weights["pre_ffn_norm"].float())
        
        # 1. Exact RMSNorm with BF16 rounding (Proposed Fused Prologue)
        rms_inv = torch.rsqrt(stream1.pow(2).mean(-1, keepdim=True) + R.RMS_EPS)
        h_fused_bf16 = (stream1 * rms_inv * pre_w).to(torch.bfloat16)
        
        # 2. Router input: FP32 for T <= 8, BF16 for T > 8
        hr_fused_f32 = stream1 * rms_inv * pre_w if T <= 8 else h_fused_bf16.float()
        
        # 3. Check Pooler and Router relative differences vs reference
        pooler_w = weights["pooler"].float()
        p_ref = front["h"] @ pooler_w.T
        p_fused = h_fused_bf16.float() @ pooler_w.T
        pooler_err = relative_l2(p_fused, p_ref)
        
        router_w = weights["router"].float()
        logits_ref = (front["h"] @ router_w.T) * R.ROUTER_MULTIPLIER
        logits_fused = (hr_fused_f32 @ router_w.T) * R.ROUTER_MULTIPLIER
        router_err = relative_l2(logits_fused, logits_ref)
        
        # Routing check
        biased_ref = front["biased"]
        biased_fused = torch.sigmoid(logits_fused) + weights["router_bias"].float()
        ids_fused = biased_fused.topk(R.TOP_K, dim=-1).indices
        ids_ref = biased_ref.topk(R.TOP_K, dim=-1).indices
        ids_match_pct = float((ids_fused == ids_ref).float().mean() * 100.0)
        routing_err = R.check_routing(ids_fused, biased_ref, front["local_start"])
        routing_pass = (routing_err is None)
        
        # Traffic Analysis:
        # Unfused:
        # Boundary kernel writes s1 (FP32: 4*T*H bytes), writes h (BF16: 2*T*H bytes),
        # writes hr if T<=8 (FP32: 4*T*H bytes).
        # Pooler GEMM reads h (BF16: 2*T*H bytes).
        # Router GEMM reads hr (4*T*H if T<=8 else 2*T*H).
        hbm_unfused_norm_traffic = (
            4 * T * H +   # write s1
            2 * T * H +   # write h
            (4 * T * H if T <= 8 else 0) + # write hr
            2 * T * H +   # pooler reads h
            (4 * T * H if T <= 8 else 2 * T * H) # router reads hr/h
        )
        
        # Fused Prologue:
        # Boundary kernel does not write h or hr to DRAM!
        # Pooler & Router compute RMSNorm on s1 in prologue from on-chip SRAM/L2.
        # Saves writing h, writing hr, reading h, reading hr.
        hbm_saved_bytes = (
            2 * T * H +   # eliminated write h
            (4 * T * H if T <= 8 else 0) + # eliminated write hr
            2 * T * H +   # eliminated pooler read h
            (4 * T * H if T <= 8 else 2 * T * H) # eliminated router read hr/h
        )
        
        # Time savings:
        # Pure DRAM transfer savings (us):
        dram_saved_us = (hbm_saved_bytes / HBM_BW) * 1e6
        # Kernel launch latency eliminated (us):
        launch_saved_us = LAUNCH_OVERHEAD_US
        total_saved_us = dram_saved_us + launch_saved_us

        # FLOP analysis:
        # Standard Pooler FLOPs: 2 * T * H * N_pooler
        # Standard Router FLOPs: 2 * T * H * N_router
        gemm_flops = 2 * T * H * (N_pooler + N_router)
        
        # Deferred Algebraic (GLE_T) FLOPs:
        # Must compute 2 GEMMs: W @ u and W @ v -> 2x FLOPs!
        deferred_flops = 2 * gemm_flops
        
        rec = {
            "name": name,
            "load": load,
            "T": T,
            "layer": layer,
            "pooler_err": pooler_err,
            "router_err": router_err,
            "ids_match_pct": round(ids_match_pct, 2),
            "routing_pass": routing_pass,
            "routing_err": str(routing_err),
            "hbm_unfused_bytes": hbm_unfused_norm_traffic,
            "hbm_saved_bytes": hbm_saved_bytes,
            "dram_saved_us": round(dram_saved_us, 3),
            "launch_saved_us": round(launch_saved_us, 2),
            "total_saved_us": round(total_saved_us, 3),
            "gemm_flops": gemm_flops,
            "deferred_flops": deferred_flops,
            "floor_us": floor_us
        }
        records.append(rec)
        print(f"[{name}] T={T:3d} | pooler_err={pooler_err:.3e} | router_err={router_err:.3e} | ids_match={ids_match_pct:5.1f}% | pass={routing_pass} | saved={total_saved_us:.2f}us")

    out_path = f"{ROOT}/.ratchet/runs/mvuvm78n-rm78n1/findings/w4.1-work/prologue_fusion_results.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved analysis results to {out_path}")

if __name__ == "__main__":
    analyze_all_shapes()
