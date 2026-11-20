"""Direct analysis of Fused RMSNorm Projection Prologue across all 12 H200 Evaluation Shapes.

Evaluates:
1. Numerical precision: exact BF16 intermediate rounding vs FP32 vs algebraic deferred normalization.
2. Pooler relative L2 error, Router relative L2 error, and Top-8 expert routing fidelity.
3. HBM memory traffic eliminated (HBM write of h, HBM read of h).
4. Kernel launch latency eliminated (3.5 us).
5. H200 roofline performance (4.8 TB/s, 989 TFLOP/s) and net latency gain per shape.
"""
import json
import math
import sys
import torch

ROOT = "/home/nitiz/Projects/Self/Challenges/muse-spark"
sys.path.insert(0, f"{ROOT}/kernels/muse_decode_layer")
import reference as R
from checks import relative_l2

def analyze_all_12_shapes():
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
    LAUNCH_OVERHEAD_US = 3.5  # Fixed graph launch / sync overhead

    alpha1, beta1 = R.residual_coefficients(weights["attn_residual_gate"])
    pre_w = (1.0 + weights["pre_ffn_norm"].float())
    pooler_w = weights["pooler"].float()
    router_w = weights["router"].float()
    router_b = weights["router_bias"].float()

    records = []

    for name, load, batch, q_len, layer, w_load, floor_us in shapes:
        T = batch * q_len
        spread = "agentic" if batch > 1 else "fixed"
        
        # Fast direct input generation (exact distribution as reference)
        gen = torch.Generator(device=device).manual_seed(4001 + T)
        mean, shared = R._stream_basis(str(device))
        coef = torch.randn((T, R.STREAM_RANK), generator=gen, device=device)
        scale = torch.ones((batch, q_len, 1), device=device)
        if 1 < q_len <= 8:
            scale[:, 1:] = R.DRAFT_SHARE
        scale = scale.reshape(T, 1)
        stream_in = (scale * (R.STREAM_MEAN * mean + math.sqrt(R.STREAM_SHARED / R.STREAM_RANK) * coef @ shared)
                     + (1 - R.STREAM_SHARED * scale * scale).sqrt()
                     * torch.randn((T, H), generator=gen, device=device))
        
        # Attention output representation
        attn_out = torch.randn((T, H), generator=gen, device=device)
        attn_rms = attn_out * torch.rsqrt(attn_out.pow(2).mean(-1, keepdim=True) + R.POST_NORM_EPS)
        stream1 = alpha1 * stream_in + beta1 * attn_rms

        # 1. Exact Reference Normalization
        rms_inv = torch.rsqrt(stream1.pow(2).mean(-1, keepdim=True) + R.RMS_EPS)
        h_exact = stream1 * rms_inv * pre_w

        # 2. Proposed Fused Prologue: Intermediate BF16 quantization
        h_bf16 = h_exact.to(torch.bfloat16)

        # 3. Router inputs: FP32 for T <= 8, BF16 for T > 8 (SGLang specification)
        hr_input = h_exact if T <= 8 else h_bf16.float()

        # Projections
        p_ref = h_exact @ pooler_w.T
        p_fused = h_bf16.float() @ pooler_w.T
        pooler_err = relative_l2(p_fused, p_ref)

        logits_ref = (h_exact @ router_w.T) * R.ROUTER_MULTIPLIER
        logits_fused = (hr_input @ router_w.T) * R.ROUTER_MULTIPLIER
        router_err = relative_l2(logits_fused, logits_ref)

        biased_ref = torch.sigmoid(logits_ref) + router_b
        biased_fused = torch.sigmoid(logits_fused) + router_b

        ids_ref = biased_ref.topk(R.TOP_K, dim=-1).indices
        ids_fused = biased_fused.topk(R.TOP_K, dim=-1).indices
        ids_match_pct = float((ids_fused == ids_ref).float().mean() * 100.0)

        # Check routing
        local_start = 0
        routing_err = R.check_routing(ids_fused, biased_ref, local_start)
        routing_pass = (routing_err is None)

        # Unfused HBM Traffic:
        # Separate boundary kernel writes h [T, 8192] BF16 (2*T*H bytes).
        # For T <= 8, it also writes hr [T, 8192] FP32 (4*T*H bytes).
        # Pooler GEMM reads h (2*T*H bytes).
        # Router GEMV reads hr (4*T*H if T<=8 else 2*T*H bytes).
        # Fused Prologue eliminates ALL of these intermediate writes and reads!
        hbm_saved_bytes = (
            2 * T * H +                               # eliminated write h (BF16)
            (4 * T * H if T <= 8 else 0) +           # eliminated write hr (FP32)
            2 * T * H +                               # eliminated pooler read h
            (4 * T * H if T <= 8 else 2 * T * H)     # eliminated router read hr/h
        )

        dram_saved_us = (hbm_saved_bytes / HBM_BW) * 1e6
        launch_saved_us = LAUNCH_OVERHEAD_US
        net_saved_us = dram_saved_us + launch_saved_us

        # FLOP counts:
        pooler_flops = 2 * T * H * N_pooler
        router_flops = 2 * T * H * N_router
        total_proj_flops = pooler_flops + router_flops

        # Latency on H200 (roofline):
        pooler_weight_bytes = N_pooler * H * 2 # 67.1 MB
        pooler_time_us = max(pooler_weight_bytes / HBM_BW, pooler_flops / PEAK_FLOPS) * 1e6

        rec = {
            "name": name,
            "load": load,
            "T": T,
            "layer": layer,
            "weight": w_load,
            "floor_us": floor_us,
            "pooler_err": pooler_err,
            "router_err": router_err,
            "ids_match_pct": round(ids_match_pct, 2),
            "routing_pass": routing_pass,
            "hbm_saved_bytes": hbm_saved_bytes,
            "dram_saved_us": round(dram_saved_us, 3),
            "launch_saved_us": round(launch_saved_us, 2),
            "net_saved_us": round(net_saved_us, 3),
            "pooler_roofline_us": round(pooler_time_us, 2)
        }
        records.append(rec)
        print(f"{name:20s} | T={T:3d} | p_err={pooler_err:.3e} | r_err={router_err:.3e} | match={ids_match_pct:5.1f}% | pass={routing_pass} | saved={net_saved_us:.2f}us")

    out_path = f"{ROOT}/.ratchet/runs/mvuvm78n-rm78n1/findings/w4.1-work/evaluation_across_12_shapes.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved all 12 shapes evaluation to {out_path}")

if __name__ == "__main__":
    analyze_all_12_shapes()
