"""Evaluate algebraic fusion vs fused prologue with exact BF16 rounding on whole-decoder correctness.

Case 1: Standard baseline (separate boundary kernel storing BF16 h and FP32 hr to DRAM, GEMMs read from DRAM).
Case 2: Fused prologue with exact BF16 rounding (RMSNorm computed on-chip, cast to BF16 in SRAM/registers before MMA).
Case 3: Unbounded algebraic shortcut (pulling 1/rms out of GEMM, W @ (s1 * (1+w)) / rms, omitting intermediate BF16 rounding).
"""
import sys
import torch

ROOT = "/home/nitiz/Projects/Self/Challenges/muse-spark"
sys.path.insert(0, f"{ROOT}/kernels/muse_decode_layer")
import reference as R

def test_algebraic_vs_exact():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = R.generate_weights(R.WEIGHTS_SEED, device=device)
    
    shapes = [
        ("decode 1 sliding", dict(batch=1, q_len=1, layer="sliding", dist="soft", page=1, ctx=80000, spread="fixed")),
        ("decode 16 sliding", dict(batch=16, q_len=1, layer="sliding", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("decode 32 full", dict(batch=32, q_len=1, layer="full", dist="soft", page=1, ctx=80000, spread="agentic")),
        ("verify 1 sliding", dict(batch=1, q_len=8, layer="sliding", dist="soft", page=1, ctx=80000, spread="fixed")),
        ("verify 16 sliding", dict(batch=16, q_len=8, layer="sliding", dist="soft", page=1, ctx=80000, spread="agentic")),
    ]
    
    for name, shape in shapes:
        cc = dict(seed=4001, **shape)
        ls = R.case_fields(cc, weights)["local_start"]
        data = R.generate_input(**cc, local_start=ls, device=device)
        T = shape["batch"] * shape["q_len"]
        
        # Reference execution
        front = R.ref_front(data, weights)
        ids_ref = front["biased"].topk(R.TOP_K, dim=-1).indices
        ref_hidden, ref_stream = R.ref_back(front, weights, ids_ref)
        
        # Residual stream after attention: stream1
        stream1 = front["stream1"]  # FP32 [T, 8192]
        pre_w = (1.0 + weights["pre_ffn_norm"].float())
        
        # Ground truth RMSNorm with BF16 rounding (Case 2: Fused prologue exact)
        rms_inv = torch.rsqrt(stream1.pow(2).mean(-1, keepdim=True) + R.RMS_EPS)
        h_exact_bf16 = (stream1 * rms_inv * pre_w).to(torch.bfloat16)
        
        # Algebraic shortcut without BF16 rounding (Case 3)
        h_algebraic = stream1 * rms_inv * pre_w  # kept in FP32
        
        # Test Pooler GEMM output:
        pooler_w = weights["pooler"].float()
        p_ref = front["h"] @ pooler_w.T  # in ref_back, front["h"] is float
        p_exact = h_exact_bf16.float() @ pooler_w.T
        p_alg = h_algebraic @ pooler_w.T
        
        # Error metrics:
        err_p_exact = float((p_exact - p_ref).norm() / p_ref.norm())
        err_p_alg = float((p_alg - p_ref).norm() / p_ref.norm())
        
        # Now run back half with p_exact and p_alg to see whole-layer output error
        # Mocking back half
        front_exact = dict(front, h=h_exact_bf16.float())
        front_alg = dict(front, h=h_algebraic)
        
        hid_exact, str_exact = R.ref_back(front_exact, weights, ids_ref)
        hid_alg, str_alg = R.ref_back(front_alg, weights, ids_ref)
        
        err_hid_exact = float((hid_exact.float() - ref_hidden.float()).norm() / ref_hidden.float().norm())
        err_hid_alg = float((hid_alg.float() - ref_hidden.float()).norm() / ref_hidden.float().norm())
        
        err_str_exact = float((str_exact - ref_stream).norm() / ref_stream.norm())
        err_str_alg = float((str_alg - ref_stream).norm() / ref_stream.norm())
        
        # Also check compare() tolerances
        carried = front["stream1"]  # residual carried
        print(f"=== {name} (T={T}) ===")
        print(f"  Pooler rel L2: exact_bf16={err_p_exact:.4e}, algebraic_f32={err_p_alg:.4e}")
        print(f"  Next hidden rel L2: exact_bf16={err_hid_exact:.4e}, algebraic_f32={err_hid_alg:.4e}")
        print(f"  Stream rel L2: exact_bf16={err_str_exact:.4e}, algebraic_f32={err_str_alg:.4e}")
        
        # Populate cache with front["k_new"] and front["v_new"] so KV cache check passes
        slots = R.new_slots(data).reshape(-1)
        data["k_cache"].view(-1, R.HKV, R.HEAD_DIM)[slots] = front["k_new"].view(-1, R.HKV, R.HEAD_DIM)
        data["v_cache"].view(-1, R.HKV, R.HEAD_DIM)[slots] = front["v_new"].view(-1, R.HKV, R.HEAD_DIM)

        # Check whole-layer implementation check on exact_bf16
        raw_exact = R.collect(data, (hid_exact.to(torch.bfloat16), str_exact.float(), ids_ref.to(torch.int32)))
        ok_exact, msg_exact = R.check_implementation(data, raw_exact, weights)
        print(f"  check_implementation(exact_bf16): {ok_exact} - {msg_exact}")
        
        raw_alg = R.collect(data, (hid_alg.to(torch.bfloat16), str_alg.float(), ids_ref.to(torch.int32)))
        ok_alg, msg_alg = R.check_implementation(data, raw_alg, weights)
        print(f"  check_implementation(algebraic): {ok_alg} - {msg_alg}")

if __name__ == "__main__":
    test_algebraic_vs_exact()
