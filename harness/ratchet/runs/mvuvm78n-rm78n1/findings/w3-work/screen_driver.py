"""Modal H200 Paired Screen for Projection w3 Investigation.
Compares base (prior hybrid route) vs candidate (expanded vjeux route) using a single prepared state.
Zero duplicate compilations; checks all 9 correctness tests + paired timings for target and control shapes.
"""
import importlib.util
import json
import os
import statistics
import sys
import torch

sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R

# Load candidate
spec = importlib.util.spec_from_file_location("candidate", "/w/run/sub.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

print("Preparing weights...", flush=True)
weights = R.generate_weights(R.WEIGHTS_SEED)
state = cand.prepare(weights)
print("Preparation complete.", flush=True)

# 1. Run the official 9 correctness tests
test_cases = [
    (1, "qkv_gate", "normal"),
    (7, "pooler", "normal"),
    (33, "o", "normal"),
    (130, "unpooler", "normal"),
    (5, "unpooler", "normal"),
    (8, "router", "normal"),
    (9, "router", "outliers"),
    (512, "qkv_gate", "outliers"),
    (1000, "unpooler", "outliers"),
]

passed_tests = 0
for m, layer, dist in test_cases:
    d = R.generate_input(m, layer, seed=123, dist=dist)
    out = cand.kernel_fn(d, state)
    ok, msg = R.check_implementation(d, out, weights)
    if not ok:
        print(f"FAILED TEST: m={m} layer={layer} dist={dist}: {msg}", flush=True)
        sys.exit(1)
    passed_tests += 1
print(f"ALL {passed_tests}/9 OFFICIAL TESTS PASSED!", flush=True)

# 2. Paired timing on target shapes and controls
target_shapes = [
    (1, "unpooler"),
    (1, "router"),
    (32, "qkv_gate"),
    (32, "unpooler"),
    (256, "router"),
    # Controls (untargeted shapes)
    (1, "qkv_gate"),
    (16, "pooler"),
    (128, "pooler"),
    (256, "pooler"),
]

flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)

def capture(fn):
    out = fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out

def elapsed_us(g, a, b):
    flush.add_(1)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0

event_a = torch.cuda.Event(enable_timing=True)
event_b = torch.cuda.Event(enable_timing=True)

results = []
for m, layer in target_shapes:
    shape_ratios = []
    base_times = []
    cand_times = []
    
    for seed in (42, 108, 999):
        data = R.generate_input(m, layer, seed=seed)
        
        # Base uses _hybrid_base_kernel
        gb, ob = capture(lambda: cand._hybrid_base_kernel(data, state))
        # Cand uses kernel_fn
        gc, oc = capture(lambda: cand.kernel_fn(data, state))
        
        # Correctness check on timed input
        ok_b, msg_b = R.check_implementation(data, ob, weights)
        ok_c, msg_c = R.check_implementation(data, oc, weights)
        if not ok_b:
            print(f"Base correctness fail on m={m} layer={layer}: {msg_b}", flush=True)
            sys.exit(1)
        if not ok_c:
            print(f"Cand correctness fail on m={m} layer={layer}: {msg_c}", flush=True)
            sys.exit(1)
            
        tb, tc = [], []
        for rep in range(30):
            if rep % 2 == 0:
                tb.append(elapsed_us(gb, event_a, event_b))
                tc.append(elapsed_us(gc, event_a, event_b))
            else:
                tc.append(elapsed_us(gc, event_a, event_b))
                tb.append(elapsed_us(gb, event_a, event_b))
                
        med_b = statistics.median(tb)
        med_c = statistics.median(tc)
        base_times.append(med_b)
        cand_times.append(med_c)
        shape_ratios.append(med_b / med_c)
        del gb, gc, ob, oc, data
        
    gm_b = statistics.geometric_mean(base_times)
    gm_c = statistics.geometric_mean(cand_times)
    gm_sp = statistics.geometric_mean(shape_ratios)
    sol = R.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989.5e12}) / 1000.0
    
    row = {
        "m": m,
        "layer": layer,
        "base_us": round(gm_b, 2),
        "cand_us": round(gm_c, 2),
        "sol_us": round(sol, 2),
        "speedup": round(gm_sp, 4),
        "diff_us": round(gm_c - gm_b, 2),
    }
    results.append(row)
    print("SCREEN_ROW", json.dumps(row), flush=True)

print("SUMMARY_JSON", json.dumps(results, indent=2), flush=True)
