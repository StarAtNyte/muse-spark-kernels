import importlib.util
import json
import os
import statistics
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

sys.path.insert(0, "/w/kernels/muse_prefill_layer")
import reference as R
sys.path.insert(0, "/w/cute_examples/cute/hopper/kernel/dense_gemm")
from dense_gemm import HopperWgmmaGemmKernel

spec = importlib.util.spec_from_file_location("parent", "/w/run/sub.py")
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)

weights = R.generate_weights(20261102)
state = parent.prepare(weights)
bench_stream = torch.cuda.Stream()
bench_stream.wait_stream(torch.cuda.current_stream())
torch.cuda.set_stream(bench_stream)
T, K = 16384, 8192
x = (torch.randn((T, K), device="cuda") * 0.5).to(torch.bfloat16)
w = state["qkv_gate"]
assert tuple(w.shape) == (2304, K)

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)


def capture(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=torch.cuda.current_stream()):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def elapsed(g):
    flush.add_(1)
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000


parent_graph = capture(lambda: parent._mm(state, x, "qkv"))
print("GPU", torch.cuda.get_device_name(), "CuTe", cutlass.__file__, flush=True)

for tm, tn, cluster in ((128, 128, (1, 1)), (128, 256, (1, 1)), (128, 128, (2, 1))):
    try:
        # CuTe GEMM computes A[N,K] @ B[T,K]^T -> C[N,T]. The singleton
        # final dimension follows the example driver's batched layout.
        a = w.view(2304, 1, K).permute(0, 2, 1)
        b = x.view(T, 1, K).permute(0, 2, 1)
        out = torch.empty((1, T, 2304), dtype=torch.bfloat16, device="cuda")
        c = out.permute(2, 1, 0)
        ma, mb, mc = [from_dlpack(v, assumed_align=16) for v in (a, b, c)]
        gemm = HopperWgmmaGemmKernel(cutlass.Float32, (tm, tn), cluster)
        gemm.occupancy = 2
        compiled = cute.compile(gemm, ma, mb, mc, stream)
        candidate_graph = capture(lambda: compiled(ma, mb, mc, stream))

        # The CuTe output must reflect a new activation tensor after capture.
        x.copy_((torch.randn_like(x.float()) * 0.5).to(torch.bfloat16))
        candidate_graph.replay()
        torch.cuda.synchronize()
        candidate = out[0]
        reference = parent._mm(state, x, "qkv")
        max_rel = float((candidate.float() - reference.float()).abs().max() /
                        reference.float().abs().max().clamp_min(1e-6))
        rms_rel = float(torch.linalg.vector_norm(candidate.float() - reference.float()) /
                        torch.linalg.vector_norm(reference.float()).clamp_min(1e-6))
        if max_rel > 0.02 or rms_rel > 0.002:
            raise ValueError(f"numerical mismatch max_rel={max_rel:.6g} rms_rel={rms_rel:.6g}")

        # Refresh the parent graph after changing x; both graphs replay the
        # same activation pointer. Alternate timing order to reduce drift.
        ratios, pt, ct = [], [], []
        for rep in range(16):
            if rep & 1:
                tc, tp = elapsed(candidate_graph), elapsed(parent_graph)
            else:
                tp, tc = elapsed(parent_graph), elapsed(candidate_graph)
            ratios.append(tp / tc)
            pt.append(tp)
            ct.append(tc)
        print("RESULT", json.dumps({
            "shape": [T, 2304, K], "tile_mn": [tm, tn], "cluster": cluster,
            "parent_median_us": statistics.median(pt),
            "candidate_median_us": statistics.median(ct),
            "paired_speedup": statistics.median(ratios),
            "max_rel_error": max_rel, "rms_rel_error": rms_rel,
        }), flush=True)
    except Exception as exc:
        print("FAIL", tm, tn, cluster, type(exc).__name__, str(exc)[:600], flush=True)
