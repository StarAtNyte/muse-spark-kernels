# Humming grouped-prefill feasibility screen

This diagnostic follows `../humming_nvfp4_dense_probe_v1/` using the same
pinned Apache-2.0 Humming checkout and public H200 image. It compares two
grouped BF16 × NVFP4 expert GEMMs with a BF16 SwiGLU intermediate against the
scored prefill MoE component. Both use the same generated 32-expert weight
tensors, E4M3 group-16 scales and per-expert FP32 scales. The route counts are
copied from one previously reference-checked short-prefill diagnostic; they
are **synthetic benchmark data only**, never a submission assumption.

The Humming arm uses precomputed grouped routing and normalized input outside
the timed graph; the incumbent arm includes its gather/plan. Thus this gives
Humming an advantage and cannot establish an end-to-end speedup. It compares
per-route outputs to the incumbent with a 5% relative-L2 exploratory threshold,
not the full challenge reference. The first GEMM writes BF16 before SwiGLU,
as the challenge baseline does, but any actual integration must independently
validate the complete precision contract and all output/cache checks. No
external Humming dependency is assumed available on the judge.

The runner verifies Humming commit `948e6fd547e703065be51529ac1db5569256774e`
from the local clone before mounting its source; the subprocess has a
300-second limit. Raw output and failures are retained in `modal.log`.

The first run `ap-aecAqH6TXOK44tIM93B0A4` stopped at Humming weight loading:
the adapter supplied the per-expert secondary scale as `[32, 1]`, while the
layer requires `[32]`. It did not produce a correctness or performance
result. The failure is preserved in `initial_failure.log`; the adapter now
uses the required shape, and the corrected run is pending.
