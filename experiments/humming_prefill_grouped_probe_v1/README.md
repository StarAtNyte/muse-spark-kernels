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
uses the required shape.

Corrected run `ap-hzUDo0V7zOrbl0qJziw6jj` completed with return code 0.
For 1044 synthetic local routes, the Humming output was finite and had 0.00448
relative L2 versus the incumbent's expert outputs. The median cold-L2 graph
times were 230.91 µs for the incumbent component including its gather/plan,
and 489.22 µs for Humming's two grouped GEMMs plus separate SwiGLU, with
Humming's gather and normalization precomputed. The median paired
parent/Humming ratio was 0.4722. This is not a full challenge correctness or
score test, and it does not isolate the cost of Humming's GEMMs versus the
separate activation. It rules against integrating this particular unfused
heuristic configuration; different kernels, fusion, tuning, or route sizes
remain untested. The incumbent remains unchanged.
