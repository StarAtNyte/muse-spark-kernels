# Exact expert-count dispatch on H200

Hypothesis: the production row-split kernel reserves resources for up to 32
routed tokens even when an expert has only a few. Stable GPU-side routing splits
work records into counts 1–8, 9–16, and 17–32. Each bucket computes gate/up then
down on a separate stream; all streams join before the original normalization
and weighted reduction. Routing, synchronization and finish are timed.

The original route list remains intact. Diagnostic checks compare the multiset
of bucket records against the original, check tier boundaries, and validate
end-to-end outputs. No token or expert is omitted.

## Resource-only screen

`screen.log`: 10/10 task checks and 5/5 metadata checks passed; outputs were
bit-identical to production. Sequential buckets were substantially slower.
Concurrent buckets gave 0.960× and 0.968× on two normal batch-128 seeds,
1.030× and 1.102× on batch 256, and 0.994× on skewed batch 128.
This is not a consistent production win and has not been submitted.

## Per-expert algorithm screen

`build_native.py` applies production warp split-K GEMV to sparse experts and
retains row-split for dense ones. A separate activation-L1-cache variant is an
ablation. CPU device compilation succeeded with no register spills. Native
variants must pass fresh H200 correctness and paired timing before full official
coverage. The diagnostic bundle prunes unrelated fallback kernels; it must never
be used as the submission source.

All GPU timings use alternating paired CUDA graphs with a 256 MiB L2 flush.
Local diagnostics use the same Triton finish for both sides; the official server
is required to validate total workload performance with its installed libraries.

### Native screen result

`native_screen.log`: all 15 task checks and 5 metadata checks passed. Native
split-K sparse dispatch lost on all normal cases (0.880–0.951×); activation L1
caching improved it but still gave only 0.927–1.002×. Do not promote it.
Row-split buckets repeated batch-256 wins: 1.023× and 1.100×, with exactly
identical outputs. Batch 128 remained slower. The complete candidate
`moe_bucket_row_m256.py` therefore gates this optimization to M=256, retaining
the production path below that threshold. It is undergoing official private
validation, including the unscored neighboring M257 correctness test.

### Occupancy / pipeline ablation

`caps_screen.log`: all 25 task checks plus 5 metadata checks passed, every output
bit-identical to production. Raising the sparse/middle grid caps to 4/4 or 5/4
blocks per SM gave little benefit. Shortening the prefetch ring to 2 stages for
the light bucket and 3 for the others hurt throughput. Keep the original 4-stage
ring and 4/2 caps. Batch-256 control gains repeated at 1.026× and 1.101×; no
second official check is warranted for these ablations.

## Official validation

Private job `8df80c8aed0f40a4` passed all 5 tests and 6 timed cases.
Aggregate 1.933478× (95% interval 1.904476–1.962480), versus previous
private 1.916556×. Batch-256 candidate latency was 270.93 µs versus the
previous 287.40 µs (different runs); unchanged shapes also vary, so the
aggregate difference alone does not establish significance. The repeated
paired H200 controls support the batch-256 optimization. Promoted full source
SHA256 `cc0a11862ff7d7a74db5b5289ad5a4376b5de5cee7ae981bed28beaede1e278e`.
Public run `08d8cf0e6cb54ca8` passed at 1.933627× (CI 1.904622–1.962631), closely matching the private check.
