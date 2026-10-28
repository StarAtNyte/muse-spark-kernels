# Decoder pre-expert normalization

Measured production decoder profiles show generic PyTorch pre-expert RMSNorm
costs about 6 µs at small batch. The pinned Muse RMSNorm applies its already
rounded BF16 effective weight `(weight + 1)` and returns BF16. The replacement
retains that weight rounding, accumulates the norm in FP32, and returns BF16.

`build.py` generates two alternatives. The standalone Triton norm has one block
per row (four warps through M16, eight beyond). The fused CUDA version adds
normalization to the existing route kernel for M<=8: eight groups of 128
threads each normalize one row, sharing partial sums at the route kernel's
existing barrier. All route metadata and normalized input are produced inside
the timed call. This does not prune experts or cache input-derived results.

CPU compile passes at 63 registers without spills, under the 1024-thread
launch bound. `screen.log` hit a diagnostic host-wrapper linking error before
numerical execution; the exported C launcher fixed it in `screen_v2.log`.

## First stage screen

`screen_v2.log` uses Torch 2.9 on H200, 256 MiB cold-L2 flushes and alternating
paired graph timings. All 34 numerical and route-metadata checks passed on two
seeds and M1/4/8/16/32/128/256. The standalone version saved about 5 µs versus
PyTorch norm plus the same routing. Relative error versus FP32 was ~1e-5.
The fused scalar version lost: ~19–21 µs versus 14 µs for the original
combined path, so it is excluded from the official candidate.

`decoder_pre_norm_v1.py` contains only the standalone replacement, based on the
immutable decoder_transpose_selected parent. It is undergoing full official
validation as `decoder_pre_norm_v1`, job `c0f2e3bd78904982`. It does not contain
the experimental fused routing, and keeps every other production operation.

## Fused vector-access ablation

`make_vector.py` changes the fused kernel to four 16-byte reads per thread per
pass and vector stores. Normalization still covers all 4096 features and uses
the existing route barriers. CPU `vector_compile.log` passes without spills.
`vector_screen.log` compares the combined norm+route path, not isolated kernel
latency. No official check before independent numerical and routing checks,
a consistent runtime gain, and complete source integration.
