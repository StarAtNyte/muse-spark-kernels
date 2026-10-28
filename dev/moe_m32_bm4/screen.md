# M32 BM4 H200 screen

Compared `submission.py` against the validated `moe_hybrid_m32_m128_v2` source on one Modal H200, with Triton 3.8. The only source change is selecting a 4-row Gluon tile when `M == 32`; all other dispatches and code remain unchanged.

The candidate was bit-identical to the validated winner on all three fresh seeds. Cold-L2 paired CUDA graph timings:

| Seed | Winner (us) | BM4 (us) | Winner / BM4 |
|---:|---:|---:|---:|
| 419 | 143.12 | 148.77 | 0.9620x |
| 7919 | 159.76 | 172.21 | 0.9277x |
| 15241 | 154.27 | 159.12 | 0.9695x |

Decision: reject for performance; preserve the official winner. No official check or score was used. The challenge reference checker is unreliable for M32 in this Modal diagnostic harness, so correctness was compared to the official, fully validated source; outputs were bit-identical.
