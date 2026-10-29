# wide2 wgmma sweep on M128/M256 projections (2026-10-29)

Driver `dev/modal/drv_wide2_sweep.py` (CUDA 13 H200; bn {64,128,256} x split {1,2,3,4,6,8} x stages {3..8} x hint;
every config checked against torch.mm; cold-L2 graph medians, event-timed incl. ~5 us launch).

| shape | current route | best wide2 (bn, split, stages, hint) | verdict |
|---|---:|---:|---|
| qkv M256 | SGLang dispatch (stubbed here); plain torch.mm 28.8 us in run 1 | 27.0 us (256,3,5,1) | ~7% faster than torch.mm; ~0.25% of the projection score, below judge noise |
| o M256 | 14.0 us | 16.2 us (256,1,5,1) | wide2 loses |
| o M128 | 12.3 us | 13.9 us (128,1,6,1) | wide2 loses |
| pooler M256 | 35.6 us (already wide2) | 36.0 us (256,2,5,1) | tie |
| unpooler M128/M256 | SGLang dispatch (stubbed) | 27.9 / 33.9 us | no valid baseline here |

Routes that call SGLang's `_bf16_gemm_dispatch_impl` cannot be timed on Modal (sglang is stubbed; the graph is empty,
3.17 us). Conclusion: the custom wgmma kernel has no material headroom over the current M128/M256 routes.
