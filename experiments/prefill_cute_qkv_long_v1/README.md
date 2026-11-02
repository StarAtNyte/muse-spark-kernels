# CuTe WGMMA/TMA for long-prefill QKV

## Hypothesis

The `T=16,384`, `K=8,192`, `N=2,304` QKV GEMM is a large compute-heavy
prefill operation. Compare the production CuBLASLt choice with NVIDIA's
CuTe Hopper WGMMA/TMA dense GEMM example on the H200. This is motivated by
the recent workload-grounded TMA study, which finds that TMA's fixed issue
cost is not paid back on bandwidth-bound decode, while compute-bound prefill
GEMMs can benefit. The study used H100, so this is only a hypothesis for our
H200 shape.

## Scope and method

Run one scored shape's QKV component only. Use the exact current prefill
submission's `prepare()` and `_mm(..., "qkv")` as the parent route; compare
the CuTe candidate on the same inputs, with cold-L2 flushes, alternating
CUDA-graph replay order, and output comparison before timing. No candidate
submission or source promotion is implied by this screen. CuTe example code
remains external to the challenge submission and its BSD-3-Clause terms must
be preserved if it is ever vendored.

Driver: [`modal_screen.py`](modal_screen.py). Source baseline:
`experiments/prefill_keell0renz_compare_v1/submission.py`.

## Result: reject

Modal H200 component screen completed successfully on the CuTe image
(`nvidia-cutlass-dsl`; PyTorch 2.9.1/CUDA 12.8.1). All three CuTe variants
matched the production BF16 QKV output exactly, but each was slower than the
production route. Paired speedups (parent time / CuTe time) were 0.840× for
tile 128×128, 0.637× for 128×256, and 0.917× for 128×128 with a 2×1 cluster.
The CuTe route is rejected for this shape; no official check or submission.
Raw measurements: [`modal_screen.json`](modal_screen.json).

The kernel is compute heavy, but the challenge's cuBLAS/cuBLASLt path remains
substantially better than this generic CUTLASS DSL example. The TMA study's
positive prefill result therefore does not transfer automatically to an
alternate implementation; its result is not evidence that TMA is absent from
or poorly tuned in cuBLASLt.
