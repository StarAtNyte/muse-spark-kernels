# MonoMoE transfer audit (2026-11-02)

## Source reviewed

- Paper: [MonoMoE: An Efficient Fused Mega-kernel for Quantized MoE Decoding](https://arxiv.org/abs/2609.04244), revised August 2026.
- Reference implementation: [FlashInfer `csrc/fused_moe/monomoe`](https://github.com/flashinfer-ai/flashinfer/tree/main/csrc/fused_moe/monomoe), checkout `0dbfeafe08cdd75dfa1382c5b6f8009fc5ec3721` (2026-11-02).
- Challenge rule allows CUTLASS/CuTe/CUDA dependencies available in the pinned image; the implementation itself is native CUDA and is not a drop-in because its weight format and shape differ.

## What the paper changes

MonoMoE is a persistent, weight-major MoE decode kernel. It assigns CTAs to expert weight tiles and puts the complete small token tile on the tensor-core N dimension, then fuses routing, quantization, up projection, activation, down projection and reduction with readiness flags. The paper reports its strongest gains for 1–8 tokens versus a token-major Triton grouped-GEMM baseline. The published code is specialized to E=256, K=2048, N=512, M<=8, block-wise FP8 weights/activations, and WGMMA/TMA. These timings are not directly comparable to this challenge's NVFP4 W4A16 kernel.

## Match against our current MoE

The central scheduling idea is already present in our incumbent lineage:

- `kernels/muse_moe_w4a16/submission.py` groups `(token, route)` entries by expert before GEMM jobs are formed (`Params.ent`; `_build_jobs` in `prepare`).
- For small M, one skinny job covers up to eight token rows for an expert, so expert weights are streamed once for the expert's routed rows rather than once per token-route pair.
- The WGS path assigns resident CTAs cost-weighted ranges of expert weight stages; `wgs_wproducer` streams each selected expert's stages through a TMA ring while the consumers compute both projections and signal phase readiness.
- The kernel already fuses routing through the final reduction in one launch, with inter-CTA readiness counters instead of a grid-wide phase barrier.

So directly porting MonoMoE's defining schedule would duplicate our existing expert-major job grouping, persistent weight stream and fused pipeline. Its other ingredients (FP8 activation quantization, block-wise FP8 scales, its N=512 shape, and 384-thread / 128-CTA schedule) cannot be substituted for the challenge's NVFP4 reference semantics without a separate numerical and H200 mapping study. Do not launch a source transplant based on its reported speedup over grouped GEMM.

## Decision and next experiment

Reject a direct MonoMoE port as redundant and format-incompatible. Treat the paper as confirmation that our small-M expert-major path has the right high-level decomposition. Further MoE work should target measured residuals inside that decomposition: for each scored M, capture CTA timelines and HBM transaction efficiency for the current exact candidate; compare them with a same-byte weight-stream skeleton; then change one source of residual (stage-end tails, TMA sector/layout efficiency, or NVFP4 decode issue rate) and screen paired on the official judge image. The existing estimates remain conditional: at M=32 the current kernel is ~130.5 us versus a ~119.9 us same-byte stream comparator, while the comparator is not itself a proven attainable lower bound.

## Current judge candidate

At audit time, check `31f94f4fb0bd43e9` for the KPERM3 + SC6 stacked MoE candidate was running on the official H200 judge. Its source only fixes the local `prepare()` timer initialization; kernel math and dispatch are unchanged from v2. Do not interpret it as a new performance hypothesis; retain its result as a packaging/conformance check for the candidate lineage.
