# Evaluation of Fused Prefill K64 MoE Path

## Primary Evidence

- Commits:
  - `ce3e4e1`: Validate gated K64 prefill intermediate on H200
  - `809f407`: Prototype K64 scale reuse in fused prefill MoE
  - `493c634`: Record fused prefill K64 correctness and timing screens
- Artifacts:
  - `experiments/prefill_k64_fused_v1/probe.log`
  - `experiments/prefill_k64_fused_v1/bench.log`
  - `experiments/prefill_k64_fused_v1/hypothesis.md`

## Experimental Findings

1. **Correctness screen on Modal H200 (`ap-LyGE9DNYmkdcpn2x25W5TS`):**
   - Candidate persistent MoE kernel matched parent kernel output on single synthetic expert routing:
     - 32 tokens: relative L2 = 1.725e-4
     - 96 tokens: relative L2 = 1.604e-4
     - 753 tokens: relative L2 = 1.383e-4

2. **Full-Layer Paired Cold-L2 Timing Screen on H200 (`ap-0O3q4b5F90gJRHLW9Rp4i4`):**
   - Four representative shapes measured across 8 paired swaps per shape:
     - `batch 1, q 753, sliding, ctx 80000`: Base 513.8 µs, Cand 512.4 µs (1.0028x)
     - `batch 4, q 753, full, ctx 80000`: Base 2980.0 µs, Cand 2982.3 µs (0.9992x)
     - `batch 1, q 16384, full, ctx 0`: Base 9144.9 µs, Cand 9191.0 µs (0.9950x)
     - `batch 1, q 16384, full, ctx 65536`: Base 15829.2 µs, Cand 15663.2 µs (1.0106x)
   - Geometric mean speedup: **1.0019x** (+0.19%, within standard ±1.5% measurement uncertainty).
   - 2 of 4 cases experienced net regression.

3. **Bottleneck Analysis:**
   - In `experiments/prefill_scale_reuse_design_v1`, the theoretical saving was hoisted scale selection across the 4 inner steps of each 64-element K stage.
   - However, the permuted activation layout requires cross-thread exchange and an additional block barrier in shared memory during gather.
   - The extra shared memory data movement and synchronisation overhead offset the instruction/register savings from scale reuse.

## Rigorous Decision

In accordance with the directive (*"finish and officially check the fused prefill K64 MoE path if its screens show a win"*):
- The screens do **not** show a win (geometric mean 1.0019x, 2 regressions).
- Spending an official check (out of 19 daily quota) on an unpromoted candidate that failed its screen would violate the experiment standard (`research/ENGINEERING.md`).
- **Verdict: Reject for official check/submission; keep archived.**
