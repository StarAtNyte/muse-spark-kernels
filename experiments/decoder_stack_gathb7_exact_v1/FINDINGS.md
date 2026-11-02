# Exact-baseline decoder GATHB=7 screen

## Decision
Reject for promotion. Candidate changes only the embedded WGS MoE gather batch from 8 to 7, matching the standalone GATHB=7 MoE winner. Exact parent source is the published decoder (`9e7c8b73b0471d04f4f0cf451bea6ad372d7356e3103e7f57d7253df234aebb2`); candidate SHA-256 is `cc569a79783b6e88ae55645a8153aff77aae872a99ae1afd79d2b1094b2c4b43`.

The judge-image Modal H200 paired screen ran six sliding decode/verify shapes at batches 1, 16 and 32, with two input seeds per shape. Both implementations passed the decoder reference check for every case. Geometric aggregation of the twelve paired speed ratios was approximately **0.9993x**. Per-shape geometric ratios: batch 1 decode 0.9901x, verify 1.0066x; batch 16 decode 0.9966x, verify 1.0033x; batch 32 decode 1.0017x, verify 0.9976x. This is neutral within measurement noise and does not justify spending an official check or leaderboard submission.

Raw timings and Modal invocation are in `modal_judgeimage.log`. Two failed setup attempts are also preserved: generic and SGLang Modal images lacked `tvm_ffi`; the judge image resolved this dependency.

## Interpretation
The standalone MoE gather batch winner does not transfer into the exact integrated decoder baseline. Integrated MoE/attention overlap and batch-specific hot paths wash out the tiny component change. Retain the incumbent decoder. Continue with changes that alter the decoder critical path materially, especially its sliding MoE phase scheduling, or stack a whole-layer-tested improvement from attention/projections.
