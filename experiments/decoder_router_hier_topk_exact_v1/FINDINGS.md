# Exact decoder router hierarchical top-k

This isolates a port inspired by the recent [vLLM Muse CUDA routing PR #64553](https://github.com/vllm-project/vllm/pull/64553), without stacking the unconfirmed WGS-prefix change. The incumbent's route key packs a monotonic float score plus reversed expert ID, giving a unique stable ordering. Every global top-8 must be among the 8 best in its 64-expert partition; four groups yield 32 candidates, then exact top-8. The four-way layout is intended to map to `_sl_tile_route`'s four-warps launch.

Only the MODE=0 top-k expression in `_sl_tile_route` changes. No additional kernel launch. The PR's `redux.sync.max.f32` specialization is excluded because it targets Blackwell SM100, while the challenge runs H200 SM90.

Paired H200 screen covered full and sliding layers, batches 1/16/32, q_len 1/8, with three input seeds and four alternating replays per pair. All 36 candidate and parent correctness checks passed. Candidate/parent geometric mean: **1.00171×** overall, **1.00355×** sliding, **0.99987×** full. The effect is small; B16 sliding decode-1 regresses on one seed and most full cases are flat. Raw timings: `multiseed_modal.log`. The first one-seed/eight-replay screen was +0.264%; its raw record is `screen1_modal.log`. Existing incumbent profiles show the fused route at about 1.8–2.1 us, so this is a bounded, secondary optimization.


Official run `d1faeb5b0a5c496d` passed correctness but scored **2.21156×** (95% CI **2.17839–2.24473×**), below the **2.26032×** incumbent. Reject it; the published decoder score and rank are unchanged. The local paired gain did not survive the official draw. This isolates the router hierarchy as insufficient and not worth stacking further without a larger whole-layer lever.
