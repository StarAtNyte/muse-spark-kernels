# Prefill MoE sparse 32-row tile MMA screen

The scored prefill kernel pads small expert remainders to 32 rows. Its source
contains an alternate `MOE_MMA32` path that uses warp-level MMA and skips
8-token groups beyond the real routed rows. The expected benefit is less
arithmetic on sparse expert tiles; the cost is additional fragment loading,
different tensor-core utilization and a different reduction order. No speedup
or correctness is assumed.

This candidate changes only `MOE_MMA32` from 0 to 1 in the scored source
(parent SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`,
candidate SHA-256
`79802862adbf5f3754b072b26b866c9d82dd659281c2253566dfcb2d96289899`).
The bounded public-image H200 driver shares prepared state and plans, checks
each full-layer output against the reference and compares 16 alternating
cold-L2 replays on three shapes. The local SGLang stub prevents attention
tuning, so attention is pinned. A promising result needs independent multi-seed
validation, race/synchronization checks and the official suite before
promotion. Raw output and any errors are retained in `modal.log`.

The launched driver and the checked-in driver differ only in corrected
docstring/comment text inherited from an unrelated invalid-output ablation;
the runtime code is the same.
