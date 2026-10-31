# Short-prefill MoE deeper small-tile pipeline

The scored prefill source uses `MOE_SMALL_WAIT=1`: small 32-row tiles wait
with at most one WGMMA group outstanding. It contains an alternative four-A-
fragment-buffer path for deeper in-flight groups. This candidate sets the
existing knob to 2, potentially overlapping more conversion with WGMMA; extra
register pressure or serialization can make it slower. The alternative changes
slot-release timing and must not be assumed safe merely because it compiles.

Parent SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`;
candidate SHA-256
`b6ab4d0d2c23102c29b272a9dc16aba66dfea30ad897fbc5dfeee7017f1b38d4`.
The bounded public-image H200 driver shares prepared weights and plans, checks
both outputs against the reference, and compares 16 alternating cold-L2
full-layer replays on three shapes. The local attention tuner is unavailable
under the SGLang stub, so attention is pinned. Raw output and any failure are
in `modal.log`. A promising result requires race/synchronization validation
and the full official check before promotion.

Run `ap-TmVMwgi4GoxfAF9aWfLYnw` completed with return code 0 and both
variants passed the three reference screens. Parent/candidate full-layer
medians were 514.0/553.9 µs for turn-1 sliding, 3021.7/3110.7 µs for turn-4
full, and 9211.7/9683.6 µs for first-turn sliding. All three regress; retain
`MOE_SMALL_WAIT=1` and do not submit. This one-seed screen does not establish
why the alternate path is slower or prove its race freedom. Raw output is
`modal.log`.
