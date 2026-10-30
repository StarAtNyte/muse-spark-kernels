# Prefill persistent MoE consumer-first layout

Tested the existing `MOE_CONS_FIRST=1` alternate layout: 320 threads with consumer warpgroups first and no dynamic register redistribution, versus the scored parent's 384-thread layout with producer registers released to the consumers. The exact parent is published 1.80923× SHA-256 `f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`; candidate SHA-256 `a35bdf71b395eaefb677d165813e72b78cadc869a6335b41c1b9504ace4124d3` differs by that single macro.

Public-image H200 app `ap-ep4PPIV0OeZVBgoWJ7nkNr` completed under a 300-second subprocess cap. Both variants shared prepared state and tuned plans. All three full-layer reference screens passed. Paired 16-replay cold-L2 parent/candidate medians were 511.7/548.6 µs (turn-1 sliding, 0.9328×), 3076.8/3184.6 µs (turn-4 full, 0.9662×), and 8889.7/9468.6 µs (first-turn sliding, 0.9389×). One seed per case; these are screens, not official scores. The SGLang import stub caused attention tuning to fail, so attention was pinned. The driver docstring is inherited from an earlier ablation; actual runtime mutation changes only `MOE_CONS_FIRST`. See `screen.log`.

Decision: reject; do not promote or submit. Reduced consumer registers or the different layout may explain the regression, but no profiling here distinguishes those causes.
