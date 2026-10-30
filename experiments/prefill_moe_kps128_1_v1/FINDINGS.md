# Prefill MoE KPS128=1 screen

Tested `MOE_KPS128=1` instead of 2 for 128-row expert tiles, to trade smaller per-stage work for more synchronization. Parent is published 1.80923× SHA-256 `f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`; candidate SHA-256 `6ebb844a5d8e0025a5bb07c57557b8db134b566799ad11da77ee2e260f359f2c`. The driver loads the parent twice and changes only the macro in the candidate module, sharing prepared state and plans; its opening docstring is inherited from an earlier ablation.

H200 public-image app `ap-IyzD655XjdOEEKuArgajIl` failed the first full-layer reference check at batch 1, q_len 753, sliding, context 80000, seed 9400: baseline passed, candidate residual update contained NaN or Inf. Subprocess return code 1; no performance readings. Attention tuning hit the known stub error and was pinned. This isolated failure does not identify the exact bug in the alternate K-stage path. Reject and do not submit; full output in `screen.log`.
