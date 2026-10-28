# M32 direct activation operands

Screen only the active M32 operatorstack route with XS=False, so `_om__x_tile` loads and converts activations directly into the MMA operand instead of staging through two shared-memory layouts. Keep current staging for every other shape. This targets the measured M32 MoE bottleneck (`_om__w2`, about 94% of route time) and removes shared-memory stores/barriers where repeated activation loads may already hit L2.

Candidate is prepared only. The Modal Gluon image did not reproduce official M32 outputs even for the current source, so use the official checker after quota reset; do not infer performance from those Modal runs.


Official H200 result (2026-10-28): correctness passed, but aggregate speedup was 1.9005x (95% CI 1.8720–1.9290), below the staged current best at 1.9698x (CI 1.9402–1.9993). Decode-32 regressed from 1.4517x to 1.1931x; direct activation operands are too costly on this latency-sensitive shape. Rejected; do not promote. Full check: `experiments/moe_m32_directx_h200check2/status.json`.
