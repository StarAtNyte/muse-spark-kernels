# projection (w3) log
- 2026-10-30 attempt 1 (timed out): screened vjeux-route expansion on Modal; m=32 qkv/unpooler sk2 routes regressed and were dropped (candidate_v1 -> v2).
- Official check da14d2b1148a4849 (projection_vjeux_router256_m1_v1): 1.5258x CI [1.503,1.549] < 1.5395 public. Real per-shape wins on unpooler m1 (+8%) and router m256 (+10%) but only +0.3% modelled overall. Not submitted.
- Sensitivity model: putting M=128/256 qkv/o/pooler/unpooler at 1.15x SOL alone gives 1.659x (> #1 1.6498x); verify16 1.297->1.584, verify32 1.227->1.575. Every other route tweak is under the noise floor.
- Details: findings/w3-work/check_da14d2b1148a4849.md, HBM_FLOORS.md, sensitivity_table.json
- 2026-10-30 w3.2 router rk M=128/256: NEGATIVE. rk already split-K + shared x tile. Load-only phase ≈4.7/6.2 us vs rk ≈5.7/7.0 us (cold 10 MB read ≈2.1 TB/s; SOL floors unreachable). 24 tile/S configs, packed-W bulk loads and int8-lo (W 8→6 MB) all fail to beat rk by >1 us (best real ≈0, int8-lo upper bound +0.5 us). No official check, nothing submitted. findings/w3.2.md
