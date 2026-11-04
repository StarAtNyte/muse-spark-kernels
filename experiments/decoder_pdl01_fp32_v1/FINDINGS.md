# Decoder SK2 PDL pre-wait depth 0/1 vs 2

On exact readable FP32 decoder source `experiments/scored_decoder_fp32_ybuf_v4/submission.py`, one shared preparation and graph captures compared changing only `BFP_PDL_N` from 2 to 1 or 0. The H200 judge-image screen used 12 alternating cold-L2 graph replays per arm, two fresh seeds each for batch 1, 16, 32 and q_len 1, 8 on sliding contexts. All 36 candidate/reference checks passed.

Across the 12 same-input rows, the geometric parent/candidate ratio was 0.9954× for depth 1 and 0.9829× for depth 0 (calculated from printed four-decimal ratios); depth 1 lost 0.7–2.2% on small decode/verify cases and was otherwise close, while depth 0 lost approximately 2–5% on decode cases. Reject both as global settings. Full driver and raw log are in this directory. These results do not rule out per-stage or role-specific changes, or prove whole-board score.
