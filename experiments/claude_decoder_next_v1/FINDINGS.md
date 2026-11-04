# claude_decoder_next_v1: nanosleep backoff in the MoE B4FIN final-pass readiness poll

## Result
Neutral. Geomean parent/candidate = 1.0023x over 6 shapes (per shape 0.9939-1.0061x, i.e. within judge-image run noise of about +-0.5%). Recommendation: reject; do not submit or port.

## Baseline
Parent = `experiments/scored_decoder_fp32_ybuf_v4/submission.py` (also `decoder_readable_fp32_ybuf_v4`), SHA-256 `616187c9dac236851cc4b76bb65cd1cf9c85490db96f241a2fb1fa3d0fcff2cc`: the readable, FP32-storage whole decoder (private check 2.24462x, scored job 529df1f504434040 at 2.23863x). `research/best.json` still names `decode_keell0renz_compare_v1` (82c098ae...), which is in the encoded-source audit and so was not used.

## Why this target
Judge profile of the parent (job 529df1f504434040, decode 32 sliding, the weakest load at 1.80x): `moe_fused` 73.5 us of 151 us, with projections 57 us, attention 9.4 us and boundary kernels about 11 us. Projection boundary, KV prefetch, PDL depth, fused gle and MoE ring/cost-table knobs are already rejected in NOTES.md / experiments, so I picked a MoE knob with no prior sweep.

## Change (one)
For 16 < T <= 32 (the `b4` build: `MOE_DEFINES + -DOKC_B4FIN=1`), the per-token final pass polls each expert's completion counter (`ld_acquire(cp) < target`) in a tight loop on 8 threads per CTA. Candidate adds `-DOKC_MFS=48`, an existing macro that inserts `__nanosleep(48)` between polls, to reduce L2 polling traffic competing with weight streaming. Source change: none beyond the extra define. All other builds, partial sums and the FP32 `ybuf` storage are untouched, so FP32 partial-sum rules are unaffected. Math is identical (waiting longer cannot change values).

## Method
`driver.py`, one Modal invocation (`dev/modal/_claude_decoder_next_v1_run.py`, H200, lmsysorg/sglang:dev-muse-spark-open, function timeout 300 s). One `prepare`; the candidate extension was built in a thread concurrently with the rest of prepare; for each case the parent and candidate graphs were captured by swapping `state['ma_ext']['b4']`, both checked with `reference.check_implementation`, then 20 ABBA-alternated cold-L2 (256 MB flush) graph replays each, median compared. Full log: `modal1.log`.

| case | T | parent us | cand us | parent/cand | both pass ref | bitwise equal outputs |
|---|---|---|---|---|---|---|
| b32 ctx80000 sliding agentic s8001 | 32 | 149.87 | 149.66 | 1.0014 | yes | no |
| b32 ctx80000 sliding agentic s8002 | 32 | 153.87 | 154.82 | 0.9939 | yes | yes |
| b32 ctx80000 full agentic s8003 | 32 | 324.38 | 323.31 | 1.0033 | yes | yes |
| b4 ctx6000 sliding q_len8 s8004 | 32 | 191.87 | 191.07 | 1.0042 | yes | no |
| b17 ctx20000 full mixed page1 s8005 | 17 | 191.49 | 190.32 | 1.0061 | yes | yes |
| b24 ctx30000 sliding agentic page1 s8006 | 24 | 165.50 | 164.69 | 1.0050 | yes | no |

Geomean 1.0023x.

## Limitations
- Only the b4 build (16 < T <= 32) was changed; T = 128/256 (main build, also containing the poll) were not measured.
- One MFS value (48 ns), one node, one seed per shape (two for decode-32 sliding). No variance estimate beyond the spread across shapes. This does not establish whole-board effect, but 1.002x is far below any detectable gain.
- Some paired outputs are not bitwise equal. I did not separate run-to-run routing nondeterminism (allowed by RULES.md within tolerance) from an effect of the define; both variants pass the reference. The 13 TASK.md tests were not run for the candidate; only the 6 shapes above.
- Only 6 of the 13 reference tests' shapes types were exercised, since the 300 s Modal cap (prepare alone took about 190 s) left no room.

## Interpretation
The result does not establish how often the poll waits or whether polling limits the MoE: neither was measured. One possible explanation is that the poll is usually satisfied promptly, but that remains a hypothesis. This single backoff setting produced no convincing end-to-end gain; other settings or structural MoE changes require separate evidence.
