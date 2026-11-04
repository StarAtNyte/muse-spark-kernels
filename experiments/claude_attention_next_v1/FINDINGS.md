# claude_attention_next_v1 findings

## Experiment
Baseline: `experiments/attention_readable_fp32_path_v1/submission.py` (sha256 ea65a78f...). Candidate: only change is the q_len-8 flat-partition virtual boundary padding SEGK 600 -> 900 (all 3 occurrences of `(L == 8) ? 600 : 400`). Hypothesis: a request-boundary crossing costs ~7 us at q_len 8, so more padding keeps CTAs from straddling boundaries. q_len 1 padding (400) untouched; 600 was previously neutral there (research/ENGINEERING.md). The q_len 8 value was not found to be swept in the notes.

Source, FP32 QK/PV/partials, and output contract are unchanged.

## Method
Judge image (lmsysorg/sglang:dev-muse-spark-open), H200, Modal timeout 300 s (run ~ under that, RETURNCODE 0). Driver `driver.py`: 3 seeds x 4 scored shapes (verify 32 full, verify 16 full, verify 32 sliding, decode 32 full; batch ctx 80000 tp8 page 1 agentic), reference check on both arms, 20 alternating-order cold-L2 CUDA graph replays, median. Full log `modal.log`; hashes `hashes.txt`.

## Results (cand_gain = ours/cand, >1 candidate faster)
| case | seed101 | seed2027 | seed4093 |
|---|---|---|---|
| verify 32 full | 0.9956 | 0.9971 | 1.0014 |
| verify 16 full | 1.0021 | 0.9975 | 1.0027 |
| verify 32 sliding | 0.9983 | 0.9991 | 1.0051 |
| decode 32 full | 1.0010 | 0.9996 | 1.0018 |

All 24 reference checks passed. Gains range from 0.9956 to 1.0051 and are sign-mixed; these results do not establish a reproducible speedup. Official judge intervals do not define a universal noise floor for this paired experiment.

## Verdict
Reject this 600-to-900 change on the available evidence. It does not establish whether other partition strategies can help or whether boundary crossings limit performance. The reference's ~127 us figure is an idealized model, not a proven universal floor. No submission was made. Only four shape configurations were tested; branch execution was not instrumented.
