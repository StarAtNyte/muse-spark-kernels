# KPERM3 MoE gather batch 6 versus 7

Tests `OKC_GATHB=6` against the recorded scored source
`scored_moe_ours_kperm3_v1/submission.py` (SHA-256
`ac008155520981cd67928bd52bb647efa64096add5acc8edabc779261105792f`),
which uses 7. The driver uses the real token-count dispatch, shared prepared
weights, cold-L2 CUDA graph replays and alternating paired measurements, and
checks both outputs against the task reference. It screens M=32, 128 and 256,
with two seeds each. It is not a full official correctness or performance
check; a promising result requires the official suite before promotion.

The H200 public-image run is bounded by the runner's 300-second subprocess
timeout. Its raw output is `modal.log`.

Run `ap-DhwIjwqfLetsSy9izRgzOU` finished with return code 0. Both variants
passed the task-reference checks for all six cases, and their outputs were
identical in this screen. Median paired baseline/candidate ratios were:

| Tokens | Seed 419 | Seed 7919 |
| --- | ---: | ---: |
| 32 | 1.0069 | 0.9977 |
| 128 | 1.0042 | 1.0047 |
| 256 | 1.0043 | 0.9988 |

These small mixed differences do not establish a reliable improvement. Do not
promote or submit this change on these results. The per-pair median ratio can
differ from the ratio of the separately reported median times. This limited
screen does not rule out a useful setting in a different implementation.
