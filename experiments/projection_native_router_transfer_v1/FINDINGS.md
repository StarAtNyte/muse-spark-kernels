# FP32 single-token router transfer

Candidate adapts the original `router_one` native CUDA experiment from iyerramji/open-kernel-challenge `v31_router_native.py`, upstream revision `1d57ff7749433d0d18b294744e607cd82e33057f` (Apache-2.0; `LICENSE`, `UPSTREAM_NOTICE`, and the upstream source retained). It replaces only the one-token router route on our current projection stack. The upstream report measured its own baseline and does not establish a gain against ours.

The first H200 attempt (`modal.log`, app `ap-IHsNiFK25wm0G9jr7tyXJc`) hit the 300-second subprocess cap during duplicate parent preparation; it produced no timing or correctness results. The following shared-state attempt was interrupted with its Modal app `ap-H8uIZzIv1UK7yu9WQG9HMr` no longer listed and produced no results (`modal_shared.log`). The retry uses one prepared parent state and identical weights for the paired comparison; see `driver.py` and `modal_shared_retry.log`.

The retry completed on H200, app `ap-0AG8BIlDSgKsNQytVGkgV7`. Both outputs passed `reference.check_implementation` for each of two seeds. Paired cold-L2 graph medians (20 alternating measurements per seed):

| Seed | Parent µs | Candidate µs | Median paired parent / candidate |
| ---: | ---: | ---: | ---: |
| 419 | 9.152 | 11.248 | 0.8183 |
| 7919 | 9.104 | 11.296 | 0.8113 |

The transferred kernel was roughly 23–24% slower by the ratio of the reported medians. Reject this route and retain the incumbent; no official check or submission was warranted. These results apply to the tested M=1 router shape and seeds. The public-image timing includes graph replay effects and is not an official leaderboard score.
