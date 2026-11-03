# General projection bucket routes

Builds on the simplified V6 candidate (official check `05a1587da81d4434` passed 9/9 at 1.54086×, 95% CI 1.51775–1.56397×). V7 adds a small set of token-range routes to already-prepared FP32 kernels instead of the inherited exhaustive off-score autotuner. A paired H200 screen passed both implementations on 60 nearby cases and measured meaningful improvements for several ranges; see `../projection_bucket_nearby_v1/FINDINGS.md`.

Private full check `a6d1ceab0bc24fb5` is pending. Other source/precision paths remain under audit; no public submission has yet been made.

Private check `a6d1ceab0bc24fb5` passed 9/9 at 1.55635× (95% CI 1.53301–1.57970×). The exact source SHA-256 `d785aa19a58a8fe5be5d34406496f649340273ffc8086ed46051dba46bacd483` was copied to the scored archive and submitted as job `1f5bfe33160a4deb`. The scored result is pending. The 60-case nearby screen does not cover every possible input, and final eligibility remains subject to organizer review.
