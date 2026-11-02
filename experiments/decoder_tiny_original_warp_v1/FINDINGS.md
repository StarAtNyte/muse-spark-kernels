# Decoder tiny MoE original warp order

Candidate SHA-256 `aa21faa03140a1fabdf58381c2c0606e8024e17d54455fc603fc47e557196785` removes the `moe_helper0` source override for tiny and tiny2 MoE builds (M<=8), leaving weight packing unchanged. M=9–16 retains the helper source.

Official private check `42157e3923564abf` passed 13/13 at **2.25398×**, 95% CI **[2.22017, 2.28779]**, versus the published decoder score 2.26032×. Separate-run intervals overlap.

The CUDA 13 H200 paired full-layer screen in `modal.log` passed all 12 reference checks. Parent/candidate timing ratios were 0.9996–1.0089 for decode-1 sliding, 1.0049–1.0090 for decode-1 full, and 1.0035–1.0124 for verify-1 sliding. The unchanged decode-16 control varied 0.9949–1.0042. The second preparation logged `bake failed: rq_e` after the tuning budget expired, so the screen does not establish identical preparation or judge parity. No scored quota remained at measurement time. Keep this candidate archived for a fresh-process or judge-image paired follow-up; no improvement is claimed and the public kernel remains unchanged.
