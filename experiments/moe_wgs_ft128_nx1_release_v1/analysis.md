# 128-token WGS one-slot release fix

The previous 128-token candidate hung because the producer needed the single x slot for the next stage while the consumer normally released the preceding slot only after starting that next stage. For `WGS_NX == 1`, this candidate waits for all WGMMA groups at the end of each stage and immediately releases the slot; the final release is guarded against an already-released slot. The two-or-more-slot pipeline is unchanged.

CPU-built baseline and candidate were transferred to H200. All six reference checks passed and outputs were bitwise identical. Median paired baseline/candidate ratios for seeds 419 and 7919 were 0.8450 and 0.8415 at M32, 0.8067 and 0.8061 at M128, and 0.7998 and 0.7847 at M256: the candidate was 18–27% slower. The enforced stage-level wait plausibly loses overlap, but this experiment alone does not prove the cause. Do not submit; retain the scored baseline.

Candidate SHA-256 `f7cbd914f0375179547a43a89c79e3619112cb041d904808342742b141eb6ec8`; baseline SHA-256 `a7fa7eac2b2f6a943ceb977197d992d2fe770aea7a1f65a82ae9d4462127d48a`. Modal app `ap-C3iE91DYIF80ZzKnVTL0Vb`; exact output `screen.log`. Command: `modal run dev/modal/moe_ft128_prebuilt.py --base kernels/muse_moe_w4a16/submission.py --candidate experiments/moe_wgs_ft128_nx1_release_v1/submission.py`.
