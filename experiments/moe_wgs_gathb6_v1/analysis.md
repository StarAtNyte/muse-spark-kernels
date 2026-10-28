# WGS activation-gather batch 6

Hypothesis: an intermediate gather batch might balance outstanding global
loads against temporary state better than the current batch of 8.
`OKC_GATHB=6` is the only source change; exact source and paired H200 results
are archived here.

All six reference comparisons were bitwise exact. Paired relative gains
(base / candidate) were 0.213%, 0.082%, 0.295%, 0.092%, 0.220%, and 0.401%
for M=32, 128, and 256 over two seeds. Though small, all paired points favor
6. Run one full private check to test the integrated aggregate before deciding
whether to submit. Driver: `dev/modal/drv_moe_wgs_gathb16.py` with `GATHB=6`.

The full private check passed (job `3de84f7864e442fe`, 5/5 tests, 2.32628x,
95% CI [2.29139, 2.36118], 5.781 ms). This is statistically tied with the
2.3263x public best; reject as a promotion and do not submit. Keep GATHB=8.
The judge's exact timing/profile artifacts are in `profiles/`.
