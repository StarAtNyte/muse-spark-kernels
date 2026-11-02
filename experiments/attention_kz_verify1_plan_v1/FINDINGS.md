# Attention verify-1 sliding: selected VP2 versus fallback split plans

Public judge image, H200, incumbent `kernels/muse_attention_decode/submission.py`, 80k context, batch 1, q_len 8, sliding window, one input seed. The baseline uses the embedded keell0renz module's dispatched `vp2` route. The 24 alternatives call that module's `_launch` fallback directly with splits 8/16/32/64, cooperative finalize on/off, and workspace copies 1/4/8; they do not change VP2 itself. All 25 captured outputs passed the reference check. Twenty-four alternating cold-L2 replays per graph used a 256 MiB read flush.

Baseline median was 9.92 µs; the best fallback was 10.64 µs (`ns64_c1_nc1`, baseline/fallback 0.932×). The next two were 10.72 and 10.91 µs. These numbers include local graph timing overhead and are not official judge timings. The screen rules out these fallback settings on this sample; it does not characterize possible VP2 tuning or other seeds. No source change, official check, or leaderboard submission was made.

Reproduction: `modal run dev/modal/run_judge_image.py --file kernels/muse_attention_decode/submission.py --drv experiments/attention_kz_verify1_plan_v1/driver.py`. Raw output: `modal.log`.
