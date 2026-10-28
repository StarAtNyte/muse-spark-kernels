# Attention K/V 64-byte L2 prefetch hint

The candidate adds the PTX `L2::64B` prefetch hint to the fast-path K/V `cp.async` loads only for flat q_len=8. It changes neither the copied bytes nor the attention computation. Hypothesis: a smaller prefetch hint could help the streaming, low-L2-hit workload. The baseline does not explicitly specify a prefetch size; this experiment does not establish what size the hardware otherwise uses.

Paired H200 CUDA-graph tests on full 80k agentic/coherent page-1 inputs used batch 16 and 32, seeds 539, 1427 and 7919, 20 alternating repetitions, and a 256 MiB cache flush between samples. All six official reference checks passed. Median paired baseline/candidate ratios were 0.99828, 1.00380, 1.00290 at batch 16 and 1.00141, 0.99743, 1.00033 at batch 32. The variation provides no repeatable improvement; retain the baseline and do not submit this component-only candidate.

Candidate SHA-256 `b8801bde894de5fdb9cb9ae84c1d62b3442c60da181060234c3fb6a1bdbe07cb`; baseline SHA-256 `ffc1c6bb4e95aa30390858eab4c43dbd592b2884b4fe3e54412282eaafdbebaa`. Modal app `ap-FpbsSU0ExyCwlg2I5Abvt7`; raw output `modal.log`. Command: `modal run dev/modal/run_h200.py --problem muse_attention_decode --file experiments/attention_kv_prefetch64_v1/submission.py --drv dev/modal/drv_attention_infraax_pair.py --cmd 'timeout 900s python /w/run/drv.py'`.
