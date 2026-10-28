# Producer page-index shuffle experiment

The candidate replaces eight duplicate page-index load lanes per row with a single lane load and an eight-lane broadcast in flat q_len=8 attention. The hypothesis was that fewer active load lanes might improve producer efficiency. It preserves the K/V copy layout and leaves the other template paths unchanged.

Paired H200 CUDA-graph tests used the official reference, full 80k agentic/coherent page-1 contexts, batches 16 and 32, seeds 539, 1427 and 7919, 20 alternating repetitions per case, and a 256 MiB flush between samples. All six candidate reference checks passed. Median paired baseline/candidate ratios were 0.9820, 0.9830, 0.9887 at batch 16 and 0.9594, 0.9596, 0.9610 at batch 32. Thus the candidate was consistently slower; retain the current kernel and do not submit this candidate. This is a component screen, not a complete official challenge check.

Candidate SHA-256 `5e5cf3a5fdec6dbdba1866e6d2c43b5c93dafc7ae106a5f306bcb91668fba203`; baseline SHA-256 `ffc1c6bb4e95aa30390858eab4c43dbd592b2884b4fe3e54412282eaafdbebaa`. Modal app `ap-rTngXtGUeheSDJ1gMUGODL`; exact output in `modal.log`. Command: `modal run dev/modal/run_h200.py --problem muse_attention_decode --file experiments/attention_page_shuffle_v1/submission.py --drv dev/modal/drv_attention_infraax_pair.py --cmd 'timeout 900s python /w/run/drv.py'`.
