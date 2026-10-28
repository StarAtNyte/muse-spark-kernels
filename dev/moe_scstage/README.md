# Two-buffer scale staging

Frozen parent: `experiments/moe_wm_v22/submission.py`. Large-token row-split kernel only.

The original three-buffer scale ring occupies 12 KiB per CTA. Two buffers occupy 8 KiB, reducing total dynamic shared memory from 60 to 56 KiB with the selected four-stage weight/activation pipeline. Scale values, weight representation, MMA order, and outputs are unchanged.

A simple modulo replacement would race: scale prefetch is attached to the weight pipeline, which runs three K64 chunks ahead. The new variant issues the next scale group at the start of the current four-chunk group instead. Its asynchronous copy commits alongside that iteration's weight copies and is waited on before the next group is consumed. This keeps two live groups without overwriting a group still in use.

Diagnostic selector 128 selects the new staging only; baseline selector 0 is retained in the same compiled extension. Driver checks six independently generated M128/M256 cases against the reference and for bit-identical output against baseline, then runs alternating cold-L2 CUDA graphs. Full official coverage is required before any promotion. This bundle supports only the large-token configuration and is not a submission.

Evidence: `screen.log`; driver `../modal/drv_moe_scstage.py`.

## Outcome

Completed H200 app `ap-Aq54S493aIRTn5Lz1zUbVu`. All six reference checks passed and all six outputs were bit-identical to baseline. M128 ratios were 0.9828, 1.0131, 0.9707; M256 ratios were 1.0134, 1.0133, 1.0047. Reject as a standalone change: no robust aggregate improvement and only ~1% at M256. No official check or public submission spent. Reducing shared memory alone did not produce the hoped-for throughput gain; register limits and scale-prefetch latency remain relevant.
