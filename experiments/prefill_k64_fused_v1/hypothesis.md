# Prefill persistent MoE K64 permutation

This candidate extends the previously validated nonpersistent WGMMA mapping to the scored fused MoE path. Both expert matrices are packed with the same K64 permutation. The gate and up output rows are permuted together so the intermediate already has the layout needed by the second matrix. The gather kernel exchanges normalized BF16 pairs in shared memory to place the first matrix input in that order. All four WGMMA conversion sites select scales by lane subgroup rather than K16 step.

`candidate.py` is experimental and is not the promoted submission. `driver.py` compares the candidate and parent persistent MoE outputs for one synthetic expert routing distribution and three token counts on the public H200 image. A passing result would be a narrow screen; official correctness, special scales, varied routing, race safety, and full-layer performance remain to be established. The shared-memory exchange and extra block barrier may offset any savings from scale reuse.

H200 persistent MoE screen, app `ap-LyGE9DNYmkdcpn2x25W5TS`: passed against the parent at 32, 96, and 753 tokens with relative L2 differences 0.00017254, 0.00016041, and 0.00013829 respectively. Each token was routed to one expert; this does not establish full-layer correctness or performance. Raw output is in `probe.log`.

`bench.py` measures paired cold-L2 full-layer graph replays on the public image using the same SGLang import stub and pinned attention choices as the earlier ablation. It is a screening measurement, not an official judge score.

Paired full-layer timing screen on H200, app `ap-0O3q4b5F90gJRHLW9Rp4i4`, one seed and eight paired samples per shape (median):

| Shape | Parent µs | Candidate µs | Parent / candidate |
| --- | ---: | ---: | ---: |
| batch 1, q 753, sliding, context 80000 | 513.8 | 512.4 | 1.0028 |
| batch 4, q 753, full, context 80000 | 2980.0 | 2982.3 | 0.9992 |
| batch 1, q 16384, full, context 0 | 9144.9 | 9191.0 | 0.9950 |
| batch 1, q 16384, full, context 65536 | 15829.2 | 15663.2 | 1.0106 |

The attention tuner reported a stub-related error and attention choices were pinned. These results are noisy, mixed, and do not demonstrate a reliable overall gain; no submission or promotion. The combined permutation also adds a shared-memory exchange and barrier, so the initial scale selection change may not yield enough savings to cover its cost. Investigate a warp-shuffle gather or reuse in other paths before spending an official check. Full raw output: `bench.log`.
