# Single-token pooler geometry confirmation

The preceding `projection_pooler1_geometry_v1` test found a small 0.3–0.7% apparent lead for `(WR,WK,KU,S)=(1,16,8,1)` versus the scored `(2,16,8,1)` geometry, but compared a direct candidate wrapper with full baseline dispatch and had noisy measurements. This follow-up constructs a candidate state with the new losslessly packed stream and calls the identical scored `kernel_fn` for all arms. It includes two independently captured baseline graphs, four input seeds, reference checks and 32 randomized cold-L2 rounds per seed. Per-round ratios and raw samples are in `modal.log`.

The public H200 image and SGLang stub are not the official environment. The diagnostic checks exact packing and task-reference outputs but does not run the exhaustive one-hot verifier or full challenge suite. A useful result would require those checks and official scoring before promotion. The subprocess is capped at 300 seconds.

Run `ap-9xgkWhuyxKS2N0BoCk4T7u` completed successfully; all 12 reference checks passed. For seeds 419, 7919, 22073 and 49151, median paired baseline1/candidate ratios were 0.9942, 1.0008, 0.9902 and 0.9879; A/A ratios were 1.0025, 1.0074, 0.9976 and 0.9902. Individual samples were noisy, and the earlier direct-wrapper lead did not reproduce. Retain the scored baseline; do not submit this geometry on these results.
