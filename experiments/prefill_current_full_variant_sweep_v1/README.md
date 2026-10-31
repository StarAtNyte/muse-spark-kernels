# Current prefill short full-attention variant sweep

Compares the current scored source (SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`)
variant 43 against variants 0, 1, 8, 9, 16 and 24, with each variant's normal
split heuristic. It covers synthetic B=1 and B=4, query length 753, context
80000, page size 1, and seeds 539 and 1427. The driver retains graph outputs,
checks finite values and relative L2 against variant 43, and alternates
cold-L2 CUDA graph timings. The subprocess uses the public challenge image
and is bounded to 300 seconds.

This is a component screen against the incumbent output, not an independent
reference check. A candidate needs separate complete-layer paired timing and
the official correctness suite before promotion. Results and any failures are
recorded in `modal.log`.

Public-image H200 run `ap-vT6YogxVNWBh2V31MjQ6MT` completed with return code
0. All candidates passed the driver's relative-output screen. Every alternative
was slower than variant 43 on both seeds in both batch sizes. The best
alternative was variant 8: parent/candidate ratios were 0.9263 and 0.9274 at
B=1, and 0.9286 and 0.9286 at B=4. Variants 0, 1 and 16 used 11 splits at
B=1 versus the incumbent's two; all variants used one split at B=4. The test
therefore compares the current complete variant-plus-heuristic choices, not
the isolated effect of each kernel configuration with split count held fixed.

Retain variant 43 and do not submit any of these alternatives. These synthetic
component checks are not full official correctness validation and do not rule
out other kernel designs or launch plans.
