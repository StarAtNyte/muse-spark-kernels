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
