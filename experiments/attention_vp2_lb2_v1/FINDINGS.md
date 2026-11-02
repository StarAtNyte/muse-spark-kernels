# Attention VP2 launch-bound screen

Candidate SHA-256 `691eaa9134ab68a874b73269aa5e5677c748d7cb4234a764c66949a0c8396aef` changes only the selected VP2 kernel's CUDA launch bound from `(256,1)` to `(256,2)` inside the embedded keell0renz source. This requests a lower register budget compatible with two blocks per SM; actual occupancy and register spills were not profiled.

On the public judge image H200, three batch-1 sliding verify inputs passed the challenge reference check for both implementations. Alternating cold-L2 paired measurements gave parent/candidate ratios 1.006, 1.008 and 1.011. These are selected-shape results, not a whole-attention score. Official private check `7f9f41ad5a5d447b` passed 16/16 at 2.62193× (95% CI 2.58260–2.66126), overlapping the published 2.6328×. The exact checked source was submitted as scored job `6f2b2edf215d4ea3`; its result is pending. Exact source, driver and raw output are archived here.
