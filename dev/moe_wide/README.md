# Row/K warp partitioning for small NVFP4 MoE

Design transfer from the public projection implementation: separate row warps
from K-split warps and keep partial reduction inside a CTA. Each warp owns one
16-row weight tile. In gate/up, the two row warps own gate versus up; in down,
they own adjacent output tiles. K is partitioned over 4/8/16 warp groups.
This avoids holding both gate and up weight fragments in every warp.

The 2-row-warp x16-K-warp kernel compiles to exactly 64 registers/thread,
zero spills, 1024 threads and 16 KiB static shared memory. That allows one
32-warp CTA per SM. Its 4/8-K-warp controls compile without spills too.
Weights retain the existing lossless NVFP4 packing; no BF16 weight expansion
is stored. The exact existing BF16 operand conversion, global scales, SiLU and
output rounding are preserved. Floating-point K reduction order may differ.

A GPU work queue uses the current call's route count; fresh normalized inputs,
routing and the original final normalization/weighted sum are timed. Empty
experts do not force a huge dummy grid. No input-derived state is cached.

screen.log tests M1/8 on three seeds, validating every candidate against the
task reference before paired cold-L2 timings. It is a diagnostic module, not a
submission: the full production fallback integration and official checker are
required if a win survives.

Outcome: all 18 task checks passed, but paired geometric means were 0.933x / 0.857x / 0.879x for WK4/8/16. Reject these implementations without official submission; increased per-CTA K parallelism did not improve the complete MoE pipeline. The app is terminal despite a final-log collection timeout.
