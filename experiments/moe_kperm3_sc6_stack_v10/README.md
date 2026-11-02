# KPERM3 + SC6 WGS MoE stack, v10 scored confirmation

V9 passed the complete official check (`bd7ee2f9d89247c8`) at 2.36004x,
95% CI 2.32464–2.39544x. The interval overlaps the 2.3820x public incumbent.
V10 adds one prepare-only diagnostic line reporting `prmt`, the validated
`sc6w` predicate, and the exact main compile defines. Timed CUDA code and
routing are unchanged from checked v9. This scored run will test whether the
lossless SC6 WGS route actually activates on judge weights and sample its
leaderboard score; a score below the incumbent does not replace it.
