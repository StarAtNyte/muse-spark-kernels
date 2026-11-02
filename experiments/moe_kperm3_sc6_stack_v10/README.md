# KPERM3 + SC6 WGS MoE stack, v10 scored confirmation

V9 passed the complete official check (`bd7ee2f9d89247c8`) at 2.36004x,
95% CI 2.32464–2.39544x. The interval overlaps the 2.3820x public incumbent.
V10 adds one prepare-only diagnostic line reporting `prmt`, the validated
`sc6w` predicate, and the exact main compile defines. Timed CUDA code and
routing are unchanged from checked v9. Scored run `1fb6b4f749ec44bb` confirmed
`prmt=True`, `sc6w=True`, `sc6s=False`, and compiled the main WGS route with
`-DOKC_SC6W=1`. It passed and scored **2.35842x** (95% CI
2.32304–2.39379x), below the 2.38203x incumbent point estimate with
overlapping intervals. Keep the incumbent; SC6W is inconclusive rather than a
demonstrated gain. The judge log and per-load profile are archived in
`profiles/`.
