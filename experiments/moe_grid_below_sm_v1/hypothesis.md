# MoE WGS CTA grid below SM count

The WGS kernel's single-resident-CTA-per-SM shared memory footprint and global grid barrier make grids larger than the SM count unsafe. The default is exactly one CTA per SM. Hypothesis: modestly fewer blocks could improve weighted stream-K partitioning, especially the phase-A/B tail, despite leaving some SMs idle; the bandwidth loss might outweigh any gain. This is a bounded schedule experiment, not a claim of headroom or a persistent change.

`driver.py` uses the current main WGS extension and tests grids `SM`, `SM-4`, `SM-12`, `SM-20` at M32, M128 and M256. It computes the full task reference once per shape and checks every grid, holds all graph inputs/outputs alive, and times 20 alternating cold-L2 replays against the default. It never exceeds SM count or WGS fixup-slot capacity, because the kernel includes a grid-wide barrier. One seed and component timing are only a screen; promising cases need independent seeds, full task checks and in-decoder validation.

Public-image app `ap-wuBdgbcfWUJmSLsmQuuPY1`; source SHA-256 `a7fa7eac2b2f6a943ceb977197d992d2fe770aea7a1f65a82ae9d4462127d48a`, driver SHA-256 `1877c280d8d3a7a32b3b05549705c72ba4e94a042e1a8ef49e1b964540096386`. Exact output will be in `screen.log`.
