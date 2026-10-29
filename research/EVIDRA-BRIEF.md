# Compact brief for idea generation (2026-10-29)

Challenge: Muse Spark Open Kernel Challenge, single H200 judge (paired vs SGLang, cold L2, CUDA graphs). Kernel track
closes Nov 4. Score per problem = geomean over loads of stock/ours. Public vs top-50 cutoff: projections 1.5395/1.556,
attention 2.6177/2.6237, MoE 2.3290/2.3736, decoder 2.0926/2.1663, prefill 1.7874/1.7994. Judge noise ~+-1.5% per run.

## MoE (NVFP4 W4A16 experts, 32 local experts of 4096x4096 x3, top-8, EP8) — biggest lever, also ~50% of decoder time
- File: kernels/muse_moe_w4a16/submission.py (vjeux-derived, Apache-2.0). Builds by token count M: TINY (M<=4),
  TINY2 (<=8), MID (<=16), WGS main (>=17). WGS = one persistent CTA/SM, 576 threads: weights producer warp (TMA 36 KB
  stages, ring WGS_NS=3), x producer warp (x ring WGS_NX=3, slots 32 KB), 4 consumer warpgroups doing register
  FP4->bf16 decode (dq4, 9 ops/8 weights) + bf16 scale HMUL2 + RS wgmma m64nNk16 (N = tokens, 8..64).
  Phase A = gate/up (8192x4096 per expert), phase B = down; per-stage doneAs flags; stream-K split units with fixups;
  final pass with grid barrier. Static cost-weighted stage ranges per CTA (wgs_range, OKC_WGSA/B tables).
- Measured (Modal H200): bs32 ~126-133 us for ~425 MB (~3.3 TB/s); pure TMA stream of same bytes 108 us (~3.9 TB/s).
  Skeleton ablation (no decode, no scale, no wgmma, no waits, no epilogue): only 6.5-11% faster -> the scaffolding
  itself (routing prologue, x gather/ring, ring handshakes, final pass) costs ~15 us. Phase tails per CTA are
  systematic (corr 0.92) but rebalancing with fitted cost tables (R^2 0.995) is slower. ncu: 57% issue active,
  DRAM 67%, 96 regs/thread (register-capped; FT_T=32 fails ptxas), smem full (231 KB).
- REJECTED already: L2 prefetch of down weights, split-K single-token kernel, deeper rings (smem), WGS knobs
  (NX, GS, FCI, DFIN, REORD, GATHB), cost-table scaling/shape, FT_T=32, MID/WGS swaps, TINY2 at bs1 (judge equal),
  scale-byte compression (bound +2-3%).

## Other boards
- Attention: full-context cases at random-gather limit (~3.7 TB/s); launch plans tuned for all shapes; flat kernel
  for B=1 slower. Projections: M128/256 at cuBLAS/tuned-plan level; M32 qkv 12.0 us vs 9.9 at M16 (generated kernel
  family supports m<=16 only). Prefill: MoE tensor-pipe saturated, attention FP8 at ~740 TFLOP/s.
- Modal lab vs judge: same hardware, driver 610 vs 580; small launch-sensitive wins did not transfer twice.
