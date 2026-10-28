# Narrow row-split output tiles

Frozen parent: experiments/moe_wm_v22/submission.py. Large-token row-split path only. This diagnostic bundle is not a submission.

Halve tiles per warp from four to two, retaining all four token-column fragments and the four-stage asynchronous pipeline. Each block computes half as many output rows; the output grid doubles. This reduces accumulator live state and weight/scale staging per block without reducing the reduction precision or spilling by construction. Increased activation fetches and CTA count are the tradeoff. For NMAX=4 dynamic shared memory drops from 60 to 38 KiB. Narrow launch bounds allow four resident CTAs, versus three for baseline.

Driver tests both stages narrow, gate/up only narrow, and down only narrow. Six fresh M128/M256 inputs, three seeds each, 18 reference checks plus bit-equality checks, 24 alternating cold-L2 replay pairs per variant. CPU-only compilation separately reports register/spill resources. Original and experimental kernels are built in the same module; routing and finish are identical.

Evidence: screen.log, compile.log; ../modal/drv_moe_narrow_rs.py.

CPU ptxas: both narrow kernels use 96 registers, zero spill stores/loads; baseline gate/up 157 and down 160, zero spills. First GPU attempt terminated during Triton import with `ValueError: bad marshal data` before any checks. Retry removes only installed Triton `.pyc` files inside the disposable diagnostic container before import; this is not part of any submitted kernel. Retry evidence: screen_v2.log.

## Outcome

Retry completed app ap-15FhsQ4xSla709lHCZ6Y5n; all 18 checks passed and outputs were bit-identical. All variants lost. Geometric mean paired ratios:
- M128, tiles [2, 2]: 0.9577x
- M128, tiles [2, 4]: 0.9531x
- M128, tiles [4, 2]: 0.9767x
- M256, tiles [2, 2]: 0.8921x
- M256, tiles [2, 4]: 0.9081x
- M256, tiles [4, 2]: 0.9834x

Reject this family. Spill-free register reduction and increased occupancy potential were insufficient: repeated activations, doubled CTAs and smaller output ownership dominate. Do not repeat isolated occupancy tuning without a new data-reuse strategy. No official quota spent. All apps terminal.
