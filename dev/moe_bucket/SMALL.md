# Expert-count specialization for M17..32

This targets the production NT4/NW4/RING1 algorithm, unlike the large-batch
row-split experiments. Most experts have <=8 tokens but share the worst-case
register reservation of the NCH4 path.

A first CPU-only design kept the exact NT4 tiling. Its sparse kernels needed
108–110 registers, which still allow only four blocks per SM: no occupancy
improvement to justify extra launches. It was not GPU-tested.

The revised sparse kernel uses NT2/NCH1, four K-split warps and a six-stage
weight ring. It compiles at 80–83 registers with zero spills, permitting six
blocks per SM. Middle kernels retain NT4/NTB2/NCH2 and compile at 107–109
registers. Dense kernels use a three-block launch bound and compile at 165–167
registers without spills. The previous four-block dense bound spilled 60–68
bytes. All use the same FP4 values, BF16 operands, FP32 accumulators and final
rounding. A register-prefetch sparse variant is a separate control.

`small_compile_v2.log` is CPU-only triage. `small_screen.log` checks six random
batch-32 routings, nearby batches 17/24, and a skewed batch-32 routing. Each
candidate must match the independent task reference and production output
bit-for-bit. Alternating cold-L2 graph timings include routing, all bucket
streams, joins and finish. A monolithic persistent-scheduling control separates
benefits of the work queue from benefits of specialization.

The diagnostic bundle is not a valid submission: unused fallback dispatches
are pruned. A winner needs full standalone integration and official coverage.

## Outcome

`small_screen.log` passed all 45 independent task checks with bit-identical
outputs. Persistent scheduling was 0.966×, async low/high caps 0.857/0.871×,
and direct-prefetch low/high caps 0.989/0.979× across the nine comparisons.
No variant consistently wins on the scored batch-32 cases. Reject without an
official check. Increased theoretical occupancy did not improve end-to-end
throughput; smaller output tiles and separate bucket scheduling remain costly.
A single nearby M17 win is insufficient to promote a batch-32 optimization.
