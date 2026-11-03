# NOTES

Measured H200 facts, negative results and credits for third-party Apache-2.0 kernels. History lives in `research/ENGINEERING.md`, `research/H200-LOWER-BOUNDS.md`, `research/FLOORS-2026-10-29.md` and the `research/*AUDIT*.md` files.

## Credits

## Facts and negative results
- 2026-11-03 standings: decode #166 (2.260x vs #1 2.490x), MoE #239 (2.382x vs 2.764x), attention #155 (2.649x vs 3.339x), proj #61 (1.592x vs 1.694x), prefill #49.
- MoE per-shape (job 98be7cf25e0c494f): bs1 25.0us (floor 11.4), bs16 95.5 (52.5), bs32 122.6 (70.8), bs8 67.0 (37.3), bs128 174.6 (101.3), bs256 211.9 (108.9).
  Pure contiguous stream of bs1's 56.6 MB is ~16 us net (h200_read_methods_v1) -> bs1 has ~9 us headroom; timeline (moe_timeline_v1) shows a 5.7 us phase-A CTA spread + 2.3 us routing.
- Decoder decode-1 (job 529df1f504434040): MoE 30.1 us of 78 us; 4 sk2 projection kernels 38 us; layer 65.9 us vs floor 40.
- Judge weight codes are uniform random over 16 values (nvfp4.random_weights): no entropy to compress; scales log-normal (SC6 already).
- SMOE v1 (experiments/moe_smoe_dyn_v1, Modal H200): correct on bs1/2/4/8 incl. skewed, but 0.55-0.66x of the parent (bs1 3 experts 46.9 vs 29.2 us). Time ~linear in experts (~12.5 us/expert = 2.3 TB/s); NS 4 vs 6 identical -> producer serialized on one atomicAdd claim per unit (claim round trip ~2 us under 132-SM contention). Never claim every unit through one hot atomic.
- SMOE v2/v3 ablations (experiments/moe_smoe_dyn_v2, moe_smoe_abl_v3, Modal H200, paired vs parent): static round-robin claims = same speed as dynamic (claims not the cause). bs1(3 experts)/bs8(10): full 46.1/133.5 us; no-compute 36.3/95.5; no-epilogue 33.8/96.3; 4-way split bulk copies 46.5/134.5; parent 29.1/76.5.
  -> per-unit epilogue (global stores + __threadfence under saturated DRAM) costs ~25%; even without it this 1-CTA/SM 36 KB-unit TMA ring with K-split over 8 warps is 15-25% slower than the parent TINY/TINY2 build. REJECTED after 3 screens (wild budget). If revisited: dedicated reducer warps, one release per expert change instead of a fence per unit, and profile why the bare ring streams only ~2.5 TB/s.
