# Decoder adaptive WGS prefix dispatch

Prepared follow-up to `decoder_wgs_parallel_prefix_v1`; it has not been screened.

V1 showed about +0.10% aggregate in a paired H200 screen, but batch 1 regressed by about 0.30% while batches 16 and 32 gained. V2 restores the exact serial lane-0 prefix/range path when `nj <= 16`, and keeps the V1 warp-parallel scan when `nj > 16`. This is a targeted dispatch hypothesis, not a predicted score gain.

The exact V1 candidate is the parent; candidate source and paired driver are archived here. Do not run the H200 comparison until official scored run `03083a8a606d42e2` is terminal, to avoid overlapping GPU spend.
