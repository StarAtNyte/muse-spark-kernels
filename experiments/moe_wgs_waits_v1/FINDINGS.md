# WGS consumer wait breakdown (2026-10-29)

Driver `dev/modal/drv_moe_wgs_waits.py` (alt source with clock64 around the consumer's weight-stage (BW_F) and
x-stage (BX_F) mbarrier waits; outputs reference-checked). Values are per consumer warpgroup for the CTA's last
segment of each phase (the counters are overwritten per segment), averaged over CTAs, in us:

| bs | phase A total | A wait weights | A wait x | phase B total | B wait weights | B wait x |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 15.4-15.9 | 1.7-1.8 | 0.5 | 7.4-8.2 | 0.7-0.8 | 0.3-0.4 |
| 128 | 16.1-16.2 | 1.8-2.1 | 0.5 | 7.8-7.9 | 0.7 | 0.3 |
| 256 | 15.9-16.3 | 1.1-1.2 | 0.5 | 8.4-8.6 | 0.6-0.7 | 0.3-0.4 |

Consumers wait ~10-15% of the time; the rest is decode + scale + wgmma issue. With the ncu profile
(`experiments/moe_wgs_stalls_v1`: ~2.4 lane-instr per weight, 57% issue-active) this makes the WGS kernel
compute/issue-bound: fewer instructions per FP4 weight is the lever.
