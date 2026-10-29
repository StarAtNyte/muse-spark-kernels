# Upper bound for compressing MoE block scales (2026-10-29)

Scales: generated log2 ~ N(7, 0.4) clamped to [8, 448] -> only 47 e4m3 codes (0x50..0x7E): 6-bit packing is lossless
without escapes; ~4-bit entropy coding would need escapes. Codes are uniform over 16 patterns (incompressible).
Scale bytes are 1/8.5 of expert bytes.

Ablation (`dev/modal/drv_moe_scale_ablate.py`, raw `modal.log`): the WGS producer copies UNIT_BYTES - X per 36 KB
stage (compute unchanged, outputs invalid by design). Paired cold-L2 medians:

| bs | drop 1 KB (6-bit) | drop 2 KB (4-bit) | drop 4 KB (no scales) |
|---:|---:|---:|---:|
| 32 | 1.007-1.008 | 1.022-1.027 | 1.063-1.069 |
| 128 | 1.011-1.015 | 1.031-1.033 | 1.059-1.061 |
| 256 | 1.010-1.014 | 1.017-1.022 | 1.030-1.043 |

These are ceilings (a real format adds decode work). 4-bit + escapes would net roughly +2% on the WGS loads (3 of 6),
about +1% on the MoE score, for changes to the WGS, MID and TINY layouts and decoders. Not pursued for now.
