# WGS MoE compute ablations (2026-10-29)

Drivers `dev/modal/drv_moe_compute_ablate.py` (logs `modal.log`, `modal2.log`; outputs invalid by design). Gains vs scored at bs 32/128/256: no decode 1.009-1.035x, no decode + no scale 1.012-1.050x, no wgmma (OKC_WGSX=1) 0.994-0.998x, skeleton (no decode, no scale, no wgmma) 1.017-1.047x. The kernel time is the data-movement + synchronization skeleton (~3.3 TB/s), not math: streaming efficiency is the lever.
