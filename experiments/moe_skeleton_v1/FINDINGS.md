# WGS MoE skeleton decomposition (2026-10-29)

Driver `dev/modal/drv_moe_skeleton.py`, raw `modal.log` (outputs invalid by design for ablated variants).
Note: `OKC_WGSX=1` (documented "no wgmma") is not wired in this source — only bit 8 is — so the earlier "nomma"
ablation in `experiments/moe_compute_ablation_v1` was a no-op; this run removes the wgmma calls in source.

| bs | x8 (no dependency waits, no epilogue) | + no wgmma | bare (also no decode, no scale) |
|---:|---:|---:|---:|
| 32 | 1.000-1.022x | 1.040-1.058x | 1.075-1.088x |
| 128 | 0.961-0.976x | 1.037-1.039x | 1.065-1.078x |
| 256 | 0.918-0.927x | 1.021-1.022x | 1.072-1.112x |

Even the bare streaming scaffolding takes ~123 us at bs32 (425 MB) vs 108 us for a pure TMA stream: ~15 us is in the
skeleton (routing prologue, x gather/ring, producer/consumer handshakes, final pass). A rewrite must restructure that
scaffolding; math and waits together are worth <=11%.
