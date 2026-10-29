# MoE achievable-floor diagnostic (2026-10-29)

Modal H200, cold L2 (256 MiB flush), CUDA-graph replay, median of 40. Driver `dev/modal/drv_moe_stream_floor.py`;
raw log `modal.log`. Pure stream = one kernel reading exactly the active experts' w13/w13_sf/w2/w2_sf bytes
(`ld.global.nc.L1::no_allocate.v4`, 4 loads in flight per thread, best of grid 264x512 / 528x256).

| bs | distinct | MB | kernel_fn us | pure stream us | stream TB/s | kernel/stream |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 84.9 | 39.0 | 30.8 | 2.76 | 1.27 |
| 1 | 2 | 56.6 | 31.3 | 23.1 | 2.45 | 1.35 |
| 16 | 13 | 368.1 | 109.6 | 106.6 | 3.45 | 1.03 |
| 16 | 12 | 339.7 | 102.2 | 99.4 | 3.42 | 1.03 |
| 32 | 15 | 424.7 | 130.5 | 119.9 | 3.54 | 1.09 |

Empty-kernel graph: 4.8 us. The current WGS MoE is within 3-9% of a naive streaming read at bs 16/32, so
further MoE gains need a faster *read method*, not better math. Next: establish the H200's true achievable
streaming bandwidth (deeper LDG pipelining, TMA bulk copies) before redesigning.

First two attempts were invalid (probe launched on the legacy default stream, so the captured graph was empty).

Companion decoder diagnostic (`experiments/decoder_moe_floor_diag_v1/`, driver `dev/modal/drv_decoder_moe_floor.py`):
inside the decoder, the isolated MoE takes 31/68/98 us at decode 1/16/32 vs 4.8 TB/s floors 12/41/59 us.
