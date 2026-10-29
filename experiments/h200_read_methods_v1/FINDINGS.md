# H200 achievable read bandwidth by method (2026-10-29)

Driver `dev/modal/drv_read_methods.py`, raw `modal.log`. One contiguous buffer, cold L2, CUDA graph, median of 30;
empty graph 4.9 us. Best rows:

| MB | best method | us | TB/s (incl. launch) | TB/s (minus empty) |
|---:|---|---:|---:|---:|
| 56.6 | LDG v4 x4 (all LDG variants tie) | 21.0 | 2.70 | 3.53 |
| 113.2 | LDG v4 x8 | 34.4 | 3.29 | 3.83 |
| 340 | TMA bulk 4x32 KB stages, 1 CTA/SM, contiguous slab per CTA | 89.1 | 3.82 | 4.04 |
| 425 | TMA bulk 8x16 KB stages | 107.8 | 3.94 | 4.13 |

- TMA bulk into smem with contiguous per-CTA slabs beats strided LDG by 1.5-2% at large sizes; LDG unroll depth
  and the L2::256B hint make no difference.
- Practical floor model: time ~= bytes / 4.1 TB/s + ~5-7 us fixed (launch + ramp).
- Applied to MoE (same bytes, `experiments/moe_stream_floor_diag_v1`): bs1 31.3 vs ~21 us, bs16 102-110 vs ~89 us,
  bs32 130.5 vs ~108 us. The segmented/strided stream probe there (120 us at 425 MB) was itself 11% off the
  contiguous-TMA floor, so the MoE kernel's gap to the true floor is 13-33%, not 3-9%.
