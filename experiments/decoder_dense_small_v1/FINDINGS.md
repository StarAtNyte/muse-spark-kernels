# Decoder dense ops at T=1/8/16/32 vs the scored projection stack (2026-10-29)

Driver `dev/modal/drv_decoder_dense_small.py` (CUDA 13 H200, isolated, event-timed incl. ~5.5 us launch). Pairs are
(decoder route, projection stack) in us. Decoder tuner picks: bfp everywhere, fused pooler+router (bfpf) at T<=16.

| T | qkv | o | unpooler | pooler+router (decoder fused / stack sequential) |
|---:|---|---|---|---|
| 1 | 15.68 / 15.26 | 10.40 / 10.30 | 9.41 / 9.18 | 24.51 / 24.10 |
| 8 | 15.71 / 15.36 | 10.30 / 10.56 | 9.57 / 9.34 | 24.51 / 24.70 |
| 16 | 15.55 / 15.81 | 10.94 / 10.85 | 20.90 / 22.05 | 24.19 / 28.00 |
| 32 | 16.93 / 16.70 | 11.30 / 11.55 | 21.50 / 21.89 | 26.08 / 32.38 |

All differences <= 0.4 us except where the decoder wins; no small-T transfer worth a check.
