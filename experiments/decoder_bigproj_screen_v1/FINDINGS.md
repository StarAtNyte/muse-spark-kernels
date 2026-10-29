# Decoder T=128 qkv and T=128/256 o through the projection stack (2026-10-29)

Isolated (event-timed, `experiments/decoder_dense_big_v1`, driver `dev/modal/drv_decoder_dense_big.py`):
qkv 128 19.6 vs 24.7 us (cuBLAS), o 128 12.3 vs 13.6, o 256 13.8 vs 14.9. Splitting the fused pooler+router into the
stack's pooler + router loses (~+2-4 us in chain); the stack routes unpooler 128/256 and qkv 256 to SGLang's dispatch.

Candidate `candidate.py` (= `dev/decoder_bigproj/submission.py`): `_qkv_mm` for the fused-attention path and a
`dense` override (o by name, qkv by weight identity for the FA3 path). Paired whole-layer screen
(`dev/modal/drv_decoder_pair.py`, CUDA 13 H200, every output reference-checked; FA3 unavailable on Modal):

| case | base us | cand us | gain |
|---|---:|---:|---:|
| verify 16 full | 341.0 / 370.1 | 333.9 / 366.7 | 1.021 / 1.009 |
| verify 16 sliding | 274.5 / 271.0 | 270.5 / 264.0 | 1.015 / 1.026 |
| verify 32 full | 519.3 / 519.6 | 516.6 / 517.5 | 1.005 / 1.004 |
| verify 32 sliding | 327.9 / 321.9 | 326.8 / 320.7 | 1.003 / 1.004 |
| decode 1 sliding (unchanged) | 88.4 / 88.4 | 87.7 / 87.6 | 1.008 / 1.009 |
| decode 32 sliding (unchanged) | 149.3 / 148.9 | 148.9 / 149.2 | 1.003 / 0.998 |

Expected decoder score effect ~+0.5%. Official check `6047870f0fe44acb` submitted.
