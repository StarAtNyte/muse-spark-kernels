# Projection qkv M32 via the specialized qkv16 kernel — not supported

Judge profile: qkv M16 runs r3_qkv16_..._push_d9_cap6 (10.6 us), qkv M32 the generic m32x_q_gemv<1,4,...> (11.4 us; case 12.0 us vs 9.9 at M16). Routing M32 to the qkv16 kernel (`candidate.py`, driver `dev/modal/drv_proj_pair.py`) fails at launch with rc -5: the instance only supports m <= 16. A 32-row instance would need the generated variant families rebuilt. Not pursued.
