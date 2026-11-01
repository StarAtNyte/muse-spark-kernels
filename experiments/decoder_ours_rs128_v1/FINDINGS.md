# rs128 GEMV block (chenwainuo via keell0renz) ported into our decoder

Source-level port: `_make_chen` .. `_rq_gemm` appended, plus rq branches in `dense` and `kernel_fn` (971,684 bytes,
under the 1,000,000-byte limit). Check 28bf49b830714862: passed 13/13, 2.208x vs ours 2.226x (4a963397dbc840f9) and
keell0renz 2.264x (c1b314a3eeaa4d36) in the same window. Picks active: qkv m=128, qkv m=256, unpooler m=128.
Targeted loads: verify 16 sliding 2.163 -> 2.183, full 2.095 -> 2.144 (+1-2%). Untouched loads moved +-2-4% with no code
change, so the whole-file number is noise; geomean effect of the targeted loads is about +0.3%. Not submitted.
Remaining differences vs keell0renz decoder: SC6_WGS (scale compression, large merge), MFAB_LAYER/PCX_CFG per-layer tuning.
