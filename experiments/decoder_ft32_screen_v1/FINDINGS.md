# Decoder WGS MoE with FT_T=32 for 17..32 tokens — does not compile

Prompted by a public (unlicensed, vjeux-derived) decoder repo claiming 0.9-3.8 us at 32 tokens from an
`OKC_FT_T=32` build with 16 KB slots. Driver `dev/modal/drv_decoder_ft32.py` swaps `#define OKC_FT_T 64` -> 32 in
the decoder's MOE_CUDA_SRC via the `@ft32` alt-source hook. ptxas (CUDA 13) fails:
`C7600 Register allocation failed with register count of '96'` in `moe_fused<true>` (plus C7511 wgmma
serialization). The standalone MoE version of this change compiled earlier but lost 3-5% at bs 32
(`experiments/moe_wgs_ft32_v1`). Their build evidently changes more than the tile; not reproduced (no license to
read their code under).
