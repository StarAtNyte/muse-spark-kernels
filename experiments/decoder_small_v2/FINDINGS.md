# decoder_small_v2: T=1/T=8 section floors and the MoE build choice

Parent: decoder_readable_fp32_ybuf_v4 (sha 616187c9...). Six Modal H200 rounds (modal1..6.log). Candidate: submission.py (sha256 319ea574...d37a7), the parent plus MOE_SMALL_BUILD = {1: "mid", 8: "mid"}. Kernels and math are unchanged.

## Floors at T=1 sliding (bytes / 4.2 TB/s)
qkv_gate 6.7 vs 10.8 us measured; attention ~1-2 vs 5.5; o+boundary 3.0 vs 4.3; pooler+router 14.0 vs 18.1; route 1.5 vs 1.4; MoE (2 experts, 56.6 MB) 13.5 vs 25.2; unpooler 1.5 vs 4.2; finish 1.5 vs 1.7. Layer 71.3 us at about 2.3 TB/s. T=8 MoE 66.5 vs 54. The MoE has a fixed cost of about 9 us at T=1 (about 7.5 us per expert).

## Variants (paired ratios, >1 = faster)
- 12-bit cgemv for qkv/unpooler at T=1: 0.997-1.023 (noise). Rejected.
- Side-stream L2 prefetch of the unpooler during the MoE: 0.967-1.001. Rejected.
- MoE build T=1 -> mid: decode-1 1.043/1.028; with T=8 -> mid as well: decode-1 1.044/1.029, verify-1 1.007/1.010. KEEP, pending okc check.
