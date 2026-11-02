# Decoder SC6 skinny M<=16 transfer: correctness failure

Candidate SHA-256 `5e875c54f4811c5eca12e2613e03b94eec05bfe06885b37e6419c8241995c470`. This experiment routes all M<=16 expert launches through the decoder's SC6-aware source and packs the skinny scales into lossless 6-bit planes.

CUDA 13 H200 paired whole-layer screen: M<=8 outputs passed reference checks. B=1, q_len=8 sliding measured 1.0145x, 1.0212x, and 1.0177x over three seeds; decode-1 was neutral. M=16 failed all three reference checks with residual relative L2 error 1.01–1.06. Do not use SC6 in the mid build. The safe M<=8-only follow-up and its official result are in `../decoder_sc6_skinny_tiny_v1/`.

Raw Modal output: `modal.log`. This candidate was not submitted to the official checker.
