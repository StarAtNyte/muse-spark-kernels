# Standalone MoE tiny CUDA source transfer into stacked decoder

Candidate SHA-256 `8a05d2409da8e86294b8a3107508d9baed320a0362639a455002deccad586fbe`, 1,002,450 bytes. The source reconstructs the standalone MoE `VAR_SRC` exactly from a compressed edit script against the embedded decoder CUDA source, then uses it with `OKC_DF=0` for M<=8. Larger MoE builds retain the stacked decoder's sources. This is an experimental local file; its size needs checking against the challenge submission limit before any submission.

CUDA 13 H200 paired full-layer screen against `kernels/muse_decode_layer/submission.py`, three seeds per case, passed all 12 reference checks. Parent/candidate ratios: decode-1 sliding 1.0015/0.9912/0.9895, decode-1 full 1.0081/0.9967/1.0029, verify-1 sliding 1.0051/1.0134/1.0061, unchanged decode-16 sliding 1.0008/0.9941/1.0035. Baseline preparation reported three tuned route keys, candidate preparation none; therefore this comparison does not establish identical tuning or judge parity. The affected results do not establish improvement over the previously tested original-warp candidate, so no official check was spent. Keep the earlier checked stack as the scored candidate.

Raw output is `modal.log`; driver is `../decoder_tiny_warp_fair_v1/driver.py`. No leaderboard submission was made.
