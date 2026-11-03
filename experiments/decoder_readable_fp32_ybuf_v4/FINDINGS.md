# Decoder WGS per-route values retained in FP32

The readable decoder's main WGS MoE normally writes each expert's completed second-matmul result into a BF16 `ybuf` before the final normalization and route combination (`OKC_YBF=1` for the non-B4 build). The organizer's quoted thread did not settle the precision requirement for MoE partials, and it is unclear whether this completed per-expert value is covered by that question. V4 conservatively stores those values as FP32 and uses the existing FP32 final-pass path; the B4 build already used FP32. The projection fallback changes from V3 remain intact.

Private check `2996c1822f9d417a` is pending. Syntax passed. This change does not by itself establish full compliance; inspect all other paths and compare per-case latency before submitting.

Private check `2996c1822f9d417a` passed 13/13 at 2.24462× (95% CI 2.21095–2.27829×). Exact SHA-256 `616187c9dac236851cc4b76bb65cd1cf9c85490db96f241a2fb1fa3d0fcff2cc` was copied byte-for-byte to the public-run archive; scored job `529df1f504434040` submitted and pending. This is a separate run and does not guarantee the same score or establish final leaderboard eligibility.
