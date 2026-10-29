# Projection qkv M256 + unpooler M128 via wide2 (2026-10-29)

Candidate `candidate.py` (= `dev/proj_wide2x/submission.py`; two wide2 keys enabled). Paired H200 screen (`dev/modal/drv_proj_wide2x.py`, CUDA 13; reference-checked) vs torch.mm, which is what the stack's SGLang fallback (F.linear) runs: qkv M256 28.21/28.51 -> 26.83/26.77 us (1.051/1.065x), unpooler M128 29.68/29.78 -> 27.42/27.74 us (1.082/1.073x). Expected projection score ~+0.6%.
