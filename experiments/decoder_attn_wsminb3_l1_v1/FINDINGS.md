# Decoder attention L=1 three-block budget

Candidate SHA-256 `dd13cc78c22978b12b69c4e918473a0db5767fa828e5bb5fbf8e5d158665b03f` changes the warp-specialized flat attention minimum-block budget to 3 only for L=1, preserving the original L=8 budget. The global variant failed compilation and is documented in `../decoder_attn_wsminb3_v1/`.

Paired CUDA 13 H200 whole-layer screen for full-context decode B=16 and B=32, three seeds each, passed all six reference checks. Parent/candidate ratios were 0.9797/0.9757/0.9796 (B16) and 0.9631/0.9652/0.9651 (B32). Candidate preparation skipped its tuning bake and reported no `chen_rq` keys, while the parent reported three, so this measurement does not isolate the attention change or establish judge parity. It gives no evidence for promotion; no official check or scored submission was used. Driver: `../decoder_attn_wsminb1_v1/driver.py`; raw output: `modal.log`.
