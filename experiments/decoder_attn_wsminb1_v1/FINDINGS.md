# Decoder full-attention one-block register budget

Candidate SHA-256 `2428225b8ae3f7d95e95bb5db5b42b5e6681a0a104805814c25d9211bed1e631` changes `OKC_WSMINB` from 2 to 1 in the embedded decoder attention source, changing the launch/register budget for warp-specialized flat attention. Parent SHA-256 `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.

Paired CUDA 13 H200 whole-layer screen: three seeds each for full-context decode batches 16 and 32; both implementations passed all six reference checks. Parent/candidate ratios were 0.9360/0.9293/0.9330 at batch 16 and 0.9180/0.9148/0.9193 at batch 32, about 7–9% slower. Both preparations reported the same tuned route keys; that does not prove all tuning choices were identical. Reject the one-block setting. No official check or scored submission was spent. Raw output and driver are archived here.
