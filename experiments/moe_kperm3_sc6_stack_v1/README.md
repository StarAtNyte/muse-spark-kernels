# KPERM3 + SC6 WGS MoE stack

This is an exact copy of the previously unvalidated `experiments/moe_stack_sc6_v1/submission.py` candidate (SHA-256 `d16a09b8d488f528db0ef70914ceec38ab0dd2acd51abb4cdc9908a77352ad04`), which combines our KPERM3 expert layout with lossless six-bit WGS scale packing. The source includes the complete matching WGS stage size, producer stride/TMA byte count, and consumer decode. This avoids the incomplete keyword-only merge that broke the decoder candidate at T=24.

SC6 is used only after preparation validates that the local scales satisfy its lossless 6-bit encoding; otherwise the candidate retains the full-scale WGS path. The prior SC6-only fallback candidate passed but scored 2.36855×, below our 2.38203× public MoE best. This integrated source tests whether KPERM3 and SC6 stack.

Candidate SHA-256: `d16a09b8d488f528db0ef70914ceec38ab0dd2acd51abb4cdc9908a77352ad04`. Full official check is being submitted; promote only if all tests pass and the measured result/profile supports a gain over the public incumbent.
