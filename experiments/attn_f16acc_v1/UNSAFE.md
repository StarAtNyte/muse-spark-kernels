# Do not submit

FP16 P·V accumulation here holds the sum in FP16 for the whole split (P up to 2^8 under the lazy offset, worst case 256·N·448). It produced NaN on every kv_scale 1/64 stress case (flat B=32/64 coherent V, 1M context, sliding B=16, soft distributions). Superseded by `experiments/attn_f16qk_v1` (FP16 QK only, PV FP32).
