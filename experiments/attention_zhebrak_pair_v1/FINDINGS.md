# zhebrak/muse-spark-kernels attention (Apache-2.0) vs ours — rejected

Their published judge result showed decode-32 full at 150.6 us vs our 166.2 us (would have been a +1.4% hybrid).
Same-run paired H200 screen (`dev/modal/drv_attn_pair.py`, CUDA 13, all 12 scored shapes, both reference-checked;
their file at `dev/attn_zhebrak/zhebrak_attention.py`, license `dev/attn_zhebrak/LICENSE.zhebrak`):
their kernel is slower on every shape — decode 0.84-0.98x, verify 0.75-0.93x; decode-32 full 185.2 vs 181.6 us.
Their 150.6 us came from a different judge revision/run and is not comparable. No hybrid.
