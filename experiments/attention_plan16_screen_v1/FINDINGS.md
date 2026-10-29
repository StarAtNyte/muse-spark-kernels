# Attention: 8 splits, no coop finalize for 16-request sliding decode

Sweep `experiments/attention_plan_multi_v1` (1.029x), paired 12-shape screen here (`dev/modal/drv_attn_pair.py`): target 13.2 -> 13.0 us (1.015x); other shapes unchanged within noise. Candidate `dev/attn_plan16/submission.py`.
