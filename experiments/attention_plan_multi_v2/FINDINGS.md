# Attention plan sweep, remaining multi-request shapes — scored plans optimal

Driver `dev/modal/drv_attn_plan_one.py` via `dev/modal/run_cuda13_env.py` (one process per shape, parallel containers), raw `modal.log`. Best alternatives vs scored: verify 16 sliding none (ns8 0.996x), verify 32 sliding ns4 1.013x (= the tuned value; noise), decode 16 full ns264 1.005x, decode 32 full 1.000x, verify 16 full 1.003x, verify 32 full 0.998x. No change.
