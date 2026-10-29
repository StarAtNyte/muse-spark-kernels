# Decoder attention plan sweep, other shapes — scored plans optimal

Driver `dev/modal/drv_decoder_attn_plans.py` (whole layer, _TUNED7 overrides, 2 seeds), raw `modal.log`. Best alternative gains: decode 32 sliding 1.004x, decode 1 sliding 1.002x, verify 1 sliding 1.008x, verify 16 sliding 1.001x. Only the 16-request sliding decode (separate screen, 1.07x) was mis-planned.
