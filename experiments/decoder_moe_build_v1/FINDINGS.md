# Decoder MoE build per token count (MID vs WGS at T=16/32) — scored choice stands

Driver `dev/modal/drv_decoder_moe_build.py`, raw `modal.log` (whole layer, reference-checked, paired). T=16: MID (scored) ties, WGS 0.91-0.96x. T=32: WGS (scored) ties, MID 0.954/1.051/0.991/0.974x — routing-dependent; a per-call choice from routing would be input-value-dependent control flow (not allowed in the captured graph).
