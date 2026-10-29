# WGS bare-skeleton pieces (2026-10-29)

Driver `dev/modal/drv_moe_skeleton2.py`, raw `modal.log` (on top of `bare` = no decode/scale/wgmma/waits/epilogue;
grid barrier kept). Time removed vs bare: x TMA copies ~1-2 us, x gather ~0.6-1.5 us, final pass ~3.8-7.4 us
(bs 32/128/256); all three together ~5-11 us. The rest of the gap to a pure TMA stream is in the weight ring and
producer/consumer handshakes.
