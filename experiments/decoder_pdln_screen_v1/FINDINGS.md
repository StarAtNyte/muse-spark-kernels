# Decoder sk2 pre-wait stages (BFP_PDL_N) 2 -> 4/8/16 — rejected

Driver `dev/modal/drv_decoder_pdln.py`, raw `modal.log`. Every deeper setting is slower on every shape (0.976-0.996x); extra early weight traffic slows the still-running predecessor (DRAM-bound chain). This also argues against overlapping MoE weight streaming with the pooler.
