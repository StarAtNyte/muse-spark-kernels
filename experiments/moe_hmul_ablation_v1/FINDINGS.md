# WGS scale HMUL2 ablation (2026-10-29)

Driver `dev/modal/drv_moe_hmul_ablate.py`, raw `modal.log`. Removing the four bf16 scale multiplies per k16 step (outputs invalid by design) gains only 0.996-1.014x at bs 32/128/256: the FP16/tensor shared pipe is not what limits the decode-size WGS kernel.
