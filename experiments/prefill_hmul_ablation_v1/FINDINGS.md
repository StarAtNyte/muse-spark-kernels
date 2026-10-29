# Prefill MoE scale HMUL2 ablation (2026-10-29)

Driver `dev/modal/drv_prefill_hmul_ablate.py`, raw `modal.log` (whole layer, attention pinned, paired). Removing convert8 scale multiplies (invalid outputs): turn-1 sliding 1.036x, turn-4 full 1.024x, first turn ctx0 1.002x, ctx65536 0.991x. Upper bound only 2-4% on short shapes; not the lever for the tensor-saturated long cases.
