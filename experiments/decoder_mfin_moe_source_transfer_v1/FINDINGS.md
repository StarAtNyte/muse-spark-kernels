# Decoder MFIN/WGS source transfer

Candidate retains the exact scored decoder's attention, projections, residual boundaries, and host orchestration. It replaces only the embedded `MOE_CUDA_SRC` with the newer source in `kernels/muse_decode_layer/submission.py`, and adds the 4096-int MFIN flag region required by that source. WGS MFIN is enabled by the transferred source; SC6 packing remains disabled by default because this candidate keeps the incumbent's weight preparation.

The transferred MoE source also includes later WGS scheduling changes, so any measured delta belongs to this coherent source bundle and cannot be assigned to MFIN alone. Exact parent source and hashes are in `manifest.json`. Screen on the judge image with paired H200 full-layer timing and correctness before deciding on an official run.


## Paired H200 screen

All 36 candidate and parent cases passed correctness. With three seeds per shape and four alternating replays, the candidate/parent geometric timing ratio was **1.00448× overall**, **1.00706× sliding**, and **1.00190× full**. The strongest repeatable gains were on sliding batch-32 decode (about 2.1%) and verify (about 1.1%); small batch-1 verify-8 had a mixed/noisy result. Full raw measurements are in `screen.json`; Modal container log is `modal.log`.

The exact candidate hash was submitted as official run `ff5cef4ac6ef4810`. It passed 13/13 correctness checks but scored **2.19734×** (95% CI **2.16438–2.23030×**), below the 2.26032× incumbent. Reject it and retain the incumbent. The judge result did not confirm the small paired screen gain. Official profile files are in `profiles/`.
