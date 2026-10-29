# 128-token WGS with two weight and two activation slots

The one-activation-slot 128-token variant had to drain all WGMMA groups before each slot reuse. This candidate retains the 128-token tile but exchanges one of three weight slots for a second activation slot, restoring cross-stage overlap. The H200 shared-memory allocation succeeded. The candidate and baseline were compiled on CPU and transferred to H200.

All six task-reference checks passed with bitwise-identical output. Median paired baseline/candidate ratios over seeds 419 and 7919 were 0.9701 and 0.9721 at M32, 0.9446 and 0.9426 at M128, and 0.9377 and 0.9233 at M256. This substantially improves on the earlier one-slot variant but remains approximately 3–8% slower than the scored baseline. Do not promote or submit. This is a component screen, not an official score.

Candidate SHA-256 `d97ccd9d947fa12d701d31ae2b500abc5c0f2b6ee055d83255021af38a473fca`; baseline SHA-256 `a7fa7eac2b2f6a943ceb977197d992d2fe770aea7a1f65a82ae9d4462127d48a`. Modal app `ap-WmxEXtKKs9Cx5ZekPmg0Ot`; raw output `screen.log`. Command: `modal run dev/modal/moe_ft128_prebuilt.py --base kernels/muse_moe_w4a16/submission.py --candidate experiments/moe_wgs_ft128_w2x2_v1/submission.py`.
