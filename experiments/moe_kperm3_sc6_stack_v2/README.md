# KPERM3 + SC6 WGS MoE stack, v2

V1 combined lossless SC6 WGS scale packing with our KPERM3 layout, but official check `0cbe889464fe4756` failed before timing. The compiler identified omitted WGS completion-flag plumbing (`mf_xr`, `mf_da`, `Params::mfl`, and `MF_INTS`) in the extracted candidate. The submitted v1 file is preserved unchanged in its folder; the complete diagnostic is archived in [`failed_v1_compile.log`](failed_v1_compile.log).

V2 restores those definitions from the current known-good MoE incumbent. No kernel math, tile schedule, KPERM3/SC6 layout, or dispatch choice is changed.

Candidate SHA-256: `5f88c2b9488a0bc654b0b0f24a6e7ab3ef3ec83a727f6ac8b41611879113f8cf`. The CPU-only CUDA compile passed with `-DOKC_PRMTSC=1` (Modal app `ap-8DFTQl0K5OKMJQMT9kOBVC`). The exact source was submitted for official check `d237a4892dc54420`; status and launch output are archived here. Do not promote until it passes and its score beats the public 2.38203x incumbent.
