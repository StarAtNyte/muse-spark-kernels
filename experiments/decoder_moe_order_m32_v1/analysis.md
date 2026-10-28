# Decoder MoE M32 work ordering

The standalone MoE paired H200 screen in `experiments/moe_wgs_reord0_v1/` found 0.5–1.3% gains at M32 when phase A does not put a trailing partial unit first. This candidate applies that order only when `p.M == 32` in the current scored full decoder, retaining the existing order for all other token counts. Since full-layer interactions and runtime branching may change the result, standalone measurements do not establish a decoder gain.

Candidate SHA-256 `5d9247a74fe331532cd3dd8ebf6342f758dda4d0db86ca83de10e6c26808fd60`. Official full-decoder check job `29875ca2354e4673` was submitted on 2026-10-28 and is running; it is not a scored submission. Check the job result and preserve its profiles before deciding whether to submit. The current canonical decoder is unchanged.
