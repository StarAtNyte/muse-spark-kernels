# Decoder MoE M32 work ordering

The standalone MoE paired H200 screen in `experiments/moe_wgs_reord0_v1/` found 0.5–1.3% gains at M32 when phase A does not put a trailing partial unit first. This candidate applies that order only when `p.M == 32` in the current scored full decoder, retaining the existing order for all other token counts. Since full-layer interactions and runtime branching may change the result, standalone measurements do not establish a decoder gain.

Candidate SHA-256 `5d9247a74fe331532cd3dd8ebf6342f758dda4d0db86ca83de10e6c26808fd60`. Official full-decoder check job `29875ca2354e4673` passed at **2.08360x** (95% CI [2.05234, 2.11485]), compared with the parent check's **2.08859x** (95% CI [2.05726, 2.11992]). The intervals overlap and the point estimate is lower, so no gain is established. Keep the current canonical decoder and do not submit this candidate. Exact judge status and profiles are preserved here. This was an unscored check.
