# Attributed public projection implementation

Upstream: https://github.com/NathenFernandes/muse-bf16-proj
Commit: 86aff75e6aca72511bf9b41b6dc190bf1e763373
License: MIT, Copyright (c) 2026 Nathen Fernandes, retained in LICENSE.nathen
and every candidate using substantial source. This is an adaptation of public
work, not an original implementation by this campaign.

Upstream SHA256 111630ff363e5988ca57f28434a2e1a7e9c1d524079c64727b52c6aa287dcb44
matches its published result record. The attributed projection source is
AST-identical; only provenance/license comments were added. Private check
575f2c20ec7f4236 passed 9 tests and 30 timings at 1.510551x. Public run
658927e88d4f4221 passed at 1.517684x. Projection rank refreshed to 84, versus
rank-50 speedup 1.5295; overall rank 331. These are snapshots, not final ranks.

The reproduction is about 20% lower latency than our previous private best,
and about 21% lower than our previous public score. Source and complete results
are archived in experiments/projection_nathen_v18 and scored_projection_nathen_v18.

build_decoder.py prefixes every public module binding, keeping its internal
CUDA and dispatch unchanged, then integrates its four BF16 dense projections
with our verified decoder_pre_norm_overlap_v1. It avoids building unused legacy
dense weight tiles. Our attention, expert kernels, router selection, norms and
residuals stay in the decoder. The decoder's <=8-token FP32 router input is not
cast to BF16 for reuse of the public projection interface. Its own full official
check is pending as decoder_public_projection_v1, job 7b726d295a6747bc.
