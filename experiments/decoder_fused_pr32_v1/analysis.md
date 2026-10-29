# Fused pooler and router at 32 tokens

The decoder already implements `bfp_fused_pooler_router`, but its current pinned 32-token dispatch uses separate BF16 pooler and router projections. This candidate changes that one pinned pick to the fused route, testing whether saving a launch and sharing input reads improves the whole decoder despite PDL interactions. Official unscored check `ca5f44975b6c4f41` is pending; no correctness or performance conclusion is available yet. Candidate SHA-256 `43dd9d91ac6d38b1646641bb9e4fee43c50612a7c9b39cf1278457af63a9fd79`; the canonical decoder is unchanged.
