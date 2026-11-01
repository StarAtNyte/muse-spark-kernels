# Projection hybrid (ours + embedded keell0renz module) - closed, unresolved

Attempts: v2 (kz prepare after ours) and v3 (+ primary-context bind) fail in setup with `routertail launch failed: 912`
(cudaErrorInvalidClusterSize, Nathen's 16-CTA cluster router-tail launch inside kz `prepare`'s tuning); v4 (kz prepare
first) fails later in our own `olayer` launch; v5 (every kz load_inline extension name prefixed `kz_`, after finding that
both our embedded vjeux copy and kz build `okc_bf16proj_v141_*` from different sources) still fails with the same 912.
So the extension-name collision is NOT the cause. Each module works alone; together a launch config breaks, most likely
some device-wide state our prepare leaves (not found). Untried: bisect our prepare by phase and call kz's routertail
after each; or drop the router shapes from the routed set and stub kz's router tuning.
Value if solved: ~+1% on the projections board (paired per-shape data in experiments/projection_alygg_compare_v1,
bf16_keell0renz_compare_v1). Fallback in place: keell0renz projections file as the base (scored 1.5853x, rank ~45).
