# KZ projection routes, v2: legal cluster size during eager prep

V1 failed during `prepare()` with `cudaErrorInvalidClusterSize`: its KZ helper eagerly benchmarked a router kernel configured for 16 CTAs per cluster. The scored dispatch does not use the router route, but preparation still launched it.

V2 changes only the embedded helper's router tuning table from 16-CTA clusters to its already-compiled 8-CTA specialization for M=8/16/32. The live KZ route selection remains unchanged, including the existing fallback for KZ route prep failures. The affected 8-CTA kernel path is an explicit supported specialization (`RSM(8,8)`).

Source SHA-256: `ea17ccc5d829e6359876dd864b4d3092d539fb8277d4a6af6e369b85dcfd9880`. Parent candidate: `projection_kz_router_fallback_v1/candidate.py`. V1 judge job `2f36597213e64092` failed at prepare before timing; the failure log is preserved in v1. Gate v2 on a complete official check, then compare the exact scored result with the 1.5853x published projection incumbent.
