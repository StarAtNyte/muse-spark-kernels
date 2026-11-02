# KZ route stack with a safe router tuning fallback

The archived `projection_stack_kz_routes_v2` combines eight KZ projection routes with our existing projection implementation. Its full H200 check failed during `prepare()` when the embedded `routertail_small` tuning kernel returned CUDA error 912, before the judge could score the candidate. This copy changes only the embedded tuning fallback: on a nonzero router-tail launch status it computes the exact FP32 router product from `router_raw`, allowing the route tuner to continue. The outer dispatch deliberately keeps router shapes on our incumbent; the fallback is limited to KZ's preparation/tuning and the live router path remains unchanged.

Parent candidate: `projection_stack_kz_routes_v2/submission.py`, SHA-256 `f918e8ef464940a258de0fc80a18234a113047cc0d6f1544082e82229c9f0bc6`. Candidate SHA-256: `c454a8212ee9116b88c85f423dfcb72b1a39e5632f42b014cdd224c5d617d74e`.

Official check `2f36597213e64092` failed during `prepare()` before scoring, with `cudaErrorInvalidClusterSize` from the eagerly benchmarked router helper. The exact status and full H200 log are preserved in this folder. V2 changes only that helper's prep specialization to an 8-CTA cluster and is checked as `a7bd51ec7e13491a`; see [`projection_kz_router_fallback_v2`](../projection_kz_router_fallback_v2/README.md).
