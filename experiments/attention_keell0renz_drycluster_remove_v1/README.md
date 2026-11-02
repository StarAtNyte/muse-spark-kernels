# Remove the attention cluster dry pass

A small, isolated variant inspired by the [keell0renz attention source](https://github.com/keell0renz/muse-open-kernel-challenge), which removes the no-load/no-store `cluster_finish(..., dry=true)` warmup. The incumbent executes this in the prologue to warm instruction cache; it adds cluster synchronization before useful attention work. The candidate removes only that call. Actual output math and the real cluster finalizer are unchanged.

Parent SHA-256: `573f179558bb2eae13faa7a58ecfdd924196bb15e9bfd25fd552289ae903c1da`

Candidate SHA-256: `5e280d2ae66a1738d8decc677b370f6c2350321e773aace9d4fd5e41e1131e85`

Screen: batch 16/32, q_len=1 sliding, two seeds, 16 randomized cold-L2 paired rounds; both outputs passed the reference check. Results were B16 1.0315×/0.9905× and B32 0.9965×/1.0112× across the two seeds. The mixed seed outcomes do not support a repeatable gain; no official check or promotion. Modal app: `ap-9Pw85CTYLioyOcr7R7reiR`.
