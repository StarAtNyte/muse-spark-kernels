# KPERM3 + SC6 WGS MoE stack, v9

Follow-up to v8's official check `a5bdd94d9dc7460b`. That check passed bs=4
then failed on bs=33, 257, and skewed bs=128 because `_make_scratch()` omitted
the 4096-int MFIN flag region required by the selected WGS kernel. The incumbent
allocator already reserves this area. V9 restores that allocation and its DF
sentinel initialization while preserving the v8 SC6 WGS dispatch and removal
of optional `okc_attrs()` diagnostics.

This is a conformance repair; it does not change the kernel performance
hypothesis. Await the official check before interpreting any timing.
