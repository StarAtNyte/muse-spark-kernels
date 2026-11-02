# KPERM3 + SC6 WGS MoE stack, v9

Follow-up to v8's official check `a5bdd94d9dc7460b`. That check passed bs=4
then failed on bs=33, 257, and skewed bs=128 because `_make_scratch()` omitted
the 4096-int MFIN flag region required by the selected WGS kernel. The incumbent
allocator already reserves this area. V9 restores that allocation and its DF
sentinel initialization while preserving the v8 SC6 WGS dispatch and removal
of optional `okc_attrs()` diagnostics.

The repaired source passed all five official correctness cases. Check
`bd7ee2f9d89247c8` scored **2.36004x** (95% CI **2.32464–2.39544x**),
below the published **2.38203x** point estimate but with overlapping
intervals. The check therefore does not establish either a regression or a
gain. Candidate component medians were 26.15, 94.29, 121.39, 64.55, 176.43,
and 211.33 us for decode 1/16/32 and verify 1/16/32, respectively. This
comparison is not sufficient to isolate SC6's effect from run-to-run noise.

The check log did not report whether the judge's scale bytes activated
`-DOKC_SC6W=1`. V10 adds a prepare-only diagnostic for the predicate and
compile defines, with unchanged timed code, and was submitted for scoring as
`1fb6b4f749ec44bb`. Until that log confirms activation and the scored result
returns, treat SC6 performance as unverified. Full check evidence and profile
are in `status.json` and `profiles/submission.bd7ee2f9d89247c8.*`.
