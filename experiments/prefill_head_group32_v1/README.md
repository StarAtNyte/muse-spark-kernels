# Grouped KV-head scheduling for long prefill

Compare 32 adjacent descending query blocks per head against the interleaved
head candidate `prefill_head_interleave_v1/candidate.py` (SHA-256
`8099523e1e9e97aba8ca51d6743f7331927548bc79bdf379a675635a27107099`).
The grouping could recover within-head cache locality while retaining much of
the load balance; its extra index arithmetic and any caching changes must be
measured. `verify_mapping.py` passed exhaustive coverage and uniqueness for
query-block counts 1–1024; this is an index check, not GPU correctness.

The bounded public-image H200 component driver checks output against the
interleaved candidate and runs 12 alternating cold-L2 graph comparisons at
full-attention contexts 32768 and 65536, query length 16384, page size 1 and
two seeds. Output is in `modal.log`. A useful result would require complete
layer testing and official correctness before promotion.

Run `ap-utw25WWaB3sCYLuX6HNJZF` completed with return code 0 and zero
observed relative output differences. Median paired interleaved/grouped ratios
were 0.99928 / 0.99961 at context 32768 and 0.99996 / 1.00009 at context
65536 (seeds 539 / 1427). This does not establish a performance improvement;
retain the simpler interleaved candidate and do not submit this variant. The
component screen does not rule out a different grouping or workload.
