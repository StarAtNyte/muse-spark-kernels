# Prefill routed-row padding diagnostic

H200 public-image app `ap-cupnEsJYQMywfntiOkuAL7` measured top-8 IDs emitted by the scored prefill submission (SHA-256 `f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`) after a reference-checked forward. It counts IDs in the selected 32-expert range and reconstructs the existing `rem_tiles` padding in Python. The `ideal_32` comparator rounds each nonzero expert's count to 32 rows; it is an arithmetic comparator, not an implemented schedule, hardware lower bound, or latency estimate.

| Case, one seed | Local routes | Current padded rows | Hypothetical 32-row rounding | Excess rows |
|---|---:|---:|---:|---:|
| Turn 1 sliding, seed 9400 | 1044 | 1440 | 1344 | 96 |
| Turn 4 full, seed 9401 | 4303 | 4704 | 4640 | 64 |
| First-turn sliding, seed 9402 | 23497 | 23808 | 23776 | 32 |

All three reference screens passed. The distribution is very skewed in these samples; the real per-expert counts and raw output are in `screen.log`. Relative to current padded rows, the excess is 6.67%, 1.36%, and 0.13%, respectively. This is not a bound on runtime or score: tile efficiency, launch scheduling, additional tile widths and load balancing matter, while 32-row rounding itself retains padding. The public-image setup uses a SGLang stub and pinned attention, with one seed per shape; the official judge chooses routing inputs differently and nearby cases or other seeds can differ. This weakens the case for adding fine remainder widths for these specific samples but does not rule out alternative layouts or other routing distributions.
