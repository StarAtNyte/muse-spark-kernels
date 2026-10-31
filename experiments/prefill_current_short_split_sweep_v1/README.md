# Current prefill full-short attention split sweep

This rechecks the previously promising four-split short full-attention route
against the current scored prefill source (`research/best.json` SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`),
and screens five, six and eight splits. The prior four-split candidate scored
below its then-incumbent despite a small paired component gain; the current
prefill baseline has since changed, so neither its timing nor its integration
can be assumed to transfer.

The driver uses three seeds for synthetic B=1, query length 753, context 80000,
full-attention inputs; it compares each output against the two-split output
and uses alternating cold-L2 graph replays. This is a component screen, not an
independent challenge-reference check or full-layer score. A promising option
requires a paired full-layer test and the full official check before promotion.
The bounded public-image H200 run writes `modal.log`.

Run `ap-vGj0XGufTTYkK3RkwtXLRO` completed successfully. Every candidate was
finite and within the driver's 3% relative-L2 limit against the two-split
output; that comparison is not an independent correctness check. Median
paired parent/candidate ratios across seeds 539 / 1427 / 7919 were:

| Splits | Ratios |
| ---: | --- |
| 4 | 0.9688 / 0.9700 / 0.9701 |
| 5 | 0.8083 / 0.8073 / 0.8076 |
| 6 | 0.9382 / 0.9397 / 0.9381 |
| 8 | 0.9152 / 0.9142 / 0.9146 |

All alternatives lost in this component screen; retain two splits and do not
launch a full-layer check on this evidence. Earlier four-split measurements
used CUDA 12.8, while this run used the public challenge CUDA 13 image. The
attention CUDA section is identical between those source snapshots, but these
separate runs do not isolate the cause of the reversal. Raw logs preserve the
measurements; the old run is in `../prefill_full_short_four_splits_v1/`.
