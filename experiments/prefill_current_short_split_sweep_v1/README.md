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
