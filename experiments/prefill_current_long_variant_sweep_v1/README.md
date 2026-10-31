# Current prefill long full-attention variants

On the exact scored prefill source (SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`),
compare the pinned variant 43 with 0, 1, 8, 9, 16 and 24 using their normal
split plans. The synthetic H200 cases are B=1, query length 16384, full
attention, context 32768 or 65536, page size 1 and two seeds. Each comparison
checks finite output and relative L2 against variant 43 and alternates six
cold-L2 CUDA graph replays. The public challenge image and bounded runner are
used; output and failures are saved in `modal.log`.

This is an exploratory component screen, not independent official correctness
validation or a full-layer score. Any promising change needs a paired
full-layer measurement and the complete official check before promotion.

Run `ap-TN0Sr45xYhTDjZRVo1wBIc` completed with return code 0. Every variant
passed the driver's relative-output screen, and all used one split. Every
alternative was slower than variant 43 on both seeds at both contexts.
Variant 8 was closest: parent/candidate median paired ratios were 0.9361 and
0.9361 at context 32768, and 0.9285 and 0.9293 at context 65536. Retain the
incumbent and do not submit these variants. Results are for synthetic
component inputs; they do not rule out new implementations or tuning choices.
