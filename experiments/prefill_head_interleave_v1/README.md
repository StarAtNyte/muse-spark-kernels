# Long prefill KV-head interleaving

The scored prefill attention route launches query-block index as the fastest
grid dimension and KV head as the next dimension. For one split it therefore
runs one head's descending query blocks before the other head's. This screen
interleaves the two KV heads per descending query block, preserving one-to-one
coverage and longest-query-first order. The split route remains unchanged.
Better distribution may reduce tail, while changing cache locality and block
placement may cause a regression; no benefit is assumed.

The driver builds both variants from the exact scored source SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`.
It checks finite outputs and relative L2 against the incumbent and measures
12 alternating cold-L2 graph replays at B=1, query length 16384, full attention,
contexts 32768 and 65536, page size 1, two seeds. The H200 public-image runner
limits the subprocess to 300 seconds and stores raw results in `modal.log`.
This is a synthetic component screen; a promising candidate requires full
layer paired timing and official correctness before promotion.

Component run `ap-I2mvtgb7yndSPYjQqXX8Fb` passed its output comparison with
zero observed relative L2 difference. Median paired parent/candidate ratios
were 1.00565 / 1.00564 at context 32768 and 1.00267 / 1.00264 at context
65536, for seeds 539 / 1427. This small gain justifies a full-layer screen,
not a score claim. Raw output is `modal.log`.

`candidate.py` applies the interleaving only for long, full, one-split
attention; SHA-256
`8099523e1e9e97aba8ca51d6743f7331927548bc79bdf379a675635a27107099`.
`full_layer.py` compares it with the scored parent on full-attention contexts
0, 32768 and 65536, with pinned attention choices and alternating cold-L2
graphs. It compares layer outputs and routing against the incumbent, not the
independent reference. Run output is `full_layer.log`; the bounded runner
limits its subprocess to 300 seconds. Official correctness is still required.
