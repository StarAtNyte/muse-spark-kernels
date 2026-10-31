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
