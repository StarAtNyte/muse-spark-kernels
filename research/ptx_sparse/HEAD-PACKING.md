# Exact two-head packing for sparse tensor cores

This prototype computes dense attention. Hardware sparsity describes only
explicit zeros introduced between independent KV heads. No real key, value,
query component, or softmax contribution is discarded.

For each adjacent pair of embedding dimensions d,d+1, expand one key column as

```
B = [K0[d], K0[d+1], K1[d], K1[d+1]]
```

The two groups of query rows have expanded operands

```
A_head0 = [Q0[d], Q0[d+1], 0, 0]
A_head1 = [0, 0, Q1[d], Q1[d+1]]
```

Each four-element block has two structural zeros. Store every real Q element
in compressed A, and describe its location using metadata. The dot product is
exactly Q0*K0 or Q1*K1, with no cross-head products. Concatenating all dimension
pairs and all key columns gives the original dense QK products.

For P×V apply the same construction to adjacent pairs of key positions. Store
all FP16 probabilities for each query in compressed A. The expanded B rows
are the corresponding values for both heads. Every key contributes to its
own head's softmax-weighted output.

Decode has eight query rows per head, while a warp MMA has at least sixteen.
Combining the two heads fills those rows. This may eliminate tensor-core work
that the separate-head implementation spends on padding. Verification already
has 64 rows per head; no equivalent padding win is assumed for it.

Operand precision stays FP8 for Q/K, FP16 for P/V operands, FP32 accumulation,
and BF16 final output, matching the permitted parent paths. Quantization of
queries and cache updates is performed by the unchanged parent kernel.

NVIDIA PTX primary reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

The FP8 m16n8k64 metadata owners alternate whole rows: even t uses 0x44444444,
odd t uses 0xEEEEEEEE. FP16 m16n8k32 packs row g in the low sixteen bits and
row g+8 in the high sixteen: 0xEEEE4444 for both selector-zero owners.

Evidence:

- `sparse_qk.log`: eight randomized shapes/seeds, zero absolute difference
  against dense FP32 products of the same FP8 inputs.
- `sparse_pv_v2.log`: eight randomized comparisons, relative error below 4e-7
  against dense FP32 products of the same FP16 inputs.
- `sparse_decode_v1.log`: full decode integration; requires reference checks
  and paired cold-cache graph timings before any performance claim.
