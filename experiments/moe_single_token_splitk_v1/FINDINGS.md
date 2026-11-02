# Single-token MoE split-K — rejected

## Hypothesis

For batch 1, only a small number of local experts are active, leaving the
current persistent kernel with little useful CTA parallelism. Split each
expert's 4096-wide down projection into 256 chunks of K=16, compute the
corresponding gate/up rows in the same CTA, then combine partial outputs with
FP32 atomics. This avoids waiting for a full gate/up intermediate before
starting the down projection.

## H200 result

Candidate source: SHA-256 `28d41c4b2ed1081343266eda6282a2fafbc78db38982c140ff8b9db1cfb48cb6`.
Parent: the publicly scored KPERM3 MoE source, SHA-256
`ac008155520981cd67928bd52bb647efa64096add5acc8edabc779261105792f`.

On an NVIDIA H200 (Modal app `ap-Mzxtrwrk7hwj1wKIiRTlds`), all six generated
batch-1 inputs passed `reference.check_implementation`. Each seed used 16
alternating CUDA-graph replays per arm with a 1 GiB cold-L2 flush outside the
timed interval. The geometric mean of the per-seed parent/candidate median
ratios was **0.82335×**: split-K was about 18% slower. Parent times were
31.0–39.7 µs; candidate times were 36.9–48.9 µs. Relative output error to the
parent was at most `1.01e-4`.

The extra partial-result traffic and synchronization cost outweighed the
additional parallelism. Reject this branch; do not spend the remaining
official MoE check on it. No scored source changed.

## Reproduction

The candidate was built from the scored parent with:

```bash
python dev/moe_splitk/build.py \
  experiments/scored_moe_ours_kperm3_stack_v1/submission.py \
  experiments/moe_single_token_splitk_v1/submission.py
```

Screen command:

```bash
modal run dev/modal/run_h200.py --problem muse_moe_w4a16 \
  --file "$PWD/experiments/moe_single_token_splitk_v1/submission.py" \
  --drv "$PWD/dev/modal/drv_moe_m1_splitk.py"
```
