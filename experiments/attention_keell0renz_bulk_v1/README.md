# Keell0renz TMA bulk-reduction attention transfer

Upstream: [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge), commit `8e30e1be99`; exact source SHA-256 `56c380b72001f88c1638c9315a02af4618b5af78263b1832ea6565e8676c95e4`.

Unlike the byte-identical upstream MoE, projection, prefill, and decoder files, this attention implementation differs from our incumbent. The relevant design change is `cp.reduce.async.bulk` for the split-KV merge, with a 72-float workspace row for multi-tile q_len=1 launches; the candidate also contains upstream verify-attention variants. Screen only the scored full-attention q_len=8 batch-16/32 shapes to determine whether its merge design beats our incumbent. Do not infer a gain from the upstream board row: its own lessons say the 2.7168x attention row is a high draw, with same-box measurements around 2.63–2.66x.

The bounded Modal A/B uses separate input/cache state for each kernel, validates both outputs against the task reference, then runs randomized cold-L2 CUDA-graph pairs. All source notices and license are retained.

## Results

- Full q_len=8, sliding, B16/B32, two seeds: speedups 1.0008/1.0008× and 0.9966/0.9999×. This is neutral on the scored attention workload.
- q_len=1, sliding B1/B16/B32, two seeds: 0.9784/1.0016×, 0.9988/1.0233×, 1.0192/1.0089×.
- q_len=1, full B1/B16/B32, two seeds: 0.9940/1.0102×, 1.0019/0.9986×, 1.0024/1.0041×.

The small signal is specific to sliding q_len=1 at B16/B32, outside the q_len=8 shapes used for the scored attention entrypoint. It does not establish an end-to-end score gain. The upstream implementation bundles several changes (bulk reduction, padded workspace, and cluster-finalizer changes), so the isolated follow-up in `attention_keell0renz_drycluster_remove_v1` did not explain the full signal consistently. **No candidate was promoted or submitted.**
