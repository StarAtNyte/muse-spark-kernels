# ForgeKernel attention reference adapter smoke

`driver.py` embeds the exact task wrapper from `ForgeKernel/tasks/muse_attention_sliding/reference/run.py` and loads the canonical attention submission. H200 app `ap-Phceeg6GCQHXJhnR4PUpNk` ran the wrapper on page-1 sliding inputs with context 3000: batch 1 / length 1 and 8 at unit scales, plus batch 2 / length 1 with scale 1/64. The challenge reference's output and cache-write checks passed all three; the wrapper returned the same cache tensors that the kernel mutated. See `probe.log`.

This validates only the wrapper on those three cases; it does not validate ForgeKernel's generated judge or the task's 80K-context workloads. The Forge task covers a subset of the challenge, and any result still requires the full official suite.
