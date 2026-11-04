# attn_b1_v2: batch-1 attention latency (diagnostic, no winning candidate)

Parent: experiments/attention_readable_fp32_path_v1/submission.py (sha256 ea65a78f..., copied here as parent.py).
Every timing is Modal H200 in the judge image, cold-L2 CUDA-graph replays, order rotated each rep, every graph
reference-checked. Event times include about 4.5 us of fixed graph cost. Noise between paired runs is about +-1%.

## 1. Floor math (B=1, Hq=16, Hkv=2, d=64, page 1, FP8 KV)

Bytes per key = 256 (a 128 B K row plus a 128 B V row). Sliding: 2048 keys = 0.52 MB, 0.11 us at 4.8 TB/s.
Full: 80000 keys = 20.5 MB + 0.32 MB page table, 4.33 us (about 5.5 us at a realistic 3.8 TB/s for random 128 B rows).
Per-SM gather is about 120 GB/s, so 16 KB per CTA = 0.14 us. On sliding the bandwidth is irrelevant and the chain is everything.

Dependency chain after a CTA starts (cold L2: DRAM about 0.7 us, L2 round trip about 0.3 us):
seq_lens || page table (parent speculates seq, 1 DRAM trip) -> KV gather (1 DRAM trip) -> QK/softmax/PV of one tile
(about 0.15 us ideal) -> cross-CTA merge -> rms*sigmoid(gate) -> store.
- Global merge minimum: reds -> gpu-scope release fence -> ticket atomic round trip -> L2 load of the sums. Measured fence alone: 0.6 us.
- Floor, sliding: 0.7 + 0.7 + 0.15 + (0.6 + 0.3 + 0.3) + 0.1 = about 2.9 us after CTA start with a global merge
  (about 2.0 us with DSMEM, but see below). Parent: 4.0 us (decode), 5.9 us (verify).
- Floor, full: max(chain, bytes/BW) gives about 1.5 us ramp + 5.5 us stream + 1.2 us merge = about 8 us. Parent: 9.8 us (decode), 13.2 us (verify).
- Split choice: with a global merge the only cost of more CTAs is red traffic. One 64-key tile per CTA (32-33 CTAs) is
  already the chain-minimizing split for sliding. The sweep confirms it: fewer splits lose 12-20%.

## 2. Measured per-phase breakdown (us from the first CTA start, median CTA; r1_stamp.log, r4_probe.log)

| phase | d1s | v1s | d1f | v1f |
|---|---|---|---|---|
| page indices landed | 0.74 | 0.83 | 0.99 | 1.18 |
| q group landed / KV issued | 0.96 | 1.02 | 1.79 | 1.73 |
| first tile landed | 1.52 | 1.89 | 2.69 | 2.88 |
| merge start (max CTA) | 2.08 (2.29) | 2.72 (3.34) | 6.62 (7.36) | 8.1 (10.4) |
| reds issued | 2.30 | 3.58 | 6.91 | 9.06 |
| ticket atomic returned | 3.30 | 4.38 | 8.43 | - |
| coop barrier released | - | 5.06 | 9.12 | 12.2 |
| last CTA end | 4.10 | 5.98 | 10.1 | 13.2 |

Probe facts:
- `fence.acq_rel.gpu` costs 0.6 us even with no reds outstanding (probe 1 vs 3). The relaxed atomic issues in about 0.03 us.
- At q_len 8 the per-thread `red.global.add.v4` issue takes 0.86 us (about 1 address/clk/SM). Skipping the reds
  makes v1s 1.04 us faster end to end, so that is the largest single removable item.

## 3. Variants tried (paired gain vs parent; >1 = faster)

| variant | d1s | v1s | d1f | v1f | verdict |
|---|---|---|---|---|---|
| clusters cs16 / cs8 (existing DSMEM path) | 0.67 / 0.62 | 0.67 / 0.64 | - | - | rejected: cluster launch costs about +4.4 us |
| n_splits 16/24 (sliding); 132/200/264 (full) | 0.83-0.85 | 0.79-0.89 | 0.92-0.99 | 0.90-0.97 | the tuned plan is optimal |
| nc 2/4 (sliding), 4/16 (full); coop on/off | 0.94-0.98 | 0.98 | 0.92-0.97 | 0.80-0.99 | rejected |
| PDL | 1.00 | | | | no effect, no preceding kernel |
| poll nanosleep 0 (SPIN 0) | 0.98-1.02 | 1.00-1.01 | 1.00-1.013 | 1.004-1.013 | marginal, below the 3% bar |
| poll nanosleep 16 / 256 | ~1.00 / 0.99 | | | | rejected |
| L2 prefetch of workspace + ticket lines at entry | 0.98-1.02 | 0.98-0.99 | 0.99-1.00 | 1.003 | rejected: the workspace is not the DRAM-cold bottleneck |
| TMA bulk reduction (cp.reduce.async.bulk) of the smem-staged partial instead of reds | 0.96-0.98 | 0.94-0.95 | 1.003-1.016 | 0.965-0.97 | rejected: wait_group completion is slower than reds + fence. Correct on t1/t1e/t1f/t9 |

## 4. What is left (structural, not attempted)

- q_len 8: split each tile's CTA by KV head (one warpgroup, 64 rows per CTA, 66 CTAs). That halves both red
  volume (0.86 -> 0.43 us) and per-CTA compute (0.83 us). KV re-reads are free (0.5 MB). This is the biggest measured lever.
- Full layers: the ramp to the first KV issue (1.8 us vs 1.0 us on sliding) and the coop tail (last merge start -> end, 2.8 us).
- DSMEM merge would save about 0.9 us of fence + round trips, but only with a cluster launch that does not cost 4 us.
  Worth a minimal standalone test before building on it.

Files: make_cand.py (spin/wspf/bulk generators), make_probe.py, drv_sweep.py, drv_ab.py, drv_probe.py, r*.log.

## 5. Round 6: row-split replicas at q_len 8 (make_rs.py -> cand_rs.py, real build switch, no env)

Each tile's CTA is launched RS times (grid z = pairs x RS). The replicas do the same tile work, but only replica r's
warps add their rows to the workspace, so each SM issues 1/RS of the reds. The ticket targets n_keys x RS, and the
coop finalize slices over splits x RS. All checks pass: v1s, v1f, t9, tv8 (edges, page 64), tv8k (kv_scale 1/64),
t1, v16f, v32f.

| variant | v1s | v1f |
|---|---|---|
| rs 2 (default for B=1 q_len 8 sliding) | 0.996-1.000 | - |
| rs 4 / rs 8 | 0.95-0.97 / 0.85-0.87 | - |
| rs 2 (504 CTAs, non-coop) / rs 2 n126 / rs 4 n66 | - | 0.49 / 0.76 / 0.53 |
| shapes left at rs 1 (d1s, d1f, v1f, t1, t9) | 0.97-1.00 | 0.98-0.99 |
| v16f / v32f (flat, untouched) | 0.996-0.999 / 0.999-1.000 | |

Conclusion: halving the red issue per SM buys nothing. The cost seen in probe 1 (no reds: v1s -1.04 us) is the total
L2 reduction work (33 CTAs x 32 KB = 1 MB of FP32 adds), which the release fence waits for, not the per-SM issue rate.
This matches the other results: spreading over nc copies didn't help, and the TMA bulk reduction was slower.
A head split (66 CTAs x 64 rows) reduces the same 1 MB in total, so it cannot help either; it only relieves per-CTA
compute contention, and rs 2 shows that is not where the time goes. Duplicated compute (rs on full layers) is much worse.
The only lever for the q_len 8 merge is fewer partial bytes reduced in L2:
- pre-reduce in the SM (2 tiles per CTA, one per warpgroup, smem-merged before the reds), or
- a DSMEM cluster merge with a cheap launch.

Side finding: even the inert rs plumbing (an integer division on blockIdx.z and the extra branches on the once path)
costs 1-3% on the untouched B=1 shapes (t9 -3%). The once-path code of this kernel is very latency-sensitive, so any
B=1 rewrite should be a separate, compact kernel.
No submission.py written: no variant reached the 2% bar on v1s/v1f.
