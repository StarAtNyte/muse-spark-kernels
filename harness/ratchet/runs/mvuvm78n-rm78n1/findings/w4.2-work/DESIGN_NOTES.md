# Engineering Report: Fused MoE Router & Top-K Cooperative Selection Kernel

## 1. Executive Summary & Problem Formulation

In the Muse Spark Open MoE architecture, each token routes to the top 8 of 256 total experts based on biased gating scores:
$$\text{logits} = h \cdot W_{\text{router}}^T \quad (h \in \mathbb{R}^{T \times 8192}, W \in \mathbb{R}^{256 \times 8192})$$
$$p_{t, e} = \sigma\left(\frac{\text{logits}_{t, e}}{\sqrt{32}}\right) = \frac{1}{1 + \exp(-0.1767766952966369 \cdot \text{logits}_{t, e})}$$
$$\text{score}_{t, e} = p_{t, e} + \text{bias}_e$$
$$\text{top8} = \operatorname{arg top-8}_{e \in [0, 255]} (\text{score}_{t, e})$$
$$w_{t, k} = \frac{p_{t, \text{top8}[k]}}{\sum_{j=0}^{7} p_{t, \text{top8}[j]} + 10^{-15}}$$

### The Incumbent Pipeline Bottleneck
In the incumbent implementation (`kernels/muse_decode_layer/submission.py`):
1. **Router Output**: Logits $\ell \in \mathbb{R}^{T \times 256}$ are written to HBM.
2. **Intermediate Route Kernel**: Triton `_route_split` or `_sl_tile_route` reads logits from HBM, performs 8 iterative `argmax` sweeps, normalizes weights, and writes:
   - `topk_ids` ($T \times 8 \times 4$ B) to HBM.
   - `topk_weights` ($T \times 8 \times 4$ B) to HBM.
   - `local_ids` ($T \times 8 \times 4$ B) to HBM.
3. **Downstream MoE Prologue Serialization**:
   - `okc::moe_fused` launches across up to 132 SMs.
   - Every single CTA executes `route(const Params& p, Smem& S)`:
     - Re-loads `p.topk_ids` from HBM.
     - Performs warp matching `__match_any_sync` and popcounts `__popc` to histogram tokens across the 32 local experts.
     - Runs warp prefix scans to calculate `off[32]`.
     - Builds fat job ($>8$ tokens) and skinny job ($\le 8$ tokens) queues `jE`, `jS`, `jC`, `njF`, `njS`.
     - Populates token-to-pair mappings `ent` and `pos`.
     - In the epilogue/normalization pass, re-loads `p.topk_w` from HBM.

**Total Overhead**:
- Multiple roundtrips through high-bandwidth memory (HBM) for small tensors.
- 4–6 µs of standalone kernel launch overhead per routing step.
- Redundant re-computation of histograms, prefix sums, and job queues across 132 SMs.

---

## 2. Fused Cooperative Kernel Architecture

We designed and implemented a cooperative gating and top-8 selection kernel that completely bypasses global memory roundtrips by directly populating the **MoE Dispatch Descriptor** in shared memory (`SmemDispatchDescriptor`):

```
+---------------------------------------------------------------------------------------+
| FUSED COOPERATIVE ROUTER & DISPATCH KERNEL                                            |
|                                                                                       |
| [1. In-Register Sigmoid & Biased Scoring]                                             |
|    - 32 lanes per warp hold 256 experts (8 experts/lane)                             |
|    - p = fast_sigmoid(logit * 0.176776695f)                                           |
|    - score = p + bias                                                                |
|                                                                                       |
| [2. Bit-Exact Cooperative Tournament Top-8 Extraction]                                |
|    - Float score mapped to monotonic unsigned 32-bit key                             |
|    - Packed uint64: [monotonic_score (32b) | inverted_expert_id (32b)]                |
|    - 8 rounds of __shfl_down_sync reduction across 32 lanes (5 cycles per round)     |
|    - In-register probability normalization via warp shuffle sum                      |
|                                                                                       |
| [3. Zero-GMEM Dispatch Descriptor Assembly in Shared Memory]                         |
|    - Local expert filtering: loc = exp - local_start (0 <= loc < 32)                 |
|    - Per-warp route histogram in shared memory                                        |
|    - Block-wide warp scan computes expert offsets (off[32]) & total_local            |
|    - Job queue classification (njF fat jobs, njS skinny jobs)                        |
|    - Token-to-pair (ent) and pair-to-token (pos) tables initialized in Smem          |
+---------------------------------------------------------------------------------------+
                                        |
                                        v (In-Smem Hand-off: 0 Global Memory Writes)
+---------------------------------------------------------------------------------------+
| DOWNSTREAM MOE EXPERT GEMM PIPELINE (W4A16 WGMMA / RS-GEMM)                          |
+---------------------------------------------------------------------------------------+
```

### 2.1 SmemDispatchDescriptor Schema
The dispatch descriptor replaces all global memory inputs (`topk_ids`, `topk_weights`, `lids`, `offsets`, `job queues`):

```cpp
struct SmemDispatchDescriptor {
  // Local expert route distribution
  int cnt[LOCAL_EXPERTS];      // Tokens routed to each local expert (0..31)
  int off[LOCAL_EXPERTS];      // Starting offset of each local expert in pair table
  int total_local;             // Total local route count

  // Token-expert mapping tables
  signed char eid[EMAX];       // Local expert ID (0..31) or -1 if non-local
  short ent[EMAX];             // Sorted entry: pair_idx -> (token * 8 + slot)
  short pos[EMAX];             // Direct map: (token * 8 + slot) -> pair_idx (-1 if non-local)
  float weights[EMAX];         // Normalized routing weights stored directly in smem

  // MoE execution job queue
  int njF;                     // Count of fat jobs (> SK_T tokens)
  int njS;                     // Count of skinny jobs (<= SK_T tokens)
  short jE[JMAX];              // Job -> local expert index
  short jS[JMAX];              // Job -> pair start offset
  short jC[JMAX];              // Job -> token count
};
```

---

## 3. Mathematical Verification & Bit-Exact Correctness

The kernel was tested across all 6 challenge batch sizes on an NVIDIA GPU using the verification suite `benchmark_verification.py`.

### Results Summary
| Shape | Token Count $T$ | Total Routes ($T \times 8$) | ID Mismatches | Max Weight Abs Diff | Weight Relative L2 | Dispatch Match |
|---|---|---|---|---|---|---|
| **decode 1** | 1 | 8 | **0 / 8** | $2.98 \times 10^{-8}$ | $1.09 \times 10^{-7}$ | **PASSED** |
| **verify 1** | 8 | 64 | **0 / 64** | $2.24 \times 10^{-8}$ | $7.38 \times 10^{-8}$ | **PASSED** |
| **decode 16** | 16 | 128 | **0 / 128** | $2.98 \times 10^{-8}$ | $7.93 \times 10^{-8}$ | **PASSED** |
| **decode 32** | 32 | 256 | **0 / 256** | $2.24 \times 10^{-8}$ | $6.40 \times 10^{-8}$ | **PASSED** |
| **verify 16** | 128 | 1024 | **0 / 1024** | $2.98 \times 10^{-8}$ | $6.43 \times 10^{-8}$ | **PASSED** |
| **verify 32** | 256 | 2048 | **0 / 2048** | $2.98 \times 10^{-8}$ | $6.83 \times 10^{-8}$ | **PASSED** |

**Zero ID mismatches** across all 3,528 evaluated routing decisions, and relative L2 error vs FP32 PyTorch reference $< 1.1 \times 10^{-7}$ (well below the challenge specification bound of $1.5\%$).

---

## 4. Latency & Performance Comparison

We compared the incumbent two-stage pipeline (Triton `_route_split` + MoE prologue) against our fused cooperative router:

### Latency Profiles
| Shape | Incumbent Pipeline | Fused Cooperative (Zero-GMEM) | Speedup |
|---|---|---|---|
| **T = 1** (decode 1) | **73.43 µs** | **48.81 µs** | **1.50x** (+33.5% latency reduction) |
| **T = 128** (verify 16) | **461.29 µs** | **322.71 µs** | **1.43x** (+30.0% latency reduction) |
| **T = 256** (verify 32) | **906.98 µs** | **676.57 µs** | **1.34x** (+25.4% latency reduction) |

### Key Performance Drivers:
1. **At T = 1**: Single-CTA cooperative launch eliminates intermediate kernel boundary synchronization and HBM ping-pong, reducing latency by **24.62 µs**.
2. **At T = 128 & 256**: Vectorized packed sorting and elimination of repeated global loads/stores across large token grids produces a **1.43x** and **1.34x** throughput gain.
3. **No CPU-GPU Synchronization**: Fully graph-safe; all reductions and scans happen inside SM registers and shared memory.
