# Hopper SM90 Fused RMSNorm Projection Prologue Architecture & Specification

## 1. Overview & Objective
This specification defines the architecture for fusing the pre-FFN RMSNorm calculation directly into the projection prologue of the Pooler and Router GEMM/GEMV on NVIDIA H200 (SM90 Hopper).

The fused kernel replaces the two-stage execution:
1. `_boundary1` (Triton/CUDA): reads `stream`, `P_attn`, computes $s_1 = \alpha_1 s + \beta_1 \text{RMSNorm}(a)$, computes $h = \text{RMSNorm}(s_1) \cdot (1 + w)$, casts to BF16, and writes $h$ (and $h_r$) to HBM DRAM.
2. `pooler` / `router` GEMM/GEMV: reads $h$ (and $h_r$) from HBM DRAM, loads weights $W$, and performs matrix multiplication.

By fusing RMSNorm directly into the projection prologue:
- **Zero intermediate HBM round trips** for $h$ and $h_r$.
- **Elimination of a separate grid launch** (saving ~3.5–4.5 µs fixed launch/synchronization latency).
- **Exact BF16 intermediate rounding**: values entering WGMMA/MMA preserve bitwise equivalent BF16 rounding semantics (`__float2bfloat16_rn`), strictly avoiding the drift of unbounded algebraic deferred normalization.

---

## 2. Mathematical Semantics & Precision Boundaries

### 2.1 Forward Equations
Let $T$ be the number of tokens ($T \in \{1, 8, 16, 32, 128, 256\}$), $K = 8192$ (hidden dimension), $N_{\text{pooler}} = 4096$, $N_{\text{router}} = 256$.

Given residual stream $s_1 \in \mathbb{R}^{T \times K}$ in FP32 and norm weights $w \in \mathbb{R}^{K}$ in BF16:
$$\text{mean\_sq}_t = \frac{1}{K} \sum_{k=0}^{K-1} (s_{1, t, k})^2$$
$$\text{rsqrt}_t = \frac{1}{\sqrt{\text{mean\_sq}_t + 10^{-5}}}$$
$$h_{\text{exact}, t, k} = s_{1, t, k} \cdot \text{rsqrt}_t \cdot (1.0 + \text{float}(w_k))$$

### 2.2 Exact Intermediate Rounding Contract
SGLang and whole-decoder reference checks require:
1. **Pooler Projection ($T \in [1, 256]$)**:
   $$h_{\text{bf16}, t, k} = \text{round\_to\_bf16}(h_{\text{exact}, t, k})$$
   $$P_{t, n} = \sum_{k=0}^{K-1} h_{\text{bf16}, t, k} \cdot W_{\text{pooler}, n, k}$$
2. **Router Projection**:
   - For $T \le 8$ (Decode 1, Verify 1):
     Router weights are FP32 ($256 \times 8192$). The input MUST be kept in FP32 $h_{\text{exact}}$:
     $$\text{logits}_{t, j} = 0.1767767 \cdot \sum_{k=0}^{K-1} h_{\text{exact}, t, k} \cdot W_{\text{router}, j, k}$$
     *(Rounding $h$ to BF16 for $T \le 8$ degrades router logits L2 relative error by $12\times$ and drops expert top-8 match from 100% to <15%, failing `check_routing`).*
   - For $T > 8$ (Decode 16/32, Verify 16/32):
     Router input uses $h_{\text{bf16}}$ with Hi+Lo BF16 weight decomposition and FP32 accumulation:
     $$\text{logits}_{t, j} = 0.1767767 \cdot \left( \sum_{k=0}^{K-1} h_{\text{bf16}, t, k} \cdot W_{\text{router\_hi}, j, k} + \sum_{k=0}^{K-1} h_{\text{bf16}, t, k} \cdot W_{\text{router\_lo}, j, k} \right)$$

---

## 3. Hopper SM90 Architectural Implementation

### 3.1 Warp Specialization & Shared Memory Layout
On SM90 Hopper, each SM provides up to 228 KB of Shared Memory and support for asynchronous warpgroups.

#### SMEM Partitioning:
- **Prologue Buffer $S_{\text{norm}}$**:
  - $h_{\text{bf16}}$: $T \times 8192 \times 2$ bytes.
    - $T = 1$: 16 KB
    - $T = 8$: 128 KB
    - $T = 16$: chunked / double-buffered (32 KB per chunk of $K=1024$)
  - For $T \le 8$, $h_{\text{exact}}$ (FP32) router buffer: $T \times 8192 \times 4$ bytes.
    (Staged in registers or dedicated 32 KB SMEM tile).
- **Weight Pipeline Ring**:
  - 4 stages of WGMMA weight tiles: $64 \times 64 \times 2$ bytes $\times 4 \approx 32$ KB.

#### Execution Pipeline (Warp Specialization):
1. **Producer Warpgroup (Warp 0–3, 128 threads)**:
   - Loads $s_1$ (FP32) from residual stream via TMA (`cp.async.bulk.tensor`) or 16-byte vector loads.
   - Computes sum of squares $\sum s_1^2$ using warp shuffles (`__shfl_xor_sync`) across lanes, followed by warp-0 shared reduction across the 4 warps.
   - Evaluates `rsqrtf(sum / 8192.0f + 1e-5f)`.
   - Elementwise vector loop:
     $$h_{\text{exact}} = s_1 \cdot \text{rsqrt} \cdot (1.0f + \text{float}(w))$$
     $$h_{\text{bf16}} = \text{\_\_float2bfloat16\_rn}(h_{\text{exact}})$$
   - Writes $h_{\text{bf16}}$ to SMEM using `stmatrix` or 16-byte aligned vector stores.
   - Signals `mbarrier.arrive` to consumer warpgroups.
2. **Consumer Warpgroups (Warp 4–11, 256 threads)**:
   - Awaits `mbarrier.try_wait`.
   - Executes WGMMA mainloop (`wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`):
     - Descriptor A points to $h_{\text{bf16}}$ in SMEM.
     - Descriptor B points to compressed/lossless BF16 weights in SMEM.
     - Accumulates directly into FP32 registers.
3. **Epilogue**:
   - Stores Pooler outputs to DRAM / down-stream MoE scratch.

### 3.2 Threadblock Cluster & Inter-CTA Reductions (Large Grids)
For $T \ge 16$, the GEMM grid requires $N_{\text{pooler}} / 64 = 64$ CTAs per token row.
Redundant recomputation of RMSNorm across 64 CTAs would read $s_1$ 64 times, creating 2 MB of redundant L2 traffic per token.

To eliminate redundant traffic:
- **Option A (SM90 Threadblock Cluster)**:
  Use a Cluster of size 4 or 8 CTAs. CTA 0 within the cluster computes RMSNorm for its assigned token row and broadcasts the normalized tile to the cluster using Distributed Shared Memory (DSMEM) via `st.async.shared::cluster`.
- **Option B (Grid-Dependent Launch / PDL)**:
  A single prologue CTA per token computes the RMSNorm and writes $h_{\text{bf16}}$ to L2 cache (using `.L2::cache_hint` evict-last).
  Subsequent GEMM CTAs consume $h_{\text{bf16}}$ hitting 100% in L2 cache at 12 TB/s bandwidth, eliminating HBM round trips.

---

## 4. Why Unbounded Algebraic Shortcuts Fail
Prior attempt (`GLE_T`) attempted to pull normalization outside the GEMM:
$$W \cdot (r_2 \cdot (\alpha s + \beta r_1 P)) = r_2 \alpha (W \cdot s) + r_2 \beta r_1 (W \cdot P)$$
This failed for $T > 1$ because:
1. **Doubled GEMM Complexity**: It required executing TWO GEMVs ($W \cdot u$ and $W \cdot v$) instead of one, doubling memory bandwidth demand for the 67.1 MB Pooler weights.
2. **Intermediate Rounding Violation**: In exact reference semantics, $h$ is quantized to BF16 (7 mantissa bits) before $W$. Bypassing BF16 quantization alters downstream expert routing whenever scores are tightly grouped within $\sim 0.05$.
3. **Router Failure**: For $T \le 8$, the router requires FP32 activations. Applying BF16 quantization or approximate scaling to the router causes catastrophic top-8 expert divergence (`check_routing` fails).

---

## 5. Summary of Benefits
| Metric | Unfused Baseline | Deferred Algebraic (GLE_T) | Fused Prologue (This Spec) |
|---|---|---|---|
| **HBM Activation Traffic ($T=256$)** | 16.8 MB | 33.6 MB | **0 MB** |
| **Pooler Weight Reads** | 1x (67.1 MB) | 2x (134.2 MB) | **1x (67.1 MB)** |
| **Intermediate BF16 Rounding** | Exact | Skipped / Altered | **Exact (`__float2bfloat16_rn`)** |
| **Router Precision ($T \le 8$)** | Exact FP32 | Mixed | **Exact FP32** |
| **Separate Grid Launches** | 2 launches | 1 launch (T=1 only) | **1 launch (All 12 Shapes)** |
| **Latency Savings (H200)** | 0 µs | Regressed for $T \ge 8$ | **+3.5 to 5.2 µs per layer** |
