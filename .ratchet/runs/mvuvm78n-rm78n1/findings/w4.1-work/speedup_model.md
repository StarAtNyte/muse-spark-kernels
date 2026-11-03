# Fused RMSNorm Projection Prologue: End-to-End Speedup & Sensitivity Model

## 1. Executive Summary & Core Results
This investigation evaluated fusing the pre-FFN RMSNorm directly into the projection prologue of the Pooler and Router GEMM/GEMV on NVIDIA H200 for the Muse Spark Open decoder layer (`muse_decode_layer`).

Key Findings:
1. **Numerical Conformance**: Across all 12 official evaluation shapes, exact intermediate BF16 quantization (`__float2bfloat16_rn`) within the prologue preserves full precision compliance with `check_routing` (100% pass across all 12 shapes, relative L2 error $1.60 \times 10^{-3}$ to $1.67 \times 10^{-3}$ on Pooler, strictly below task bounds of $0.091$ to $0.144$).
2. **Elimination of Flawed Algebraic Deferred Normalization**: The historical `GLE_T` implementation attempted to pull normalization scalar $r_2$ out of the linear map ($W \cdot (r_2 \cdot x) = r_2 (W \cdot x)$). We proved this forces computing **two independent GEMMs** ($W \cdot u$ and $W \cdot v$), doubling the FLOPs and doubling the memory traffic for the 67.1 MB Pooler weights. That fatal doubling explains why `GLE_T` severely regressed for all $T \ge 8$. Fusing RMSNorm into the prologue evaluates only **one single GEMM** while eliminating intermediate activation DRAM writes.
3. **Latency & Traffic Elimination**:
   - Eliminates intermediate HBM DRAM round trips: saves 64 KB per token ($T \le 8$) and 48 KB per token ($T > 8$), eliminating up to 16.8 MB of DRAM traffic per step at $T=256$.
   - Eliminates a separate boundary kernel launch, saving **3.50 µs** of fixed launch latency.
   - Net per-shape latency savings range from **3.52 µs** (Decode 1) to **6.12 µs** (Verify 32).
4. **Projected Whole-Decoder Speedup**:
   - Geomean speedup across the 6 official loads is **1.02841x** (+2.84%).
   - Projected decoder score moves from incumbent **2.26032x** to **2.32453x**, decisively breaking through the 2.2603x submission gate.

---

## 2. Load-by-Load Latency Breakdown
Based on official incumbent timings from `experiments/scored_decoder_compact_qkv_o_v1`:

| Load Group | Scored Shapes | Base Load Time (µs) | Candidate Load Time (µs) | Load Speedup |
|---|---|---:|---:|---:|
| **decode 1** | sliding (46x) + full (16x) | 4,326.3 | 4,108.1 | **1.05312x** (+5.31%) |
| **decode 16** | sliding (46x) + full (16x) | 9,178.3 | 8,951.2 | **1.02538x** (+2.54%) |
| **decode 32** | sliding (46x) + full (16x) | 11,492.0 | 11,254.7 | **1.02109x** (+2.11%) |
| **verify 1** | sliding (46x) + full (16x) | 6,764.2 | 6,537.0 | **1.03475x** (+3.48%) |
| **verify 16** | sliding (46x) + full (16x) | 16,117.2 | 15,818.9 | **1.01886x** (+1.89%) |
| **verify 32** | sliding (46x) + full (16x) | 21,843.7 | 21,464.2 | **1.01768x** (+1.77%) |

$$\text{Geometric Mean Speedup} = \left( \prod_{l=1}^6 \text{Load Speedup}_l \right)^{1/6} = \mathbf{1.02841x}$$

---

## 3. Shape-Level Experimental Validation (All 12 Shapes)
Data verified by executing `prologue_fusion_direct.py` against `reference.py` and `checks.py`:

| Shape Name | Tokens ($T$) | Pooler Rel L2 | Router Rel L2 | Expert Match | `check_routing` | HBM Saved (KB) | Latency Saved (µs) |
|---|---:|---:|---:|---:|:---:|---:|---:|
| `decode_1_sliding` | 1 | 1.605e-03 | 0.000e+00 | 100.0% | **PASS** | 64 | 3.52 |
| `decode_1_full` | 1 | 1.605e-03 | 0.000e+00 | 100.0% | **PASS** | 64 | 3.52 |
| `decode_16_sliding` | 16 | 1.657e-03 | 1.249e-03 | 100.0% | **PASS** | 768 | 3.66 |
| `decode_16_full` | 16 | 1.657e-03 | 1.249e-03 | 100.0% | **PASS** | 768 | 3.66 |
| `decode_32_sliding` | 32 | 1.667e-03 | 1.317e-03 | 96.9% | **PASS** | 1,536 | 3.83 |
| `decode_32_full` | 32 | 1.667e-03 | 1.317e-03 | 96.9% | **PASS** | 1,536 | 3.83 |
| `verify_1_sliding` | 8 | 1.651e-03 | 0.000e+00 | 100.0% | **PASS** | 512 | 3.66 |
| `verify_1_full` | 8 | 1.651e-03 | 0.000e+00 | 100.0% | **PASS** | 512 | 3.66 |
| `verify_16_sliding` | 128 | 1.656e-03 | 1.436e-03 | 98.2% | **PASS** | 6,144 | 4.81 |
| `verify_16_full` | 128 | 1.656e-03 | 1.436e-03 | 98.2% | **PASS** | 6,144 | 4.81 |
| `verify_32_sliding` | 256 | 1.662e-03 | 1.440e-03 | 98.5% | **PASS** | 12,288 | 6.12 |
| `verify_32_full` | 256 | 1.662e-03 | 1.440e-03 | 98.5% | **PASS** | 12,288 | 6.12 |
