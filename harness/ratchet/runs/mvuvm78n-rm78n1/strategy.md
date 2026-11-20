# Muse Spark Strategy: Roadmap to #1 Overall

**Current Standing (2026-10-30 Snapshot)**:
- **Overall Rank**: **#28** (Team: `StarAtNyte`, 1,680 ranked teams)
- **Problem Ranks**:
  - `muse_decode_layer`: **#28** (Score: **2.2603x**, CI [2.2264, 2.2942], Job `b72f02c7eb8745f5`)
  - `muse_moe_w4a16`: **#82** (Score: **2.3820x**, CI [2.3463, 2.4178], Job `190d8914bacb4aa2`)
  - `muse_attention_decode`: **#100** (Score: **2.6177x**, CI [2.5784, 2.6569], Job `b95cd80bd3e6497f`)
  - `muse_bf16_proj`: **#147** (Score: **1.5395x**, CI [1.5164, 1.5626], Job `beb3e26afb2544c9`)
  - `muse_prefill_layer`: **#115** (Score: **1.7874x**, CI [1.7606, 1.8142], Job `68ac9af85a3a4e3d`) — *Separate board; does not count toward overall rank*.

---

## 1. How Overall Rank is Computed

The okc overall leaderboard evaluates **“how much your best runs together would speed up a decode step”**.
- For any team that has submitted an integrated `muse_decode_layer`, the platform takes:
  $$\text{Overall Decode Speedup} = \max\left(S_{\text{integrated\_decoder}}, S_{\text{synthetic\_decode\_step}}\right)$$
  where $S_{\text{synthetic\_decode\_step}} = \frac{B_{\text{dec}}}{\text{Overhead} + \sum \frac{B_i}{S_i}}$.
- For `StarAtNyte`, our integrated decoder speedup is **2.2603x** ($T = 10,007.6\ \mu\text{s}$), whereas our isolated synthetic combination yields only **2.0950x** ($T = 10,797.2\ \mu\text{s}$).
- **Therefore, our Overall Rank is 100% determined by `muse_decode_layer`** (or by component transfers that accelerate the integrated decoder).
- Across all 1,680 participants, the top 365 ranks are strictly monotonic with `muse_decode_layer` speedup.

### Marginal Overall Rank Gain per +0.01x on Each Board

| Board | Baseline $B_i$ (µs) | StarAtNyte Score | Time Saved per +0.01x | Transferred Decoder $\Delta S_{\text{dec}}$ | Immediate Overall Rank Gain | Max Transfer Speedup (at Leader Score) | Overall Rank at Leader Transfer |
|---|---|---|---|---|---|---|---|
| **Whole Decoder (`muse_decode_layer`)** | 22,620.25 | **2.2603x** | 44.08 µs | **+0.0100x** | **+4 ranks** (#24) | **2.4178x** (`mpeng19`) | **#1 OVERALL** |
| **Routed MoE (`muse_moe_w4a16`)** | 13,448.54 | **2.3820x** | 23.60 µs | **+0.0053x** | **+2 ranks** (#26) | **2.4485x** (transfers `dolanzhao` 2.7578x) | **#1 OVERALL** |
| **Projections (`muse_bf16_proj`)** | 4,830.73 | **1.5395x** | 20.25 µs | **+0.0046x** | **+2 ranks** (#26) | **2.3087x** (transfers `naklecha` 1.6498x) | **#16 Overall** |
| **Attention (`muse_attention_decode`)** | 3,766.33 | **2.6177x** | 5.48 µs | **+0.0012x** | **+1 rank** (#27) | **2.3229x** (transfers `tjiang-fair` 3.2221x) | **#12 Overall** |
| **Prefill (`muse_prefill_layer`)** | 387,403.35 | **1.7874x** | 0.00 µs | **+0.0000x** | **0 ranks** (no effect) | 2.2603x (separate board) | **#28 (Unchanged)** |

> [!IMPORTANT]
> **Key Strategic Insight**: MoE constitutes **59.45%** of the decoder layer baseline time ($13,448.54\ \mu\text{s} / 22,620.25\ \mu\text{s}$). Closing the gap to leader `dolanzhao` (2.7578x) saves **769.35 µs**, boosting our integrated decoder to **2.4485x**, which **surpasses #1 overall (`mpeng19` at 2.4178x)** without touching any other component.

---

## 2. Competitive Intel & Per-Case Gap Analysis

| Competitor | Problem & Rank | Score | Primary Technique & Evidence | Critical Shapes Where They Beat Us | Removable Gap |
|---|---|---|---|---|---|
| **`mpeng19`** | Decoder **#1** (Overall **#1**) | **2.4178x** (9,355.7 µs) | Whole-layer fused dataflow: 2.616x MoE (#4) + compact projection stack + concurrent PDL stream chaining. Job: `f91b449a02764f8f`. | Verify 32 full (ours 493.9 µs), Verify 16 full (ours 349.5 µs), Decode 32 full (ours 304.8 µs). | **651.9 µs** total layer gap. |
| **`dolanzhao`** | MoE **#1** | **2.7578x** (4,876.5 µs) | Persistent WGMMA streaming near ~3.9 TB/s TMA ceiling, KPERM-style scale reuse across 4 groups, minimal lane instructions per FP4 weight. Job: `850d69d2713343c4`. | bs=256 (ours 220.0 µs vs SOL 108.9 µs), bs=128 (ours 179.6 µs vs SOL 101.3 µs), bs=32 (ours 132.3 µs vs SOL 70.8 µs). | **769.4 µs** component gap. |
| **`tjiang-fair`** | Attention **#1** | **3.2221x** (1,168.9 µs) | Coupled head-pair 128B contiguous DRAM loads (matching 32-lane warp transactions), 2-warpgroup ping-pong overlapping TMA/cp.async with WGMMA, non-cooperative fast finish. Job: `7d1f783733cf49c6`. | Verify 32 full (ours 183.2 µs vs SOL 126.8 µs), Decode 32 full (ours 165.9 µs vs SOL 126.4 µs). | **269.9 µs** component gap. |
| **`naklecha`** | Projections **#1** | **1.6498x** (2,928.1 µs) | Fused shape-dispatched SM90 MMA kernels avoiding cuBLASLt graph launch overhead on M=128/256; fused pooler/router paths. Job: `9f7fe9fede8c410d`. | Verify 32 M=256 qkv (ours 18.0 µs vs SOL 9.8 µs), pooler (ours 25.2 µs vs SOL 17.4 µs). | **209.8 µs** component gap. |
| **`zzz0906`** | Prefill **#1** | **2.0325x** (190,604 µs) | Custom grouped NVFP4 GEMM tiling bypassing Marlin BM64/32 heuristics on T=753/16384, maximizing SM90 tensor utilization. Job: `b08e5d36bce14fb2`. | First turn 16k full (ours 9,486 µs), Turn 4 full (ours 3,121 µs). | **26,137 µs** (prefill board only). |

---

## 3. Audit of Experiments

1. **Passing Check Runs vs Public Best**:
   - `moe_ours_kperm3_stack_v1`: Check speedup **2.39112x** (job `2ecdb5bf891d40e4`), officially submitted as `190d8914bacb4aa2`, scored at **2.3820x** (our current public best; run-to-run variation).
   - `prefill_split_penalty20_v1`: Check speedup **1.80396x** (job `2f13c4c92c8f4549`). Unsubmitted because split counts matched 0.02 plan (noise).
   - `prefill_vjeux_public_check_v1`: Check 1.79783x, submitted as `68ac9af85a3a4e3d` (scored 1.78740x, public best).
   - `prefill_boundary_256_hint_v1`: Check 1.79396x, submitted as `9937a79b7f8448c4` (scored 1.76757x).
2. **Promising Screens**:
   - `decoder_bigproj_screen_v1` / `decoder_alygg_bigproj_screen_v1`: Showed that standalone compact QKV and O projections beat cuBLAS at T=128/256. Successfully stacked into canonical decoder (job `b72f02c7eb8745f5` @ 2.2603x).
   - `projection_wide2x_screen_v1`: Isolated win on M=256 qkv (1.06x) and M=128 unpooler (1.08x). Check `05933bad61fc488a` passed at 1.5326x. Worth tuning and integrating into the whole decoder.

---

## 4. Quota Usage Status (Today)

*Caps reset daily at 00:00 UTC.*

| Problem | Scored Used / Cap (Left) | Checks Used / Cap (Left) | Status |
|---|---:|---:|---|
| `muse_bf16_proj` | 0 / 10 (10 left) | 0 / 20 (20 left) | Full budget available |
| `muse_attention_decode` | 0 / 10 (10 left) | 1 / 20 (19 left) | 19 checks, 10 submissions left |
| `muse_moe_w4a16` | 1 / 10 (9 left) | 2 / 20 (18 left) | 18 checks, 9 submissions left |
| `muse_decode_layer` | 2 / 10 (8 left) | 7 / 20 (13 left) | 13 checks, 8 submissions left |
| `muse_prefill_layer` | 1 / 10 (9 left) | 1 / 20 (19 left) | 19 checks, 9 submissions left |
| **Total** | **4 / 50 (46 left)** | **11 / 100 (89 left)** | **89 checks, 46 submissions available** |

---

## 5. Ranked Action Plan for the Five Kernel Researchers

### Rank 1: MoE Researcher — Push MoE WGS Streaming to 2.70x+ & Transfer to Decoder
- **Target Cases**: `bs=32` (reduce from 132.3 µs toward 100 µs), `bs=128` (reduce from 179.6 µs toward 130 µs), `bs=256` (reduce from 220.0 µs toward 150 µs).
- **Core Strategy**: Build on alygg77 Apache-2.0 KPERM3. The bottleneck is ALU instructions per weight (2.4 lane-instr/weight) and DRAM bandwidth (3.27 TB/s vs 4.1 TB/s TMA ceiling). Implement warp-specialized consumer WGMMA with K64 unrolled scale reuse (from commit `809f407`), decoupling TMA weight loads from dequant ALU.
- **Expected Gain**: +0.25x to +0.35x on MoE ($500\text{--}700\ \mu\text{s}$ saved).
- **Overall Impact**: **Reaches #1 OVERALL** (Decoder reaches **~2.42–2.45x**).

### Rank 2: Decoder Researcher — Integrate MoE KPERM3 + Compact Projections + Overlapped Routing
- **Target Cases**: `verify 32 full` (493.9 µs -> 420 µs), `verify 16 full` (349.5 µs -> 300 µs), `decode 32 full` (304.8 µs -> 260 µs).
- **Core Strategy**:
  1. Transfer the enhanced MoE KPERM3 kernel into `kernels/muse_decode_layer/submission.py`.
  2. Overlap the FP32 hi/lo router GEMV with the attention epilogue using CUDA stream event dependencies (PDL chaining).
  3. Keep the proven compact T128 QKV/O paths and fused pooler/router.
- **Expected Gain**: +0.10x to +0.15x direct decoder speedup (saving 400–600 µs).
- **Overall Impact**: **Reaches #1–#3 OVERALL** (Score **~2.36–2.42x**).

### Rank 3: Attention Researcher — Full-Context 128B Coupled Head Gathers & 2-WG Ping-Pong
- **Target Cases**: `verify 32 full` (183.2 µs -> 140 µs), `decode 32 full` (165.9 µs -> 135 µs).
- **Core Strategy**: In 80k full-context page-1 attention, DRAM random-gather latency dominates (1.65 TB/s at small batches). Implement tjiang-fair's architecture: fetch coupled 128B rows across paired KV heads, use 2-warpgroup ping-pong (one WG loading KV while the other executes WGMMA), and eliminate cooperative finalization barriers on 16/32 splits.
- **Expected Gain**: +0.40x to +0.60x on Attention ($200\text{--}250\ \mu\text{s}$ saved).
- **Overall Impact**: Transferred to decoder, yields +0.05x on decoder (**Reaches Top 10 Overall**).

### Rank 4: Projections Researcher — Replace M=128/256 cuBLASLt with Specialized SM90 MMA Kernels
- **Target Cases**: `verify 32 M=256` qkv_gate (18.0 µs -> 12 µs) and pooler (25.2 µs -> 18 µs); `verify 16 M=128` unpooler (18.3 µs -> 15 µs).
- **Core Strategy**: Port `wide2` / Nathen-style custom SM90 WGMMA kernels directly into the projection dispatcher for M=128 and M=256, eliminating the ~4 µs cuBLASLt graph launch overhead and memory roundtrips.
- **Expected Gain**: +0.08x to +0.11x on Projections ($150\text{--}200\ \mu\text{s}$ saved).
- **Overall Impact**: Transferred to decoder, yields +0.035x on decoder (**Gains ~10 ranks**).

### Rank 5: Prefill Researcher — Large-Token Grouped NVFP4 GEMM Tiling
- **Target Cases**: `turn 1 full q_len 753` (915 µs -> 750 µs), `turn 4 full q_len 753` (3,121 µs -> 2,500 µs), `first turn q_len 16384` (9,486 µs -> 7,500 µs).
- **Core Strategy**: Grouped NVFP4 Marlin currently suffers from poor occupancy under skewed routing at T=753. Implement zzz0906-style dynamic work-stealing tile schedule or fused K64 WGMMA tile.
- **Expected Gain**: +0.15x to +0.25x on `muse_prefill_layer` (**Top-50 cutoff 1.8065x reached; targets #1 at 2.0325x**).
- **Overall Impact**: **0 overall rank gain** (separate board), but critical for the November 5 serving track priority and full-model prefill throughput.
