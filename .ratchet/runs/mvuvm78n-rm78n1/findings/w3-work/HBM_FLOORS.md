# HBM Floor Analysis and Projection Case Auditing

## Theoretical and Measured Bounds

The theoretical reference Speed of Light (SOL) is defined in `kernels/muse_bf16_proj/reference.py` as:
$$T_{\text{SOL}} = \max\left(\frac{Q_{\text{bytes}}}{4.8 \times 10^{12}}, \frac{F_{\text{FLOPs}}}{9.895 \times 10^{14}}\right)$$
where:
- Weight precision byte width $bw = 2$ for BF16 projections (`qkv_gate`, `o`, `pooler`, `unpooler`) and $bw = 4$ for the FP32 `router`.
- Activation byte width is 2 (BF16).
- $Q_{\text{bytes}} = N \cdot K \cdot bw + 2 \cdot M \cdot K + M \cdot N \cdot bw$
- $F_{\text{FLOPs}} = 2 \cdot M \cdot N \cdot K$

Modal cold-L2 probe benchmarks (`research/FLOORS-2026-10-29.md`, `experiments/h200_read_methods_v1`) establish:
- Contiguous TMA bulk read ceiling: 3.8–3.9 TB/s including event launch overhead; ~4.05–4.13 TB/s steady-state.
- Small stream / latency ramp: ~2.7 TB/s at 56 MB; empty CUDA graph replay fixed overhead is ~4.8–5.5 µs (the judge subtracts ~4 µs fixed cost from both candidate and baseline).

## Full Per-Case Floor and Sensitivity Table

Derived from the official published best run `beb3e26afb2544c9` (score: 1.53951x, 95% CI [1.51641, 1.56260]):

| Load | $M$ | Layer | SGLang (µs) | Yours (µs) | SOL (µs) | Gap (µs) | $Y / \text{SOL}$ | Speedup | +20% Case Gain |
|---|---|---|---|---|---|---|---|---|---|
| verify 32 | 256 | qkv_gate | 19.90 | 17.99 | 9.77 | 8.22 | 1.84 | 1.106x | +0.747% |
| verify 32 | 256 | pooler | 30.59 | 25.20 | 17.36 | 7.84 | 1.45 | 1.214x | +1.057% |
| verify 16 | 128 | qkv_gate | 17.02 | 14.33 | 8.42 | 5.91 | 1.70 | 1.188x | +0.749% |
| verify 16 | 128 | pooler | 24.78 | 20.32 | 14.64 | 5.68 | 1.39 | 1.219x | +1.074% |
| verify 32 | 256 | unpooler | 23.97 | 22.69 | 17.36 | 5.33 | 1.31 | 1.056x | +0.948% |
| verify 32 | 256 | router | 17.46 | 7.92 | 2.68 | 5.24 | 2.96 | 2.205x | +0.324% |
| verify 32 | 256 | o | 9.28 | 8.64 | 4.48 | 4.16 | 1.93 | 1.074x | +0.354% |
| verify 16 | 128 | unpooler | 19.58 | 18.32 | 14.64 | 3.68 | 1.25 | 1.069x | +0.964% |
| decode 32 | 32 | qkv_gate | 14.35 | 11.64 | 8.00 | 3.64 | 1.45 | 1.233x | +0.768% |
| verify 16 | 128 | router | 16.10 | 5.67 | 2.21 | 3.46 | 2.56 | 2.840x | +0.292% |
| decode 32 | 32 | router | 15.25 | 5.10 | 1.86 | 3.24 | 2.74 | 2.990x | +0.332% |
| verify 16 | 128 | o | 7.42 | 6.84 | 3.99 | 2.85 | 1.72 | 1.085x | +0.353% |
| verify 1 | 8 | unpooler | 7.28 | 4.52 | 1.76 | 2.76 | 2.56 | 1.611x | +0.414% |
| verify 1 | 8 | router | 26.14 | 4.24 | 1.78 | 2.46 | 2.39 | 6.165x | +0.388% |
| decode 16 | 16 | router | 14.72 | 4.08 | 1.81 | 2.27 | 2.26 | 3.608x | +0.299% |
| decode 1 | 1 | unpooler | 6.94 | 3.75 | 1.75 | 2.00 | 2.14 | 1.851x | +0.365% |
| decode 1 | 1 | router | 10.40 | 3.69 | 1.75 | 1.94 | 2.11 | 2.818x | +0.359% |
| decode 16 | 16 | qkv_gate | 13.90 | 9.87 | 7.93 | 1.94 | 1.24 | 1.408x | +0.734% |
| decode 32 | 32 | o | 6.30 | 5.30 | 3.62 | 1.68 | 1.46 | 1.189x | +0.345% |
| verify 1 | 8 | qkv_gate | 14.53 | 9.49 | 7.90 | 1.59 | 1.20 | 1.531x | +0.884% |
| decode 16 | 16 | o | 6.59 | 5.13 | 3.56 | 1.57 | 1.44 | 1.285x | +0.377% |
| verify 1 | 8 | o | 6.56 | 5.04 | 3.53 | 1.51 | 1.43 | 1.302x | +0.463% |
| decode 1 | 1 | qkv_gate | 14.16 | 9.19 | 7.87 | 1.32 | 1.17 | 1.541x | +0.911% |
| decode 1 | 1 | o | 6.46 | 4.69 | 3.50 | 1.19 | 1.34 | 1.377x | +0.458% |
| decode 32 | 32 | unpooler | 18.43 | 15.29 | 14.14 | 1.15 | 1.08 | 1.205x | +1.018% |
| decode 32 | 32 | pooler | 21.58 | 14.53 | 14.14 | 0.39 | 1.03 | 1.485x | +0.966% |
| verify 1 | 8 | pooler | 21.31 | 13.59 | 14.02 | -0.43 | 0.97 | 1.568x | +1.284% |
| decode 1 | 1 | pooler | 21.26 | 13.40 | 13.99 | -0.59 | 0.96 | 1.587x | +1.348% |
| decode 16 | 16 | unpooler | 18.59 | 13.46 | 14.06 | -0.60 | 0.96 | 1.381x | +1.011% |
| decode 16 | 16 | pooler | 21.09 | 13.44 | 14.06 | -0.62 | 0.96 | 1.569x | +1.009% |

## Key Architectural Findings

1. **Verify 32 and Verify 16 Dominance**:
   - `verify 32` ($M=256$) alone accounts for 30.79 µs of removable overhead above SOL; its speedup is only 1.228x.
   - `verify 16` ($M=128$) accounts for 21.58 µs of removable overhead; its speedup is only 1.297x.
   - These two loads are 62% of the entire problem gap across all 6 loads.

2. **The M=256 Router Double Launch Flaw**:
   - In baseline `candidate_sglang`, `router` at $M=256$ executes two separate `torch.mm` matrix products (`router_hi_t` and `router_lo_t`) and an in-place accumulation.
   - This takes 7.92 µs on the judge vs SOL of 2.68 µs.
   - Running the fused `rk_kernel` (TMA+WGMMA) reduces this case to 6.82 µs (6.1 µs kernel time), saving 1.10 µs in one launch.

3. **Decode 32 Bottleneck**:
   - `decode 32` load speedup is 1.464x.
   - `qkv_gate` at $M=32$ runs a fallback at 11.64 µs (SOL 8.00 µs); vjeux balanced `sk2` achieves 10.88 µs (-0.76 µs).
   - `unpooler` at $M=32$ runs at 15.29 µs (SOL 14.14 µs); vjeux `sk2` achieves 14.59 µs (-0.70 µs).
