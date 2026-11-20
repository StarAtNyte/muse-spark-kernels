# Intel Report & Mathematical Derivation of Overall Scoring

## 1. Overall Rank Mechanism Derivation

From `kernels/muse_decode_layer/TASK.md`:
> "Your best run is your row on the board, which is on the site too, at https://openkernelchallenge.dev/problems/muse_decode_layer. It also counts toward the overall kernel board (okc leaderboard), which ranks by how much your best runs together would speed up a decode step."

### Mathematical Proof
1. A single decode step under TP8/EP8 executes:
   - Dense Projections (`qkv_gate`, `o`, `router`, `pooler`, `unpooler`)
   - Gated GQA Attention
   - Routed Experts (Top-8 of 256 NVFP4 experts)
   - RMSNorms and residual additions
2. SGLang baseline benchmark times:
   - Projections: $B_{\text{proj}} = 4,830.73\ \mu\text{s}$
   - Attention: $B_{\text{attn}} = 3,766.33\ \mu\text{s}$
   - MoE: $B_{\text{moe}} = 13,448.54\ \mu\text{s}$
   - Whole Decoder Layer: $B_{\text{dec}} = 22,620.25\ \mu\text{s}$
   - Non-component residual overhead: $B_{\text{dec}} - (B_{\text{proj}} + B_{\text{attn}} + B_{\text{moe}}) = 574.65\ \mu\text{s}$ (2.54% of baseline).
3. Component Shares of Total Decode Step:
   - MoE: **59.45%**
   - Projections: **21.36%**
   - Attention: **16.65%**
   - Glue / Residuals: **2.54%**
4. Platform Overall Scoring Rule:
   - If a participant submits an integrated `muse_decode_layer`, its measured speedup $S_{\text{dec}} = B_{\text{dec}} / T_{\text{dec}}$ is compared with the synthetic speedup:
     $$S_{\text{synth}} = \frac{B_{\text{dec}}}{574.65 + \frac{B_{\text{proj}}}{S_{\text{proj}}} + \frac{B_{\text{attn}}}{S_{\text{attn}}} + \frac{B_{\text{moe}}}{S_{\text{moe}}}}$$
   - Overall board uses $\max(S_{\text{dec}}, S_{\text{synth}})$.
   - For StarAtNyte:
     $$T_{\text{synth}} = 574.65 + \frac{4830.73}{1.5395} + \frac{3766.33}{2.6177} + \frac{13448.54}{2.3820} = 10,797.20\ \mu\text{s} \implies S_{\text{synth}} = 2.0950\text{x}$$
     $$T_{\text{dec}} = \frac{22620.25}{2.2603} = 10,007.56\ \mu\text{s} \implies S_{\text{dec}} = \mathbf{2.2603\text{x}}$$
   - Because $2.2603 > 2.0950$, our overall rank is strictly governed by `muse_decode_layer`.
   - Inspection of all 1,680 rows in `live_overall.json` confirms that all top 365 positions match the descending order of `muse_decode_layer`.

---

## 2. Marginal Sensitivity & Overall Rank Gain

Let $T_{\text{dec}} = 10,007.56\ \mu\text{s}$, $B_{\text{dec}} = 22,620.25\ \mu\text{s}$.
$$\frac{\partial S_{\text{dec}}}{\partial T_{\text{dec}}} = -\frac{B_{\text{dec}}}{T_{\text{dec}}^2} = -0.00022585\ \mu\text{s}^{-1}$$

- For MoE:
  $$\Delta T = \frac{B_{\text{moe}}}{S_{\text{moe}}} - \frac{B_{\text{moe}}}{S_{\text{moe}} + 0.01} = \frac{13448.54}{2.3820} - \frac{13448.54}{2.3920} = 23.60\ \mu\text{s}$$
  $$\Delta S_{\text{dec}} = 23.60 \times 0.00022585 = +0.00534\text{x}$$
  Overall rank moves from #28 to #26 (+2 ranks).
- For Projections:
  $$\Delta T = \frac{4830.73}{1.5395} - \frac{4830.73}{1.5495} = 20.25\ \mu\text{s}$$
  $$\Delta S_{\text{dec}} = 20.25 \times 0.00022585 = +0.00458\text{x}$$
  Overall rank moves from #28 to #26 (+2 ranks).
- For Attention:
  $$\Delta T = \frac{3766.33}{2.6177} - \frac{3766.33}{2.6277} = 5.48\ \mu\text{s}$$
  $$\Delta S_{\text{dec}} = 5.48 \times 0.00022585 = +0.00124\text{x}$$
  Overall rank moves from #28 to #27 (+1 rank).
- For Whole Decoder:
  $$\Delta S_{\text{dec}} = +0.01000\text{x}$$
  Overall rank moves from #28 to #24 (+4 ranks).
- For Prefill:
  Prefill is evaluated on a completely separate board (`muse_prefill_layer`) and does NOT enter the decode step calculation.
  $$\frac{\partial S_{\text{overall}}}{\partial S_{\text{prefill}}} = 0.0000$$

---

## 3. Leaderboard Intel & Case Gaps

### Projections (`naklecha`, 1.6498x vs StarAtNyte 1.5395x)
- SGLang baseline: 4,830.7 µs
- `naklecha`: 2,928.1 µs
- `StarAtNyte`: 3,137.9 µs
- Gap: 209.8 µs
- Dominant cases: M=256 qkv_gate (18.0 µs vs 9.8 µs SOL), M=256 pooler (25.2 µs vs 17.4 µs SOL).

### Attention (`tjiang-fair`, 3.2221x vs StarAtNyte 2.6177x)
- SGLang baseline: 3,766.3 µs
- `tjiang-fair`: 1,168.9 µs
- `StarAtNyte`: 1,438.8 µs
- Gap: 269.9 µs
- Dominant cases: Verify 32 full (183.2 µs vs 126.8 µs SOL), Decode 32 full (165.9 µs vs 126.4 µs SOL).

### MoE (`dolanzhao`, 2.7578x vs StarAtNyte 2.3820x)
- SGLang baseline: 13,448.5 µs
- `dolanzhao`: 4,876.5 µs
- `StarAtNyte`: 5,645.9 µs
- Gap: 769.4 µs
- Dominant cases: bs=256 (220.0 µs vs 108.9 µs SOL), bs=128 (179.6 µs vs 101.3 µs SOL), bs=32 (132.3 µs vs 70.8 µs SOL).

### Whole Decoder (`mpeng19`, 2.4178x vs StarAtNyte 2.2603x)
- SGLang baseline: 22,620.2 µs
- `mpeng19`: 9,355.7 µs
- `StarAtNyte`: 10,007.6 µs
- Gap: 651.9 µs
- mpeng19 combines a 2.616x MoE with compact projection routing and non-blocking streams to lead the entire challenge.
