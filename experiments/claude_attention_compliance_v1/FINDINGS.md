# Source-level compliance audit: attention_readable_fp32_path_v1

Candidate: `experiments/attention_readable_fp32_path_v1/submission.py`
SHA-256 verified locally: `ea65a78fd87b50539b5ffbdec6b703ee6444dd393890de79a16297e1377245f8` (prefix ea65a78f, matches `manifest.json`).
Size: 155,409 bytes, 3,191 lines (limit 1 MB). Audited against `RULES.md`, `kernels/muse_attention_decode/TASK.md`, local `reference.py`.
Method: complete read of the file (lines 1-3191), grep inventories, a replay of `_plan` / `_TUNED` / `launch` selection logic for all official shapes (host logic only, no GPU, no Modal, no okc). No other file was modified.

Legend: **PROVEN** = established from source text alone. **UNKNOWN** = cannot be settled from source / needs organizer or judge answer. **NONCOMPLIANT** = violates a written rule. **RISK** = not a written violation but a policy or fragility concern.

## Verdict

- NONCOMPLIANT findings against the written rules: **none found** in the paths selected for the 12 scored loads and the 16 tests.
- This is **not an unconditional clearance.** Unknown items U1-U5 below are not resolved by source reading, and passing tests / the judge `checks.ok` flag does not resolve them (RULES.md line 10).
- The user must decide on U1/U2 (organizer-policy questions) and on the NOTES.md credit gap (R1) before a scored `okc submit`. Any edit to the file, including comment-only edits, changes the SHA and voids the "exact privately checked candidate" property (the check job passed 2.604x, CI [2.565, 2.643], `flagged: []`, `rejected: []`, job 12167249d81c46ee).

## 1. Readable source, size, no blobs, no timing manipulation

| Item | Status | Evidence |
|---|---|---|
| Readable source, no encoding | PROVEN | Max line length 161 chars, zero non-ASCII bytes, no `base64`/`zlib`/`lzma`/`marshal`/`pickle`/`ctypes`/`exec`/`eval`/`__import__` (grep). Python imports are only `typing.TypedDict` and `torch` (lines 54, 56) plus `torch.utils.cpp_extension.load_inline` inside `_build` (line 2954). |
| No prebuilt binary embedded or loaded | PROVEN | No `.so`/`cubin`/`fatbin`/PTX file loads. CUDA source is a readable raw string `CUDA_SRC` (lines 119-2913) compiled at call time by `load_inline` (lines 2956-2964, `-gencode=arch=compute_90a,code=sm_90a`). Runtime compilation of readable source is allowed by RULES.md line 7. |
| Size limit | PROVEN | 155,409 bytes < 1 MB. |
| Inline PTX is readable asm | PROVEN | Inventory: cp.async (194-238), mbarrier (213-234), bar.sync/arrive (241-249), ex2.approx (254), prmt (259), cvt e4m3<->f16 (265-272, 470-471), mma.sync f16 (294), red.global.add f32 (330-334), ldmatrix/stmatrix (338-345), wgmma fence/commit/wait (360-367), wgmma e4m3 (379, 393, 414), mapa/ld.shared::cluster (517-526), atom.acq_rel / ld.acquire (2264, 2273, 2284, 2328, 2397, 2407), griddepcontrol (443, 446), setmaxnreg (2744, 2747). No obfuscated or opaque asm. |
| No timing manipulation in code that runs | PROVEN | No `%globaltimer`/`clock64` read in code (only inside comments at 132, 179). `p.dbg` is passed as `None` by `kernel_fn` -> `_launch(data)` (lines 3187, 3071), so the `dbg` path (2369) is dead. No host sync (`.item()`, `.cpu()`, `synchronize`, `print`) in the call path (grep). `__nanosleep(64)` in spin loops (2276, 2288, 2409) is back-off, not timing manipulation. |
| No harness reads, network, subprocess in source | PROVEN for source text | grep for `subprocess`/`os.`/`sys.`/`socket`/`urllib`/`requests`/`environ`/`inspect`/`sys.modules`: no hits. `load_inline` spawns ninja/nvcc inside torch (UNKNOWN U4). |
| Third-party credit in file header | PROVEN | Line 2: "Adapted from vjeux/okc-2026 (Apache-2.0), exact public scored attention entry source." |
| Credit in NOTES.md | **NOT MET** (R1) | `grep -i 'vjeux\|okc-2026' NOTES.md` has no match (only `research/ENGINEERING.md` lines 605-663 mention vjeux). PROGRAM.md requires credit in NOTES.md and the file header. Project-procedure gap, not a judge rule. Proposed fix (not applied, other file): add a NOTES.md credits line "attention: adapted from vjeux/okc-2026, Apache-2.0; see header of experiments/attention_readable_fp32_path_v1/submission.py". |
| Trailing no-op comments | RISK (R2) | Line 3188 `# recheck 1793522192` and line 3191 `# Readable FP32-accumulation attention path; see experiment notes.` are no-ops. They only change the file hash (okc returns the earlier result for identical code, TASK.md line 117). Not timing manipulation, but disclose them. |

## 2. Precision trace (QK, PV, row sums, cross-CTA partials)

### 2.1 QK
- q_len 1 (L==1): q is RMS-normalized in FP32 (1426-1432), RoPE in FP32 (1433-1442), then rounded to **f16** pre-scaled by `sl2` (1448, 1459). K is e4m3 converted exactly to f16 (`cvt2`/`cvt2h`, 1610-1611). MMA is `mma.sync...m16n8k16.f32.f16.f16.f32` (294) with FP32 accumulator (`float se[4], so[4]`, 1607); partial sum `s = se + so` in FP32 (1613). **PROVEN FP32 accumulation.** Query rounding is f16, finer than the FP8 E4M3 TASK.md allows ("no coarser").
- q_len 8 (L==8): q rounded to e4m3 (1408-1410, 1454), which TASK.md allows; K stays e4m3 in smem; `wgmma ...m64n64k32.f32.e4m3.e4m3` (379/393), accumulator `float s[32]` (1809). **PROVEN FP32 accumulation.**
- K dequant scale folded in FP32 via `sl2 = scale_log2 * kscale` (1364); host `scale_log2 = sm_scale * qk_scale_factor * log2e` (3111).

### 2.2 P (probabilities) and PV
- L==1: exp2 via `ex2.approx.ftz.f32` in FP32 (1637; `ex2` at 252-256), then **P packed to FP16** (`pack_h2`, 1652-1653) and used as the A operand of `mma.sync.f32.f16.f16.f32` (1655); V is e4m3 converted exactly to f16 (`load_v_frags`, 304-322). Accumulator `o[0][t][e]` is FP32 (1554-1560). Row sum `ls` accumulates the **unrounded** FP32 `pv` (1635-1640). **FP32 PV accumulation PROVEN; P format is FP16 (U1).**
- L==8: P is **E4M3** (`e4m3x4f` -> `cvt.rn.satfinite.e4m3x2.f32`, 468-473, used at 1960-1963 and 2019-2022); V^T is e4m3 (1849-1869); PV is `wgmma m64n72k32 ...f32.e4m3.e4m3` (414) with FP32 accumulators `o` (32) and `lacc` (4) (1711-1717, 2006-2007, 2076). Row sums come from an e4m3 ones column in the same wgmma (1842-1847, comment 1952-1953), so numerator and denominator use the same quantized P and both are FP32. **FP32 PV accumulation PROVEN; E4M3 P matches the organizer allowance in form (U2).**
- Output rmsnorm and gate stay FP32 until the final BF16 store (e.g. 905-923, 2303-2321). `pr8` (284-286) is the identity, so the "conformance probe" rounding is not active.
- `ex2h2` (279) is defined but unused (one occurrence in the file), so no FP16-scores exp2 path exists.

### 2.3 Cross-CTA / cross-warp partials
All FP32, **PROVEN**:
- Global workspace `ws_acc` / `ws_l` are `float*` (Params 175-176; allocated float32, 2993-2995); accumulated with `red.global.add.f32` / `.v4.f32` (330-334; used 2215, 2231, 2234, 593-597).
- Final sum of copies in FP32 (`sum_copies`, 795-867; `fin_nc1`, 872-926).
- Two-key-half merge in L==1 through shared FP32 (`xs`, 2135-2154).
- DSMEM cluster partial buffers `PB`/`PL` are `float` (2162-2193, 650-660): **dead code for supported shapes** (section 3).
- No FP16/BF16/FP8 partial anywhere: the only `__half` uses are pack helpers (275, 476); BF16 appears only for qkv/gate/out and staged raw q/k/v rows.

### 2.4 Numerical-design caveats (not violations)
- C1: Splits merge after rescaling to the fixed reference 2^0: `f = ex2(min(m, 126))` (2176, 2208). Exact in FP32 only while log2-scores stay below 126. QK-norm bounds the log2 score near `64 * 0.987 * kscale ~ 63 * kscale`, so it holds for `kscale` up to about 2 (scored cases use 1, tests use 1/64). Beyond that the clamp saturates silently. PROVEN under the stated contract, UNKNOWN beyond kscale ~2.
- C2: FP32 `red.add` order is nondeterministic run to run. RULES.md line 15 discusses only routing nondeterminism. Low risk.
- C3: Observed relative L2 errors on the check run (job 12167249d81c46ee): q_len 8 cases 0.064-0.097 against a bound of 0.15 (worst row up to 0.124 against the 5x-row limit); decode cases about 0.003-0.006. The q_len 8 margin is real but not large.

## 3. Path coverage

Host selection replayed from `_plan` (3019-3068), `_TUNED` (2944-2948), `_launch` (3071-3116), `launch<L>` (2763-2856), 132 SMs. MODE numbers cross-checked against kernel names in the saved profile (`profiles/submission.12167249d81c46ee.txt`): verify1 sliding `<8,false,15>`, verify1 full `<8,false,11>`, verify16/32 sliding `<8,false,3>`, verify16/32 full `<8,true,0>` agree with the derivation. The decode (L==1) rows use the same logic but their profile blocks were not individually compared.

| Load | Plan | Kernel instance | Notes |
|---|---|---|---|
| decode 1 sliding | n_splits 32, nc 1, coop 0 | `<1,false,13>` (SPEC+ONE, plain) | B==1, P==1 speculative-length build |
| decode 1 full | n_splits 252, nc 8, coop 1 (tuned) | `<1,false,11>` (SPEC, coop) | `sum_copies` nc==8 branch (812-835) |
| decode 16 sliding | n_splits 8, nc 1, coop 0 (tuned) | `<1,false,1>` | last-CTA finalize `fin_nc1` |
| decode 16 full | flat, n_splits 264, coop 0 | `<1,true,0>` | producer-warp + TMA page-index path (TMAPG_ON, L==1) |
| decode 32 sliding | n_splits 8, nc 1 | `<1,false,1>` | |
| decode 32 full | flat | `<1,true,0>` | |
| verify 1 sliding | n_splits 33, coop 1 | `<8,false,15>` | |
| verify 1 full | n_splits 252, nc 8, coop 1 | `<8,false,11>` | |
| verify 16 sliding | n_splits 8, coop 1 | `<8,false,3>` | |
| verify 16 full | flat, 264, coop 1 | `<8,true,0>` | grid barrier `flat_finalize` (2381-2463) |
| verify 32 sliding | n_splits 4 (tuned), coop 1 | `<8,false,3>` | |
| verify 32 full | flat, 264, coop 1 | `<8,true,0>` | |

- Tests with `page_size` 64 use the same kernels via the non-`pfast` copy paths (1085-1106, 1158-1189). Tests with `heads dp` (128 q / 16 kv, pairs 8) pass `_supported` (`hq == 8*hkv`, line 3121). All 16 tests passed on the judge, which is evidence of correctness only.
- **Fallback (`_fallback`, 3140-3173) is not selected for the official cases (PROVEN via `_supported`/`_fits`, 3119-3123, 3176-3181).** Only q_len not in {1,8}, `hq != 8*hkv`, odd `hkv`, non-power-of-two page, batch > 512, or index-limit overflow route there. Official configs are tp8 (16,2) and dp (128,16), q_len 1 or 8, page 1 or 64, B <= 32. Content reviewed anyway: pure torch ops (`einsum`, `softmax`, `index_put`, `cos/sin`, `sigmoid`), no external library, no host sync, FP32 math. UNKNOWN: whether the judge's torch supports `index_put_` on `float8_e4m3fn` (3158-3159) and whether `einsum` runs TF32 (no `allow_tf32`/`set_float32_matmul_precision` in the file or the local reference/checks). Irrelevant unless a held-out unsupported shape reaches it.
- **Cluster/DSMEM code is dead for supported shapes (PROVEN):** `_plan` returns before the cluster block (3054-3068) for every window case with L in {1,8}: L==1 reqs<24 (3037), L==1 reqs>=24 (3050), L==8 reqs<16 (3041), L==8 reqs>=16 (3046). No `_TUNED` entry sets `cs`. MODE 2 instances `kn[0]`/`kn[2]` are compiled but not launched.
- PDL off (`pdl=None` -> `use_pdl=False`, 3105); the `tma` bitfield carries only `COOP_BIT`.
- External library calls in the timed path: **none.** Timed path is `_launch` -> `run` (custom kernel) plus `torch.empty` for the output (3080). No cuBLAS/FA3/Marlin/flash call. `_workspace`, `_inv_freq`, `_num_sms` are small cached constants/scratch (2933-3001).

## 4. Unknowns requiring an answer (not cleared by passing tests)

- **U1 - FP16 P at q_len 1 (lines 1652-1655).** RULES.md line 15 authorizes exp2 approximations "using FP16-rounded scores" and E4M3 P at a different power-of-two scale, "when QK, PV, and all partials remain FP32". FP16 P is not named. It is finer than E4M3 and accumulation is FP32 (proven), so it is consistent with the stated principle, but it is not enumerated. UNKNOWN whether organizers treat it as an allowed P format. Minimal correction if rejected: round P to E4M3 as the L==8 path does (kernel change, new check required, speed impact unmeasured).
- **U2 - E4M3 P scale and underflow at q_len 8 (lines 1925-1966, 2019-2022, 468-473).** P = round_e4m3(exp2(score - offset)), offset = running row max (lazy; raised only when P would exceed 256). That is scale 1, a power of two, but different from FA3's ~448. E4M3 flushes values below about 2^-10 of the row max to zero, and the ones-column row sum uses the same quantized P, so very small-weight keys contribute to neither numerator nor denominator. The organizer allowance covers different power-of-two scales "within the error bounds"; TASK.md also forbids "skipping values by their attention weight". This is quantization underflow, not a decision to skip, and observed error is 0.064-0.097 vs bound 0.15, but UNKNOWN whether organizers would object at scale 1. If they do, the natural fix is scaling P by 2^k before the E4M3 convert and un-scaling via the row sum/epilogue (kernel change, new check).
- **U3 - speculative length and per-shape tables.** `MODE & 8` builds (selected at 2812-2815) speculate `seq = page_table width - L` for B==1, page 1 (2485) and verify the real `seq_lens` before any K/V use (1292-1297, 2508, 2527), so results do not depend on the guess; only latency does. It keys off the harness page-table width convention (comment 2477-2482; `reference.py` line 184). `_TUNED` (2944-2948), `SEGK` constants (2584, 2614, 2662) and `_plan` thresholds are per-shape tuning. RULES.md line 8 favors integrable implementations over extensive per-shape tuning. Classified RISK (R3), not NONCOMPLIANT. Minimal correction if challenged: delete the `_TUNED` rows and the `ksp*` selection (2812-2815); changes timing, needs re-check.
- **U4 - `load_inline` and the "starts processes" rule.** TASK.md line 100 rejects submissions that start processes. `load_inline` invokes ninja/nvcc inside torch, not in the submission text, and RULES.md line 7 says runtime compilation is not itself prohibited. The judge's static check passed on this content (`checks.flagged = []`, `rejected = []`, job 12167249d81c46ee); that is evidence, not a rule citation. Also unknown: whether first-call compile happens inside the 15-minute budget and outside graph capture (the check run completed with status ok).
- **U5 - binding of the check job to this file.** `status.json` `content_hash` (e3e640eb...) differs from the file SHA, but it differs for all 360 experiment status files versus their manifest SHA (0 of 360 equal), so okc hashes a different representation. Binding of job 12167249d81c46ee to ea65a78f rests on `manifest.json`, not a hash match. Not resolvable from source.

## 5. Other risks (not rule violations)

- R4 - Misleading comments in the production path: line 1649 `(timing probe only, wrong results)` sits on the real L==1 PV loop (which computes correct PV); line 1801 has the same wording on a non-WS `cp_wait`; lines 282-286 "conformance sensitivity probe" on the identity `pr8`; lines 349-353 empty `PROF_*` macros; `p.dbg` timestamp plumbing (179, 2369). A reviewer could misread these as shipped-wrong or instrumented code. Proposed correction: delete or reword. Comment-only, but it changes the SHA, so the file must be re-checked.
- R5 - Persistent device state across calls: epoch/sense counters in `ws_cnt` (2260-2268, 2395-2401, offsets 30020-30022) and spin-waits that `__trap()` after 2^24 polls (131-133). Not derived from inputs (so not a result cache), but a trap or killed launch leaves the shared workspace inconsistent. Cooperative barriers need co-residency; the host disables `coop` when the grid exceeds occupancy x SMs (2801, 2816-2827). That assumes no other kernel holds SMs concurrently: UNKNOWN under the judge's stream usage.
- R6 - `_inv_freq` is memoized by `(theta, device, n_pairs)` (2984) where `theta` comes from `config`. A fixed function of config, not of activations, so likely within the rule, but strictly keyed on a config field.

## 6. Summary table

| Area | Status |
|---|---|
| Readable, unencoded, < 1 MB, no prebuilt blobs | PROVEN |
| Apache-2.0 credit in file header | PROVEN |
| Credit in NOTES.md | NOT MET (R1; fix is a NOTES.md edit, proposed, not applied) |
| No timing manipulation / harness reads / host sync | PROVEN (source level) |
| QK accumulators FP32 (both L) | PROVEN |
| PV accumulators FP32 (both L) | PROVEN |
| Row sums, cross-CTA / cross-warp partials FP32 | PROVEN |
| P format E4M3 at q_len 8 | allowed in form; scale/underflow UNKNOWN (U2) |
| P format FP16 at q_len 1 | not enumerated by organizers; UNKNOWN (U1) |
| Fallback reachable for official cases | No (PROVEN); content reviewed |
| Cluster/DSMEM code reachable | No (PROVEN) |
| External library calls in timed path | None (PROVEN) |
| Hard rule violation found | None |
| Safe to submit without further input | **Not established**: depends on U1/U2 organizer position and R1; edits to resolve R4 change the SHA |
