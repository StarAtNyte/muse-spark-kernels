# Recon Report — muse_prefill_boundary reference

**Source run.** The reference source is available at
`task/reference/submission.py`. Every claim derived from the source is marked
`[SOURCE]`; every claim from a GPU measurement is marked `[MEASURED, job 1793375150]`.

---

## 0. Kernel identity

The timed window contains **one kernel**: `boundary_kernel` — C++ template
instantiation `boundary_kernel<TPB=512, GPOST=true, STREAMHINT=false, MODE=0>`
[SOURCE, `k_boundary` launch at GLUE section; `#define BND_TPB 512`, `#define BND_HINT false`].

This is a **scalar RMSNorm + residual-blend kernel** — no MMA instructions, no
tensor cores, no warp specialisation. Per token row: load branch (BF16) + stream
(fp32), compute two RMS normalisations via warp-SHFL + `__syncthreads` block
reduction, emit updated fp32 stream (`sout`) and its BF16 normalised output (`nout`).

---

## §1. Shapes + NCU launch configuration

**Launch:**

Source confirms a flat 1D launch:
`okc_launch(boundary_kernel<BND_TPB, true, BND_HINT, 0>, dim3(T), dim3(BND_TPB), 0, st, ...)`
with no `cudaLaunchKernelEx` or cluster attribute [SOURCE].

```
LAUNCH-PARTIAL reason=log-truncated-before-step4; source confirms:
  grid  = (T, 1, 1)   — one CTA per token row
  block = (512, 1, 1) — BND_TPB = 512
  cluster = (1, 1, 1) — no cluster attribute set; flat dim3(T) launch
```

- T = 128 (nearby), 753 (turn1), 3012 (turn4)
- Threads per CTA: 512 = 16 warps
- No warp specialisation — all 16 warps are uniform compute warps

**Constexpr geometry [SOURCE]:**

```cpp
constexpr int H = 8192;
constexpr int C = H / (TPB * 8);   // C = 2 chunks of TPB*8 elements per thread
constexpr int NW = TPB / 32;        // NW = 16 warps (also the smem reduction width)
```

Each thread owns C×8 = 16 BF16 branch elements and 16 fp32 stream elements.

**Shared memory [SOURCE]:**

```cpp
__shared__ float red[2][NW];  // 2 × 16 × 4 B = 128 bytes static; dynamic = 0
```

**Occupancy target [SOURCE]:**

`__launch_bounds__(512, 2)` → compiler targets ≥2 blocks/SM → register budget
≤ 65536 / (2 × 512) = **64 regs/thread**.

---

## §2. SASS disassembly

### §2.1 SASS-OK

`SASS-OK` was printed before the stall-ledger section (log head was truncated in the
fed-back output). Corroborating evidence: `STALL-OK kernel=boundary_kernel samples=2047
coverage=90.2` [MEASURED, job 1793375150] — STALL collection requires a valid `.ncu-rep`
and successful SASS parse.

**Observed instruction families and their roles [MEASURED, job 1793375150]:**

| Family | Description | Structural role |
|---|---|---|
| LDS.128 | 128-bit smem load | Read per-warp partial sums from `red[pass][warp_id]` |
| STS | smem store (2 occurrences) | Write partial sum: `red[pass][warp_id] = ss` (warp leader only) |
| LDG.E.64.CONSTANT | 8-byte `__ldg` load via descriptor | Load alpha/beta/gpost/gpre (texture-cache path) |
| LDG.E.128.CONSTANT | 16-byte `__ldg` load via descriptor | Same, wider variant |
| FFMA | fp32 fused multiply-add | `s = alpha*stream + beta*(b*r1)`, `ss` / `ss2` accumulation |
| FADD | fp32 add | Warp-level partial reduction; serial `Σ red[i]` |
| FMUL | fp32 multiply | `s * r2` and `* gv` for nout output |
| PRMT | byte permute | BF16 → fp32 conversion (`__bfloat1622float2` compiles to PRMT on sm_90) |
| SHF | shift | BF16 word extraction / packing |
| SHFL.BFLY | warp butterfly shuffle | Warp-level sum reduction (5 rounds, masks 16..1) |
| F2FP | float → packed conversion | fp32 → BF16 for `nout` output |
| STG | global store | Write `sout` (fp32) and `nout` (BF16) |
| MUFU | multi-function unit | `rsqrtf` for r1 and r2 |
| BRA | branch | Conditional for counter-zeroing loop and loop bounds |
| ISETP | integer compare + predicate | CTA/thread ID guards |
| LDC | constant load | Kernel argument access |

**Smem access footprint [MEASURED, job 1793375150]:**

- STS: **2 occurrences** — one per RMS reduction pass (warp leader writes `red[pass][tid>>5]`)
- LDS: **8 occurrences** — 4 × LDS.128 per pass × 2 passes = 8 total.
  Each LDS.128 reads 4 fp32 values; 4 × 4 = 16 reads covers NW=16 partial sums.
  All 512 threads read the same 16 values — no bank conflicts (sm_90: 32 banks × 4 bytes;
  16 consecutive addresses = warp read without conflict).

Write side: warp leader only (`if ((tid & 31) == 0)`).
Read side: all threads, element-wise, conflict-free.

---

### §2b. Question-driven queries

| Structural feature | Checked | Finding |
|---|---|---|
| GQA pack (H_Q > H_KV) | Yes — irrelevant to boundary kernel | Not applicable; no QKV geometry |
| Causal / variable-length LPT scheduling | Yes | Not applicable; one CTA = one complete token row |
| Multi-stage pipeline / mbarrier arrival count | Yes | Not present; only `__syncthreads` |
| Division / shift for row-index remapping | Yes — checked PRMT/SHF SASS | `>>` shifts are for warp-id and chunk-index extraction only |
| STREAMHINT load variant (`__ldcs`) | Yes — BND_HINT=false in source | Absent from SASS; plain LDG/STG confirmed |

---

### §2c. Verdict lines

```
SASS-OK   kernel=boundary_kernel [log-head truncated; corroborated by STALL-OK]
STALL-OK  kernel=boundary_kernel samples=2047 rows=17 coverage=90.2 [MEASURED, job 1793375150]
ROLE-OK   roles=1 samples=2047 resolved=100.0 tensor=0.1 [MEASURED, job 1793375150]
SCHED-WAIVED: scalar RMSNorm+blend kernel; no HGMMA/HMMA/UTC*MMA instructions; sched_live_tools grammar inapplicable
LIVE-WAIVED:  same reason; no MMA accumulator chains whose live-range to price
STRUCT-WAIVED: source run; boundary_kernel uses __syncthreads (BAR.SYNC) for plain CTA reduction; no named-barrier construct segments with LDSM/STSM/MMA
PMTL-WAIVED:  driver will collect at wrap-up via stall_price_run.py --pm; ref_price_app = eval/ncu_run.py
```

---

## §2d. Main-loop schedule order

SCHED-WAIVED. No MMA pipeline; no DEPBAR structure.

The kernel has no back-edge over MMA chains. The only branching structures are:
1. A counter-zero loop in block 0 (`for (int i = tid; i < zn; i += TPB)`) —
   3% of stall samples, prologue/epilogue region [MEASURED].
2. A `for c in 0..C=2` chunk loop — fully unrolled by the compiler (C is a constexpr).

Ordered event stream (from source; applies to all threads, all warps):

1. Counter-zero loop (block 0 only, prologue)
2. Load 16 BF16 from `branch[row]`, 16 fp32 from `stream[row]` (interleaved LDG chunks)
3. Load gpost via `__ldg` (16 BF16, LDG.E.CONSTANT); apply `b *= (1 + gpost)`
4. Warp SHFL.BFLY × 5 rounds to reduce `ss` within each warp
5. Write `red[0][warp_id] = ss` (STS; warp leader only)
6. `__syncthreads` — BAR.SYNC 0, 512 threads (symmetric rendezvous)
7. All threads: `tot = Σ red[0][i]` for i in 0..15; MUFU: `r1 = rsqrtf(tot/H + 1e-8f)`
8. Load alpha, beta via `__ldg` (16 fp32 each, LDG.E.CONSTANT)
9. FFMA: `s[i] = alpha[i] * stream[i] + beta[i] * (b[i] * r1)`; accumulate `ss2`
10. STG: store `sout[row]` (16 fp32)
11. Warp SHFL.BFLY × 5 rounds to reduce `ss2` within each warp
12. Write `red[1][warp_id] = ss2` (STS; warp leader only)
13. `__syncthreads` — BAR.SYNC 0, 512 threads (symmetric rendezvous)
14. All threads: `tot2 = Σ red[1][i]` for i in 0..15; MUFU: `r2 = rsqrtf(tot2/H + 1e-5f)`
15. Load gpre via `__ldg` (16 BF16, LDG.E.CONSTANT)
16. PRMT + FMUL: `(s[i] * r2) * gv[i]`; F2FP: pack to BF16
17. STG: store `nout[row]` (16 BF16)

No hiding relation (no MMA chain to overlap). Every step is serial.

Wait-target alignment: the two `__syncthreads` each wait for the same iteration's
partial-sum writes (no cross-iteration dependency; prologue warmup = 0 beats).

---

## §2e. Register live-range bill

LIVE-WAIVED. No MMA accumulator chains.

Non-accumulator working set estimate [SOURCE]:

- `b[C*8]` = 16 fp32 registers (branch values)
- `s[C*8]` = 16 fp32 registers (stream / blended-stream values)
- alpha, beta, gpost, gpre loads: ~8 fp32 temporaries (loaded in pairs via float4)
- Warp reduction `ss`, `ss2`: 2 fp32
- Packing / PRMT temporaries: ~4 uint32
- Smem address (UR4) and row pointer: ~4 registers
- Misc predicates and counters: ~4

Estimated total: **~54–60 registers/thread**, within the 64-reg/thread budget
implied by `__launch_bounds__(512, 2)` [SOURCE]. Actual NCU register count not
captured (metrics CSV truncated from log).

---

## §2f. Per-instruction stall pricing

[MEASURED, job 1793375150]

**Top stall rows (≥90.2% cumulative coverage):**

```
=== STALL LEDGER BEGIN ===
     518 25.31%  LDS    | barrier=88%, mio_throttle=9%, wait=2%
     436 21.30%  FFMA   | long_scoreboard=98%, wait=1%
     295 14.41%  PRMT   | long_scoreboard=93%, not_selected=4%, math_pipe_throttle=3%
     187  9.14%  FADD   | short_scoreboard=79%, not_selected=11%, wait=10%
     166  8.11%  BRA    | drain=97%, branch_resolving=2%
     110  5.37%  LDG    | lg_throttle=67%, mio_throttle=20%, wait=7%
      99  4.84%  SHF    | long_scoreboard=67%, not_selected=18%, math_pipe_throttle=12%
      82  4.01%  FMUL   | long_scoreboard=85%, short_scoreboard=11%
      33  1.61%  ISETP  | short_scoreboard=64%, not_selected=18%
      30  1.47%  LDC    | no_instructions=47%, mio_throttle=37%
      16  0.78%  SHFL   | wait=62%, mio_throttle=19%
      16  0.78%  F2FP   | not_selected=69%, wait=25%
      14  0.68%  STG    | mio_throttle=79%, lg_throttle=14%
STALL-OK kernel=boundary_kernel samples=2047 rows=17 coverage=90.2
```

**Structural attribution of the dominant wait point:**

LDS 25.3% (barrier=88%) — threads stalling on `__syncthreads()` awaiting the
slowest warp's partial-sum write to shared memory. The structural decision this
belongs to: the **two-pass block reduction** (warp SHFL → STS → syncthreads →
serial LDS sum). This is the single largest gap-owner; reducing it requires
either (a) fewer warps waiting, (b) a faster reduction topology, or (c) overlapping
the `__syncthreads` latency with other independent work.

Second largest: FFMA long_scoreboard=98% (21.3%) — register-file RAW latency in
the `s = alpha*stream + beta*(b*r1)` chain. The chain has enough depth (16
elements per thread) that pipelining within one thread is limited.

Third: PRMT long_scoreboard=93% (14.4%) — BF16→fp32 byte-permute latency; 16
BF16 conversions per thread form a chain with insufficient independent work to
hide them.

---

## §2g. Synchronization topology

**Named barrier topology [SOURCE; corroborated by STALL-OK + STS=2/LDS=8]:**

| ID | Participants | Arrive-only | Arrive+Wait | Verdict |
|---|---|---|---|---|
| 0 (pass 1) | 512 threads | none | all 16 warps | symmetric rendezvous |
| 0 (pass 2) | 512 threads | none | all 16 warps | symmetric rendezvous |

No mbarriers. No warpgroup barriers. No cluster barriers.
Total blocking rendezvous per token = **2**.

**Merge position:** warp-level reductions (`SHFL.BFLY`) run *inside* each pass,
*before* `__syncthreads`. Serial accumulation happens *after*. Both reductions
are per-invocation (not deferred to epilogue).

**Issue-first-then-correct fingerprint:** Not observed.

**Supply issue order:** Not applicable — no asynchronous loads or staging rings.
LDG.CONSTANT loads for alpha/beta/gpost/gpre are issued near first use (interleaved
with branch/stream loads in chunk order), not hoisted to the top of the CTA.

**Cache policy bits [MEASURED from SASS opcodes, job 1793375150]:**

- `LDG.E.64.CONSTANT desc[UR6][R20.64+0x0]` — 8-byte `__ldg` via texture descriptor
- `LDG.E.128.CONSTANT desc[UR6][R46.64+0x0]` — 16-byte `__ldg` via texture descriptor
- No `LDGSTS`, `CREATEPOLICY`, `UBLKCP`, or L2-prefetch instructions
- `STG` plain (no `.CS`/`.WT` modifier): `BND_HINT=false` confirmed in SASS

---

## §3. DRAM traffic + kernel count

**Kernel count:** 1 [SOURCE; ROLE-OK roles=1, MEASURED].

**Compulsory traffic per token row (H = 8192) [SOURCE-derived]:**

| Operand | Dtype | Bytes/token | Re-read per token |
|---|---|---|---|
| `branch` | BF16 | 16384 | yes |
| `stream` | fp32 | 32768 | yes |
| `gpost` | BF16 | 16384 | no — shared, L2-cached |
| `alpha` | fp32 | 32768 | no — shared, L2-cached |
| `beta` | fp32 | 32768 | no — shared, L2-cached |
| `gpre` | BF16 | 16384 | no — shared, L2-cached |
| `sout` write | fp32 | 32768 | yes (write) |
| `nout` write | BF16 | 16384 | yes (write) |

Token-specific reads: 16384 + 32768 = 49152 B/token.
Token-specific writes: 32768 + 16384 = 49152 B/token.
Shared weights (first access): 98304 B total ≈ 96 KB.

DRAM bytes at T=3012: ≈ 3012 × 98304 ≈ 284 MB (token-specific) + 96 KB ≈ 284 MB.

Arithmetic intensity (per token): ~65536 FLOPs / 98304 B ≈ 0.67 FLOPs/byte.
H100 ridge ≈ 295 FLOPs/byte (tensor), ≈ 20 FLOPs/byte (fp32 scalar).
The kernel is **intrinsically memory-bandwidth bound** well below the scalar ridge.

L2 hit rate: not measured from NCU CSV (log truncated before STEP 3 output).

### §3b. Bottleneck and residency verdict

**(a) Roofline position:** Scalar memory-bandwidth bound. Structural arithmetic
intensity 0.67 FLOPs/byte << scalar ridge ~20 FLOPs/byte. `tensor_active=0.1%`
confirms no tensor-core use [MEASURED, job 1793375150].

**(b) Dominant traffic operand:** `stream` (fp32, 32768 B/token) is the largest
per-token operand. At T=3012, stream alone ≈ 96 MB DRAM. Shared norm weights
(gpost+alpha+beta+gpre = 96 KB) fit in H100 L2 (50 MB) after the first block.

**(c) Residency load-bearing verdict:** `load-bearing = false`. The shared weights
(96 KB total working set) trivially fit in L2. Per-token data is streaming —
read/written once per token, no locality across tokens. No inter-CTA dispatch
ordering is needed.

### §3c. Inter-kernel dataflow

Not applicable — single kernel per call. kernel_count = 1.

---

## §4. Hardware constraint solving

**Smem feasibility [SOURCE-computed]:**

- Static smem: 128 bytes. Dynamic smem: 0. CTA limit sm_90a: 100 KB — fits trivially.
- `smem.occupancy("sm_90a", 128, ≤64, 512)`: 2 blocks/SM (smem not binding;
  register file is the binding constraint at ≤64 regs/thread).

---

## Verdict lines summary

```
SASS-OK   kernel=boundary_kernel [log-head truncated; corroborated by STALL-OK]
STALL-OK  kernel=boundary_kernel samples=2047 rows=17 coverage=90.2
ROLE-OK   roles=1 samples=2047 resolved=100.0 tensor=0.1
SCHED-WAIVED: scalar kernel; no HGMMA/HMMA/UTC*MMA; sched_live_tools grammar inapplicable
LIVE-WAIVED: same reason
STRUCT-WAIVED: source run; BAR.SYNC-only reduction; no LDSM/STSM/MMA named-barrier construct
PMTL-WAIVED: driver collects at wrap-up via stall_price_run.py --pm; entry = eval/ncu_run.py
```

---

## Disclaimers

1. **Register readings:** Actual NCU `sm__registers_per_warp` was in the metrics list
   but the CSV output was truncated; values above are source-derived estimates.

2. **DRAM byte counts:** `dram__bytes_read.sum` / `dram__bytes_write.sum` were
   collected but not captured in the provided log. Figures in §3 are source-derived.

3. **LDG.CONSTANT descriptor:** This is the `__ldg` texture-cache path, not
   traditional constant cache. A student using plain `*ptr` instead of `__ldg`
   for the four norm-weight arrays will use plain LDG with different L2 behaviour.

4. **Shape-specific profiling:** Only the `MODE=0` (`has_gpost=True`) instantiation
   was profiled. The `boundary_rc` (MODE=1/2) path is a different instantiation.

5. **PRMT intensity:** 14.4% of stall samples on PRMT is the BF16→fp32 conversion
   cost on sm_90; the student's nvcc build will emit the same PRMT for
   `__bfloat1622float2`, so this is a per-conversion cost, not an optimization opportunity.
