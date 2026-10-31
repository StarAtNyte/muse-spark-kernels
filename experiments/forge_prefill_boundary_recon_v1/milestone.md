# muse_prefill_boundary milestone

## 0. Problem statement and target

**Operator**: boundary kernel — gated RMSNorm + residual blend for the prefill boundary step.
One CTA per token row; no cross-CTA collaboration.

**Scored shapes**:

| T   | H    | branch dtype | stream dtype | nout dtype | GPOST |
|-----|------|--------------|--------------|------------|-------|
| 128 | 8192 | BF16         | fp32         | BF16       | true  |
| 753 | 8192 | BF16         | fp32         | BF16       | true  |
| 3012| 8192 | BF16         | fp32         | BF16       | true  |

**Architecture reference**: job 1793375150 — `boundary_kernel<TPB=512, GPOST=true,
STREAMHINT=false, MODE=0>` launched via `okc_launch` with
`cudaLaunchAttributeProgrammaticStreamSerialization=1`.

**Hardware**: ARCH_REF device (H100 class, sm_90a). `target_window: device`
(CUDA graph replay).

**Target**:

```
target_ms:     0.0272   [MEASURED, job 1793375150]
target_window: device
```

**BND_CUDA_MIN = 1** [SOURCE]: the CUDA path is used for all T ≥ 1; Python fallback is never
taken for the scored shapes.

---

## 1. Problem contract (Math Spec)

**Inputs** (per token row `t`, element index `h ∈ [0, H)`):

| Tensor  | Shape | Dtype | Ownership |
|---------|-------|-------|-----------|
| branch  | [T, H]| BF16  | per-token |
| stream  | [T, H]| fp32  | per-token |
| gpost   | [H]   | BF16  | shared norm weight |
| alpha   | [H]   | fp32  | shared norm weight |
| beta    | [H]   | fp32  | shared norm weight |
| gpre    | [H]   | BF16  | shared norm weight |

**Outputs**:

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| sout   | [T, H]| fp32  | intermediate blended activations |
| nout   | [T, H]| BF16  | normalized output |

**Math contract — two-pass computation**:

Pass 1 (gated branch + blend → sout):
```
b_h   = branch[t][h] × (1 + gpost[h])       for h in [0, H)
ss    = Σ_{h} b_h²
r1    = rsqrt(ss / H + 1e-8)
sout[t][h] = alpha[h] × stream[t][h] + beta[h] × (b_h × r1)
```

Pass 2 (second RMSNorm → nout):
```
ss2   = Σ_{h} sout[t][h]²
r2    = rsqrt(ss2 / H + 1e-5)
nout[t][h] = F2FP_BF16( sout[t][h] × r2 × gpre[h] )
```

H = 8192 [SOURCE]. Pass 1 epsilon = 1e-8; pass 2 epsilon = 1e-5 — not interchangeable.

**Diagnosis checklist**:
- Epsilon swap: 1e-8 in pass 1, 1e-5 in pass 2
- `gpost` scaling is `(1 + gpost_h)`, not `gpost_h` alone
- F2FP BF16 rounding applies to `nout` but **not** `sout`
- Denominator is H (constant 8192), not T or the actual element count

---

## 2. Decision DAG + phase order

```
M0 (architecture fixed + first runnable version)
  D0.1  CTA-per-token decomposition + warp roles  [PHASE: none]
  D0.2  Thread tile: C=2 chunks, 16+16 elements/thread
  D0.3  Two-pass block reduction                  [SYNC: named-barrier]
  D0.4  __ldg for shared norm weights             [GATHER: none]
  D0.5  __launch_bounds__(512, 2) register cap
  D0.6  128 B static smem: float red[2][NW]
  D0.7  Plain STG for output stores (BND_HINT=false)
  D0.8  Counter-zero prologue (CTA 0 only)
    └── M0 Gate (correctness + architecture-conformance + mechanism lines; no perf line)

M1 (performance gate)
    └── M1 Gate: device_ms ≤ 0.0272 [MEASURED, job 1793375150]
```

No free-play phase outside the decision graph.

The ref baseline (job 1793375150) is an authoring input, not a phase in the DAG.

---

## 3. Decision details

### M0 — architecture + first runnable version

**Intermediate consumption-path ledger**: none. Tensor active = 0.1%
[MEASURED, job 1793375150]; no MMA instructions, no RS / smem ruling rows.
No smem buffer needs endorsement via MMA ledger (the one smem buffer, `red[2][NW]`,
is a scalar reduction staging array, not an MMA operand buffer).

**Exchange-face access-atom transcription** (smem buffer `float red[2][NW]`, static 128 B):

| Side | Instruction family | Measured source |
|------|--------------------|-----------------|
| Write | STS — scalar store; warp leader only (1 write / pass / warp → 2 STS total per token) | [MEASURED, job 1793375150] |
| Read | LDS.128 — vectorized scalar load; all 512 threads read; 4 LDS.128 per pass × 2 passes = 8 total | [MEASURED, job 1793375150] |

No bank conflicts: 16 consecutive fp32 elements fit within the 32-bank smem width; all
threads read the same 64-byte stripe.

**Issue sequence** (transcribed from recon §2d [MEASURED, job 1793375150]):

1. CTA 0 prologue: counter-zero loop (`blockIdx.x == 0`; ~3% of stall samples)
2. Load `branch[t]` (16 BF16) and `stream[t]` (16 fp32) for C=2 chunks
3. Load `gpost[h]` via `__ldg` (16 BF16); apply `b *= (1 + gpost_h)`; accumulate `ss += b²`
4. Warp SHFL.BFLY × 5 rounds (masks 16,8,4,2,1): reduce `ss` to warp leader
5. STS: warp leader writes partial `ss` to `red[0][warp_id]`
6. BAR.SYNC id=0, 512 threads [SYNC: named-barrier] — first rendezvous
7. All threads: serial LDS.128 × 4 reads summing `red[0][0..15]`; MUFU `r1 = rsqrtf(tot/H + 1e-8)`
8. Load `alpha[h]`, `beta[h]` via `__ldg` (fp32); FFMA: `sout = alpha×stream + beta×(b×r1)`; accumulate `ss2 += sout²`
9. STG: store `sout[t]` (16 fp32)
10. Warp SHFL.BFLY × 5: reduce `ss2` to warp leader
11. STS: warp leader writes partial `ss2` to `red[1][warp_id]`
12. BAR.SYNC id=0, 512 threads [SYNC: named-barrier] — second rendezvous
13. All threads: serial LDS.128 × 4 summing `red[1][0..15]`; MUFU `r2 = rsqrtf(tot2/H + 1e-5)`
14. Load `gpre[h]` via `__ldg` (16 BF16); PRMT + FMUL; F2FP → BF16 pack
15. STG: store `nout[t]` (16 BF16)

**Segmented live-set budget** (no MMA accumulators; non-accumulator working set only):
- Working set = `b[C×8]` BF16 broadcast to fp32 + `s[C×8]` fp32 + addressing temporaries
  ≈ 54–60 regs/thread [ESTIMATE, source-derived from C=2, 16 fp32 b-registers + 16 fp32 s-registers]
- Cap: ≤ 64 regs/thread from `__launch_bounds__(512, 2)` [SOURCE]
- No spill ladder: estimated working set (54–60) fits the cap (64); verify `spill_ld_sectors == 0`

---

#### D0.1 — CTA-per-token decomposition + warp roles

```
Action            — Launch T CTAs of 512 threads each; assign all 16 warps to uniform compute
Purpose           — Partition token rows across the grid with no cross-CTA dependencies;
                    full intra-CTA parallelism over H=8192 elements
Trigger condition — always; one complete token row per CTA; H=8192 with TPB=512 and C=2
                    fills exactly 512 threads × 16 elements = 8192 elements per CTA
Cost              — no cross-CTA communication; uniform row length means no load imbalance
Depends on        — (root decision)
```

`[PHASE: none]` — single consumer group (all 16 warps are symmetric uniform-compute; no
warp specialization; no scalar segment requiring phase cover from a second warpgroup).

`[VERIFY: ncu-direct]` — `block_size == 512`; `grid_size == T` (verifiable per scored shape).

---

#### D0.2 — Thread tile: C=2 chunks, 16 BF16 + 16 fp32 per thread

```
Action            — Each thread covers C=2 chunks of 8 BF16 + 8 fp32 (16 elements each dtype)
Purpose           — Provide 16 independent FMAs per thread for FFMA chain ILP; avoids the
                    halved-ILP penalty of TPB=1024 which would give C=1 [ESTIMATE, source-derived]
Trigger condition — constexpr: C = H / (TPB × 8) = 8192 / (512 × 8) = 2 [SOURCE]
Cost              — fixes elements-per-thread at 16; cannot be changed independently of TPB or H
Depends on        — ⇐ D0.1
```

`[VERIFY: ncu-derived]` — `grid_size × block_size × 16 == T × H`; verifiable from
ncu-direct grid and block dimensions (T × 512 × 16 = T × 8192).

---

#### D0.3 — Two-pass block reduction: SHFL.BFLY + STS + BAR.SYNC + serial LDS + MUFU

```
Action            — Reduce the per-thread partial sum across all 512 threads in two phases:
                    (a) intra-warp: SHFL.BFLY × 5 rounds → warp leader accumulates partial;
                    (b) cross-warp: STS from warp leader → BAR.SYNC → serial LDS by all threads → MUFU rsqrtf
Purpose           — Compute exact block-wide sum for RMSNorm; scalar kernel has no tensor-core
                    reduction path; named barrier is the correct protocol for this all-hands rendezvous
Trigger condition — scalar kernel (tensor_active = 0.1% [MEASURED, job 1793375150]); full H=8192
                    must be reduced per token; all 16 warps must contribute
Cost              — two BAR.SYNC rendezvous per token (dominant stall: stall_barrier 22%
                    [MEASURED, job 1793375150]); 128 B static smem for red[2][NW]
Depends on        — ⇐ D0.1, D0.2
```

`[SYNC: named-barrier]` — barrier id=0; 512 threads participate; symmetric arrive-and-wait;
2 rendezvous per token (one at end of pass 1, one at end of pass 2); low-frequency
(2 total per kernel iteration, not streaming), appropriate for a named-barrier ruling.

`[VERIFY: ncu-direct]` — mechanism line: `stall_barrier ≤ 27.5%`
(ref: 22% [MEASURED, job 1793375150]; band = ref × 1.25 = 27.5%).

---

#### D0.4 — `__ldg` texture-cache path for shared norm weights

```
Action            — Load gpost, alpha, beta, gpre via __ldg; per-token branch and stream
                    use plain LDG (not shared across CTAs)
Purpose           — Route shared norm weights through the texture cache (L1 TEX);
                    the four weight tensors total ~96 KB [ESTIMATE, derived from 4 × H × dtype_bytes]
                    and fit in H100 L2 (50 MB); reuse across T CTAs amortises the load cost
Trigger condition — BND_HINT=false [SOURCE]; weights are broadcast-read (many CTAs load
                    the same elements at different token offsets)
Cost              — requires weight pointers to be __restrict__ / non-aliased with outputs;
                    no barrier or smem cost
Depends on        — ⇐ D0.1
```

`[GATHER: none]` — all gmem feeds (norm weights gpost/alpha/beta/gpre and per-token
branch/stream) use affine addressing; no data-dependent index load (no `x[idx[i]]` pattern).

`[VERIFY: sass-derived]` — SASS contains `LDG.E.64.CONSTANT` for gpost and gpre (4×BF16
vector load via texture descriptor `desc[UR6]`) and `LDG.E.128.CONSTANT` for alpha and beta
(4×fp32 vector load); plain `LDG.E.64` / `LDG.E.128` for branch and stream
[MEASURED, job 1793375150].

---

#### D0.5 — `__launch_bounds__(512, 2)` register cap

```
Action            — Annotate boundary_kernel with __launch_bounds__(512, 2)
Purpose           — Target ≥2 resident blocks per SM; caps compiler register allocation
                    to 65536 / (2 × 512) = 64 regs/thread [SOURCE]
Trigger condition — TPB=512; estimated working set ≈ 54–60 regs fits within the 64-reg cap
                    [ESTIMATE]; spill must be verified zero
Cost              — if actual working set exceeds 64 regs, the compiler spills to local memory
                    (local memory accesses → bandwidth + latency penalty)
Depends on        — ⇐ D0.1, D0.2
```

`[VERIFY: ncu-direct]` — `regs_per_thread ≤ 64`; paired check: `spill_ld_sectors == 0`.

---

#### D0.6 — 128 B static smem: `float red[2][NW]`

```
Action            — Declare __shared__ float red[2][NW] with NW = TPB/32 = 16
Purpose           — Provide the cross-warp staging area for two reduction passes; no dynamic smem
Trigger condition — NW = 512/32 = 16; 2 passes × 16 warp leaders × 4 B = 128 B [SOURCE]
Cost              — 128 B static smem per CTA; negligible on H100 (max 228 KB shared per SM)
Depends on        — ⇐ D0.3
```

`[VERIFY: ncu-direct]` — `smem_static == 128` (same predicate as D0.3 architecture-conformance
line; this decision is its source).

---

#### D0.7 — Plain STG for output stores (BND_HINT=false)

```
Action            — Use plain STG (no .CS evict hint) for sout and nout stores
Purpose           — Output is written once per token and not reused; the default streaming
                    behaviour is sufficient; .CS would also risk evicting the shared norm
                    weights already resident in L2
Trigger condition — BND_HINT=false [SOURCE]; output tensors are streaming (no reuse)
Cost              — no additional cost
Depends on        — ⇐ D0.4
```

`[VERIFY: sass-derived]` — SASS output stores are plain `STG` without `.CS` or other
evict-hint modifiers [MEASURED, job 1793375150].

---

#### D0.8 — Counter-zero prologue (CTA 0 only)

```
Action            — Execute a counter-zero loop in CTA 0 (blockIdx.x == 0) before the
                    main kernel body
Purpose           — Zero work counters that are read by the host or driver; must complete
                    before any token's CTA reads a counter
Trigger condition — always (per correctness invariant); measured at ~3% of total stall
                    samples [MEASURED, job 1793375150], not a performance concern
Cost              — small serialization in CTA 0 only; other CTAs are unaffected
Depends on        — ⇐ D0.1
```

`[VERIFY: correctness]` — stale counters produce incorrect scheduling; correctness gate
catches missing zeroing.

---

### M0 Gate

**Prerequisite** (must be in hand before the gate row is submitted): whole-machine smoke
certificate — all three scored shapes (T=128, T=753, T=3012) launch without a runtime crash.
Values are not compared at smoke time.

**Correctness**:
- [ ] `max_diff ≤ judge_tolerance` vs FP32 autograd golden reference [VERIFY: correctness]

**Architecture-conformance lines**:
- [ ] `block_size == 512` [VERIFY: ncu-direct]
- [ ] `smem_static == 128` [VERIFY: ncu-direct]
- [ ] `regs_per_thread ≤ 64` [VERIFY: ncu-direct]
- [ ] `spill_ld_sectors == 0` [VERIFY: ncu-direct]
- [ ] `tensor_pct ≤ 2%` (scalar kernel; ref: 0.1% [MEASURED, job 1793375150]) [VERIFY: ncu-direct]

**Mechanism-verification line** ([SYNC: named-barrier] from D0.3):
- [ ] `stall_barrier ≤ 27.5%` [VERIFY: ncu-direct]
  (ref: 22% [MEASURED, job 1793375150]; band = ref × 1.25)

**No absolute performance line** — the M0 Gate verifies architecture, not timing.

---

### M1 — Performance gate

**Goal**: confirm the M0 architecture meets the device timing target (CUDA graph replay).

No new structural decisions in M1. This memory-bandwidth-bound kernel's performance is
fully determined by the M0 topology; all levers (TPB, register cap, cache path) are frozen.

Bottleneck characterization (advisory):
- Arithmetic intensity ≈ 0.67 FLOPs/byte [ESTIMATE, derived from 2×H FFMA + 2×H scale / per-token byte count at H=8192]
- Scalar fp32 ridge ≈ 20 FLOPs/byte [ESTIMATE, H100 hardware specification]; AI << ridge → memory-bandwidth bound
- Dominant stall in reference: stall_long_sb 41% (FFMA chain latency) [MEASURED, job 1793375150]; stall_barrier 22%

#### M1 Gate

- [ ] `device_ms ≤ 0.0272` [MEASURED, job 1793375150]
  (geometric mean over T ∈ {128, 753, 3012}, GPOST=true path, CUDA graph replay)

---

## 4. Known dead ends

*(Empty in v1; populated by trajectory harvest.)*

| Decision / bundle | Symptom | Trigger iteration | Retry precondition |
|---|---|---|---|
| — | — | — | — |

**Candidate B (TPB=1024) pre-falsified** [SOURCE]: `#define BND_TPB 512` in the ORACLE
rules out TPB=1024. The structural argument: TPB=1024 gives C=1 (half ILP), 32 warps
(wider BAR.SYNC wait), and ≤32 regs/thread cap (spill risk). No measurement needed;
the source is decisive. Do not sweep TPB=1024.

---

## 5. Global correctness / performance gate

**Correctness tolerance**: `max_diff` vs FP32 autograd; tolerance is the task's
`judge_tolerance` (frozen for this run [SOURCE]). Pass/fail is binary; no relaxed form.

**Health criteria** (advisory; machine-read fields from NCU):

| Metric | Target | Basis |
|--------|--------|-------|
| `tensor_pct` | ≤ 2% | scalar kernel; ref 0.1% [MEASURED, job 1793375150] |
| `stall_barrier` | ≤ 27.5% | named-barrier sync; ref 22% [MEASURED, job 1793375150] |
| `regs_per_thread` | ≤ 64 | __launch_bounds__(512, 2) [SOURCE] |
| `spill_ld_sectors` | == 0 | working set fits cap [ESTIMATE] |

Advisory (diagnosis only, not a gate line):
- `stall_long_sb` ≈ 41% [MEASURED, job 1793375150] — FFMA chain latency; expected for
  this memory-bound scalar kernel; not a gather symptom ([GATHER: none] confirmed)

---

## 6. Rules

**Free delivery granularity**: no acceptance line for sub-decisions inside a milestone;
commit granularity is the agent's choice (single slice to the whole M0 bundle). The gate
row (decision_id containing the substring `gate`) must be committed alone — not bundled
with any decision. After green, a delivery with `[CLOSES: M0-gate]` allows the driver to
file the gate row separately using that delivery's reading.

**[MEASURED]/[ESTIMATE]/[SOURCE] marking discipline**: every number is marked:
- `[MEASURED, job N]` — directly measured on the architecture reference
- `[ESTIMATE]` — derived from measured facts or hardware specs; derivation written out
- `[SOURCE]` — read verbatim from the ORACLE source (no hardware measurement)

Unsourced expected numbers are forbidden.

**Watermark ratchet**: committed device_ms never gets worse (within ±1% noise floor).
Differences within ±1% are not computed finely and not chased. Intermediate commits
inside a declared `⊗` companion group are exempt during the build phase.

**Bundle ledger**: one commit with several changes states shared budgets/knobs in the
commit message; joins decision ids with `+` (e.g. `D0.5+D0.6`).

**Anti-plagiarism clause**:
- **Allowed**: `hw_kb/` reference documentation; public CUDA / PTX / SASS ISA manuals;
  academic papers on RMSNorm; H100 hardware specifications.
- **Forbidden**: reading or transcribing `task/reference/submission.py` or any copy of
  it. The optimized kernel body must not be copied verbatim, renamed, or translated
  line-by-line. Contract snippets (function signatures, struct shapes) that the student
  independently designs to the same contract specification are allowed.
- **Detection trigger**: any function body reproducing > 3 consecutive non-trivial
  statements from the ORACLE kernel is treated as a violation; the eval is rejected.

**INFEASIBLE clause**: before declaring INFEASIBLE on any gate line, provide a
metric-by-metric comparison with the architecture reference's NCU signature
(job 1793375150). A metric the reference meets with the same resource envelope
(TPB=512, smem=128B, regs≤64, GPOST=true, H=8192) is no basis for INFEASIBLE.
The only channel for stopping the loop is meta's `cause_class: INFEASIBLE`.

**Tightness rules for performance lines** (template §B): the M0 Gate carries no
absolute performance line — how far the first version gets cannot be known in advance;
the tightened target (device_ms ≤ 0.0272) sits in the M1 Gate which has the means to
close it. Correctness lines are binary and rigid throughout; no relaxed form and no
deferral.

**Downgrade discipline**: pre-scheduled downgrades (build a conservative placeholder
first, then upgrade after green) are forbidden. If a bundle fails whole-machine twice,
it may be downgraded as a failure response to get first green, then swapped back one
decision at a time after green. A downgrade applied without a failure trigger is
rejected by the driver.

---

## Worked Example

**Scored shapes** (this run's scope):

| T    | H    | BND_TPB | BND_HINT | GPOST | MODE | grid_size | block_size |
|------|------|---------|----------|-------|------|-----------|------------|
| 128  | 8192 | 512     | false    | true  | 0    | 128       | 512        |
| 753  | 8192 | 512     | false    | true  | 0    | 753       | 512        |
| 3012 | 8192 | 512     | false    | true  | 0    | 3012      | 512        |

One CTA per row; 512 threads × 16 elements/thread = 8192 elements = H. No remainder.

**Resource budget** (all shapes identical per-CTA):

| Resource | Value | Source |
|----------|-------|--------|
| smem_static | 128 B | [SOURCE] |
| regs/thread cap | ≤ 64 | [SOURCE] |
| BAR.SYNC rendezvous/token | 2 | [MEASURED, job 1793375150] |
| LDG.E.CONSTANT width (BF16 weights) | 64-bit (4×BF16) | [MEASURED, job 1793375150] |
| LDG.E.CONSTANT width (fp32 weights) | 128-bit (4×fp32) | [MEASURED, job 1793375150] |
| stall_barrier (ref) | 22% | [MEASURED, job 1793375150] |
| stall_long_sb (ref) | 41% | [MEASURED, job 1793375150] |
| tensor_pct (ref) | 0.1% | [MEASURED, job 1793375150] |

**Shape outside scope**:
- T < 1: not launched (BND_CUDA_MIN = 1 [SOURCE])
- T ≥ 8192: uses BND_RECOMPUTE path (BND_RC_MIN_T = 8192 [SOURCE]); separate kernel
  variant, not scored in this run
