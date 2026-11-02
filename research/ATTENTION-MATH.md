# Derive the design before choosing instructions

Let N be the number of visible keys, d=64, and R the number of query rows
sharing a KV head. For one KV head, with FP8 K/V and their full reuse:

- Necessary KV reads: 2 N d bytes (ignoring cache residency/compression).
- QK and PV work: approximately 4 R N d FLOPs.
- Arithmetic intensity: 2R FLOPs/byte. Decode R=8 =>16; verify R=64 =>128.

These are useful-work counts, not predictions of elapsed time. Page tables,
staging, reductions, conversion, occupancy, synchronization, and duplicate
physical transactions contribute additional costs. For an isolated kernel,
T >= max(bytes/achievable_bandwidth, work/achievable_compute). Achievable
rates must be measured for this access pattern rather than assumed from a
datasheet. Adding launch/sync penalties is a model, not a universal bound
because costs can overlap.

The two-head design changes transaction arrangement, not the necessary byte
count. It must pay for its larger CTA and coupled scheduling using a measured
transaction-efficiency benefit. The initial hypothesis failed: all eleven
task checks passed, but nine timings were slower (0.83–0.90x production).
Reject it; do not treat contiguous addresses alone as a speedup proof.

An algebraic opportunity: each attention partition can return sufficient
statistics (m_i, l_i, u_i), where m_i is a log2 score maximum,
l_i=sum exp2(score-m_i), and u_i=sum exp2(score-m_i)*v. Then for
m=max m_i and w_i=exp2(m_i-m),

    l=sum w_i*l_i, u=sum w_i*u_i, attention=u/l.

This avoids normalizing each partial and then reconstructing its denominator
via log/exp. Empty partitions require zero weights; epsilon remains nonzero.
The production BF16 rounding of attention before RMS normalization must be
preserved or independently justified against the task accuracy contract.

Without that rounding, RMS(u/l)=u/sqrt(mean(u^2)+eps*l^2) for positive l.
This is exact real arithmetic and avoids division by l. It is NOT a proof of
bitwise equivalence through an intermediate BF16 cast. The cluster merge
probe tests this identity, while the integrated attention preserves the cast.

Integrated follow-up (`experiments/attention_rms_denominator_v1/`) inspected
the incumbent: its `pr8()` helper currently returns its input, so no such
intermediate BF16 cast is present in these finalizer paths. I replaced the
per-element `vscale/l` normalization with the corresponding scaled-RMS form
`u/sqrt(mean(u^2)+eps*(l/vscale)^2)` in all five epilogues. All 12 H200
reference checks passed. The paired cold-L2 geometric time ratio was only
1.00275 overall, with a mixed per-case signal, so the branch is archived and
not promoted. Correctness/timing evidence is only against the public image and
does not establish an official score.

Finally optimize the actual objective: for K load groups,
log(score)=mean_k(log(baseline_k)-log(candidate_k)). For a component t_j
inside load total T_k, -d log(score)/d log(t_j)=t_j/(K*T_k).
Use this sensitivity to reject improvements with negligible score impact.

Experiments following this derivation:

- `stats_run.log`: direct sufficient statistics, BF16 cast preserved, R=1/2/4/8
  rows per merge CTA. All 18 independent task checks passed. Whole-pipeline
  gains versus production were about 1%, less than the earlier FP8 parent;
  do not promote the algebraic rewrite merely because it removes operations.
- `qreg_compile.log`: keep Q in WGMMA register-A form for the entire KV loop.
  Eight 32-bit registers describe each thread's invariant query fragment;
  remove Q staging/copying and repeated shared-A reads. On this workload the
  A fragment is fixed across all tiles. CPU compiler reports 118 registers
  versus parent's119, no spills. GPU evidence is in `qreg_run.log`.

Q-register integration status: the initial full implementation fails for
multi-tile chunks and must not be submitted. The isolated primitive passed
all12 checks: identity/permutation/integer products exact in both register-A
modes and shared-A control, random FP8 errors identical (~7.54e-5). The
full-kernel first score tile matches torch FP32 dot to ~7.8e-5; short one-tile
attention passes, multi-tile attention fails. Thus the initial assumption of
a fragment-mapping error was disproved. Empty read/write operand fences
alone did not fix it. `qreg_hazards.log` compares stronger asm constraints
and a wait between the two query MMAs; no timing is valid until task checks
pass. Consult NVIDIA's PTX register-fragment documentation and CuTe's
`mma_sm90_gmma.hpp` for operand constraints rather than guessing layouts.

Both hazard variants failed the same two early checks (~1.41 relative error),
so neither operand constraints nor inter-MMA waits fixed the integration.
Set this implementation aside. Its primitive remains a validated building
block, but there is no evidence of an end-to-end benefit or a resolved cause.

Combined official experiment `attention_fp8p_cluster_v1`, job
`7a6f649cc26f4e6b`, uses only the two separately official-validated branches,
not the failing Q-register experiment. Source SHA
4823f74b7fb8f4c8247f92a6a4c884f492750aaa3c7f74f86fc402a2864b9a02.

That combined candidate passed all16 tests and12 timed cases, private1.73260865x,
and was promoted/publicly submitted as966fde6359c343d3. Its public run remains
pending at this note's update. Decoder integration is separately under official
check aa0afe844e254d62; attention coverage is not a decoder correctness proof.

FP8 lazy-max bound: with probability offset8, any lag t satisfying
2^(8+t)<=448 cannot overflow finite E4M3. The theoretical limit is
log2(448)-8=0.807354922. Tested t=.25/.5/.75 preserves this invariant.
All18 independent task checks passed. Paired whole-pipeline geometric gains
versus the stable-max parent were1.00342/1.00340/1.00459x respectively across
two batches/two seeds. Correct but too small to justify promotion/submission.
The exact real-arithmetic rescaling identity is preserved; FP8 rounding may
differ, hence the independent checks rather than a bitwise-equivalence claim.
