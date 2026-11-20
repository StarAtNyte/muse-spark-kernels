# Post-competition public repos: techniques and agent methods (2026-11-20)

Read-only survey of repos published after the kernel track closed (2026-11-05). No code was run. Check each repo's license before reusing anything.

## Kernel techniques we lacked (ranked by transferability)
1. Attention: byte-domain E4M3 P exp (a Schraudolph-style integer 2^x writes P's E4M3 code directly: 2 f16x2 FMA + a byte permute, replacing MUFU ex2 + cvt) plus an f16-accumulating QK wgmma (m64n64k32.f16.e4m3.e4m3). Sources: adamleex (attention 2.895x), ethduke. Depends on the 11-02/11-03 rulings on FP16 QK.
2. Batch-1 sliding decode/verify on per-KV-head 16-CTA clusters with an st.async DSMEM push merge: d1s 7.4->4.4 us, v1s 11.0->5.5 us (gauravjain14, adamleex).
3. Barrier-free MoE ends: sentinel-initialised per-token rows polled by 8 CTAs, or a data-flag A->B handoff (adamleex R16D, ethduke).
4. T<=8 fused pooler + FP32 router + routing tail with a last-CTA ticket counter (joeyhlu G2, idea only; predicted -1/-3.5 us at T=1/8). Our 16-row attempt lost because of deferred norm; theirs keeps the norm.
5. Cold-TLB/i-cache-aware startup: early page touches, L2::evict_first on verify-full K/V, compact per-shape-class modules (~1 us per 32 KB of cold i-cache); worth ~1-2% (gauravjain14).
6. Exhaustive cuBLASLt search at m=128/256 (44,199 configs; pooler +22-25%) (gauravjain14).
7. In-prepare per-T route races, pinning flip-prone choices (yigitkucuk, EROREx, ethduke).
8. MoE: per-job scale/(1+w) tables in smem filled by the idle helper warp; x gather as whole 128-B lines; drop fence.acq_rel.gpu (CUTLASS GenericBarrier pattern) (adamleex).
Also: issue the x load before the W flood in projections (+1.2-1.4 us); register-direct 12-bit streaming beats TMA at m<=16; F11 8-exponent window saves another 4.5% of bytes.

## Agent and harness methods
- EAT (caoshiyi, #8; arXiv 2610.36848): a supervisor that never edits code reflects at plateaus and evolves the team (roles, prompts, workflow); it never accepts ceiling claims. In one run an added "energy auditor" role found a measurement mismatch and the score reached 3.03x.
- K-Search (caoshiyi; arXiv 2602.19128): evolutionary search over a world model of hypotheses and bottlenecks.
- kernel-autoresearch-plugin (gauravjain14, #7): lanes survey/plan/build/stack/audit; a frozen evaluator; canary cheat kernels; bootstrap CIs over memory placements (1.9% placement noise).
- AshishKumar4: 29 lanes + ~125 cross-model reviewers, then short-lived agents (90 min / 3 attempts) with handoff notes; 28/1031 confirmations won; structural predictions had the wrong sign 0/12.
- infraax: many variants timed inside prepare() per judge run; a prediction and kill line logged before each check.
- keell0renz: judge noise ~0.63% SD per run, so 1% needs 6-8 runs per arm; large gains came only from structural changes.

## Links
github.com/caoshiyi/EAT, caoshiyi/K-Search, caoshiyi/eat-kernels, gauravjain14/kernel-autoresearch-plugin, gauravjain14/muse-okc-submissions, abhishekgahlot2/pi-dsh, adamleex/okc-2026-muse-kernels, ethduke/open-kernel-challenge, EROREx/okc-muse-spark-kernels, joeyhlu/okc-muse-kernels, yigitkucuk/muse-spark-kernels, arihantbansal/open-kernel-challenge, AshishKumar4/open-kernel-challenge, infraax/okc-kernel-challenge, keell0renz/muse-open-kernel-challenge, betitanx/muse-open-kernel-challenge, gonzalezerik/muse-open-kernel-log, akorite/okc-findings, msaroufim/openkernelchallenge-audit
