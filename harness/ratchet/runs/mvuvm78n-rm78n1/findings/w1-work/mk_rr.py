"""Make cand_rr.py: OKC_RR=1 => TINY/TINY2/MID (WS skinny ring) process units round-robin over CTAs
(CTA c's i-th unit = i*G + c), so every CTA streams job 0 first, and doneA is credited per job as soon as this
CTA's last unit of that job is done. Phase B of early jobs then overlaps the phase-A tail."""
import sys
src = open(sys.argv[1]).read()
def rep(old, new, cnt=1):
    global src
    n = src.count(old)
    assert n == cnt, (n, old[:80])
    src = src.replace(old, new)

# helper + flag
rep("""__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {""",
"""#ifndef OKC_RR
#define OKC_RR 0
#endif
// OKC_RR: CTA c's i-th unit of an exact floor split [w0, w1) of N units is unit i*G + c (the CTAs with one extra unit
// take the tail in CTA order): a bijection onto [0, N) that walks every job in order on every CTA
__device__ __forceinline__ int rr_unit(int i, int w0, int N, int G, int c) {
#if OKC_RR
  const int L = N / G;
  return i < L ? i * G + c : L * G + (w0 - L * c);
#else
  (void)N; (void)G; (void)c;
  return w0 + i;
#endif
}
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {""")

# FAST1 preissue, SPLIT (MID): job of unit i and its source offset
rep("""      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)""",
    """      const int jj = rr_unit(i, v01, nS1, gridDim.x, blockIdx.x) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)""")
rep("""      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = v0 + i;
        const int e = fex[i];""",
"""      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = rr_unit(i, v0, nS1, gridDim.x, blockIdx.x);
        const int e = fex[i];""")
rep("""      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);""",
"""      for (int i = 0; i < n; ++i) {
        const int v = rr_unit(i, v0, nS, gridDim.x, blockIdx.x);
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);""")
# compute loop
rep("""  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;""",
"""  const int nTot = S.njS * rbs;
  for (int i = 0; i < nU; ++i) {
    const int v = rr_unit(i, v0, nTot, gridDim.x, blockIdx.x);
    const int j = njF + v / rbs;""")
rep("""    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;""",
    """    const int v = rr_unit(ul, v0, nTot, gridDim.x, blockIdx.x), j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;""")
# ws_issue
rep("""  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);""",
"""  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, rr_unit(q, a0, S.njS * RB_A, gridDim.x, blockIdx.x), ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, rr_unit(q - nA, b0, S.njS * RB_B, gridDim.x, blockIdx.x), ring + sl * UNIT_BYTES, S.bars + sl);""")
# helper unit id
rep("""    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;""",
"""    const bool PB = q >= nA;
    const int v = PB ? rr_unit(q - nA, b0, S.njS * RB_B, gridDim.x, blockIdx.x) : rr_unit(q, a0, S.njS * RB_A, gridDim.x, blockIdx.x);""")
# per-job doneA credit
rep("""      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }""",
"""#if OKC_RR
      ++runA;
      if (q == nA - 1 || njF + rr_unit(q + 1, a0, S.njS * RB_A, gridDim.x, blockIdx.x) / RB_A != j) {
        __threadfence();  // this CTA's last unit of job j: credit its run of units now (phase B of j may start)
        __syncwarp();
        if (lane == 0) atomicAdd(p.doneA + j, runA);
        runA = 0;
      }
#else
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }
#endif""")
rep("""  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;""",
    """  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  int runA = 0;
  (void)runA;""")
src = src.replace('name=f"okc_moe_w4a16_v216x_', 'name=f"okc_moe_w4a16_v216rr_')
open(sys.argv[2], 'w').write(src)
print("ok")
