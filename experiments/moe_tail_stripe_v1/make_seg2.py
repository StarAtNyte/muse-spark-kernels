"""Build submission.py from base.py: two-segment unit schedule in the WS skinny builds (VAR_SRC only, -DOKC_SEG2=1 in
TINY / TINY2). Every edit is an exact, unique text replacement inside VAR_SRC."""
import sys
src = open('../moe_keell0renz_compare_v1/submission.py').read()
a = src.index("VAR_SRC = r'''"); b = src.index("BW_SRC = r'''")
head, var, tail = src[:a], src[a:b], src[b:]
def rep(old, new):
    global var
    assert var.count(old) == 1, (var.count(old), old[:90])
    var = var.replace(old, new)

# 1. Smem: the two segment ranges per phase
rep("  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges\n",
    "  int wsr[4];                          // WS: this CTA's phase A and phase B unit counts (0, nA, 0, nB)\n"
    "  int wsq[8];                          // WS: per phase (A 0..3, B 4..7) this CTA's segment ranges: start, count, start, count\n")
# 2. the split
rep("""  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {""",
"""  return q;
}
#ifndef OKC_SEG2
#define OKC_SEG2 0
#endif
// Two-segment schedule (SEG2): the J skinny jobs (experts, in order) split into segments [0, J1) and [J1, J),
// J1 = ceil(J / 2); every CTA runs its share of segment 1's units, then its share of segment 2's (same order in phase
// B). Segment 1's phase A is then complete everywhere half way through phase A, so phase B starts on jobs that are
// already done while segment 2's A tail and its doneA credits finish. SEG2 = 0: one segment = the plain split.
// a CTA's units: [s1, s1 + n1) then [s2, s2 + n2), of rbs units per job
__device__ __forceinline__ void seg_share(int J, int rbs, int c, int G, int& s1, int& n1, int& s2, int& n2) {
  const float rG = __fdividef(1.f, (float)G);
  const int J1 = OKC_SEG2 ? (J + 1) >> 1 : J;
  const int u1 = J1 * rbs, u2 = (J - J1) * rbs;
  s1 = muldiv_g(u1, c, G, rG); n1 = muldiv_g(u1, c + 1, G, rG) - s1;
  s2 = muldiv_g(u2, c, G, rG); n2 = muldiv_g(u2, c + 1, G, rG) - s2;
  s2 += u1;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {""")
# 3. unit of local index i (phase pb) + ws_issue through it
rep("#define OKC_WSRF 0\n",
    "#define OKC_WSRF 0\n"
    "// unit id of this CTA's i-th unit of phase pb (the segment ranges set up in moe_fused)\n"
    "__device__ __forceinline__ int ws_v(const Smem& S, int pb, int i) {\n"
    "  const int* w = S.wsq + 4 * pb;\n"
    "  return i < w[1] ? w[0] + i : w[2] + (i - w[1]);\n"
    "}\n")
rep("""  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);""",
"""  (void)a0; (void)b0;
  if (q < nA) ring_issue<false>(p, S, ws_v(S, 0, q), ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, ws_v(S, 1, q - nA), ring + sl * UNIT_BYTES, S.bars + sl);""")
# 4. skinny_ring_ws: count and unit ids from the segment ranges
i0 = var.index("__device__ void skinny_ring_ws(")
i1 = var.index("__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {")
fn = var[i0:i1]
for old, new in (("  const int nU = v1 - v0;\n", "  const int nU = S.wsq[4 * PB + 1] + S.wsq[4 * PB + 3];\n  (void)v0; (void)v1;\n"),
                 ("    const int v = v0 + i;\n", "    const int v = ws_v(S, PB, i);\n"),
                 ("    const int v = v0 + ul, j = njF + v / RB_B,", "    const int v = ws_v(S, 1, ul), j = njF + v / RB_B,")):
    assert fn.count(old) == 1, old
    fn = fn.replace(old, new)
var = var[:i0] + fn + var[i1:]
# 5. ws_helper: unit ids, and the doneA credit at the end of each segment
rep("    const int v = PB ? b0 + (q - nA) : a0 + q;\n", "    const int v = ws_v(S, PB ? 1 : 0, PB ? q - nA : q);\n")
rep("""      if (!OKC_DF && q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);""",
"""      if (!OKC_DF && (q == S.wsq[1] - 1 || q == nA - 1)) {  // a segment of this CTA's phase A is done: credit its units
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int sg = q < S.wsq[1] ? 0 : 2;
          const int u0 = S.wsq[sg], u1 = u0 + S.wsq[sg + 1];
          const int jf = njF + u0 / RB_A, jl = njF + (u1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(u0, (jj - njF) * RB_A), b = min(u1, (jj - njF + 1) * RB_A);""")
# 6. FAST1 preissue (non-split tiny path): the first units of the segment schedule
rep("""      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);""",
"""      static_assert(!SKDYN, "the tiny builds have no run-time pool");
      int fs1, fn1, fs2, fn2;
      seg_share(__popc(m), RB_A, blockIdx.x, gridDim.x, fs1, fn1, fs2, fn2);  // this CTA's phase-A units (as in moe_fused)
      const int n = min(OKC_FAST1, fn1 + fn2);""")
rep("""      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);""",
"""      for (int i = 0; i < n; ++i) {
        const int v = i < fn1 ? fs1 + i : fs2 + (i - fn1);
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);""")
# 7. moe_fused: the segment ranges
rep("""    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);""",
"""    (void)rG;
    seg_share(S.njS, RB_A, c, G, S.wsq[0], S.wsq[1], S.wsq[2], S.wsq[3]);
    seg_share(S.njS, RB_B, c, G, S.wsq[4], S.wsq[5], S.wsq[6], S.wsq[7]);
    S.wsr[0] = 0; S.wsr[1] = S.wsq[1] + S.wsq[3];
    S.wsr[2] = 0; S.wsr[3] = S.wsq[5] + S.wsq[7];""")
head2 = head
for name in ("TINY_DEFINES = (", "TINY2_DEFINES = ("):
    old = name + '"-DOKC_NW=8", '
    assert head2.count(old) == 1
    head2 = head2.replace(old, name + '"-DOKC_SEG2=1", "-DOKC_NW=8", ')
head2 = head2.replace("# okc: problem=muse_moe_w4a16\n",
    "# okc: problem=muse_moe_w4a16\n# tail_seg2: TINY / TINY2 builds run a two-segment unit schedule (OKC_SEG2): each CTA streams its share of the first\n"
    "#   half of the experts, then of the second half, in both phases, so phase B never waits on the global phase-A tail.\n", 1)
open(sys.argv[1] if len(sys.argv) > 1 else 'submission_seg2.py', 'w').write(head2 + var + tail)
print('ok')
