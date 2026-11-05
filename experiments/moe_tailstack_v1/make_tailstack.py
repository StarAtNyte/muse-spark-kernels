"""Build experiments/moe_tailstack_v1/submission.py from moe_keell0renz_compare_v1 (edits in VAR_SRC only).

TFIN (small skinny builds TINY / TINY2 / MID): no grid barrier before the final pass.
  * own launch counter (atomicInc parity) picks this launch's flag set; the other parity is zeroed at launch start
    (doneA, per-token counters, ss copy) and the DF inter rows the previous launch dirtied go back to sentinel there,
    instead of after a grid barrier at the end;
  * after phase B every CTA credits its phase-B units to per-token counters (red.release, no return) and exits unless
    it owns a final item; a final item waits (ld.acquire) only for its own token's counter.
bs8 (t2_wsi): -DOKC_DF=1 (fence-free data-flag A->B hand-off) on top.
"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
src = (ROOT / 'experiments/moe_keell0renz_compare_v1/submission.py').read_text()
a = src.index("VAR_SRC = r'''"); b = src.index("BW_SRC = r'''")
head, var, tail = src[:a], src[a:b], src[b:]

def rep(s, old, new, count=1):
    n = s.count(old)
    assert n == count, (n, old[:120])
    return s.replace(old, new)

# --- knobs and flag layout
var = rep(var, "#define INTER_SENT_ROWS 128\n", """#define INTER_SENT_ROWS 128
#ifndef OKC_TFIN
#define OKC_TFIN 0   // tailstack: per-token completion counters instead of the grid barrier before the final pass
#endif
#ifndef OKC_TFW
#define OKC_TFW 0    // TFIN final-item poll: nanosleep ns per iteration (0: tight)
#endif
#ifndef OKC_TFR
#define OKC_TFR 0    // TFIN credit: 0 red.release, 1 fence + relaxed atomicAdd
#endif
""")
var = rep(var, "  unsigned long long* dbg;\n};\n", """  unsigned long long* dbg;
  int* tfl;              // TFIN: flag region (scratch[0] after the WGS MFIN flags)
  __nv_bfloat16* idf;    // TFIN: DF inter rows, two parity copies of INTER_SENT_ROWS rows each
};
""")
var = rep(var, "  float* ssp;       // this launch's ss copy\n", "  float* ssp;       // this launch's ss copy\n  int* dap;         // TFIN: this launch's doneA copy\n")
var = rep(var, "__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)\n",
"""__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
#if OKC_TFIN
// TFIN flag region (ints, relative to p.tfl): launch counter, DF dirty rows per parity, doneA / token counters / ss per parity
constexpr int TF_EXIT = 0, TF_DIRTY = 4, TF_DA = 8, TF_TK = TF_DA + 2 * JMAX, TF_TKN = 32, TF_SS = TF_TK + 2 * TF_TKN, TF_SSN = 256;
constexpr int TF_INTS = TF_SS + 2 * TF_SSN;
// DF: this launch's inter copy (parity S.gen)
__device__ __forceinline__ __nv_bfloat16* inter_cur(const Params& p) {
#if OKC_DF
  return reinterpret_cast<const Smem*>(smem_raw)->gen ? p.idf + (size_t)INTER_SENT_ROWS * ID : p.idf;
#else
  return p.inter;
#endif
}
#define INTERP inter_cur(p)
#define DONEA (S.dap)
#else
#define INTERP p.inter
#define DONEA p.doneA
#endif
""")
# --- doneA / inter users of the skinny paths
var = var.replace("ld_acquire(p.doneA + j) < RB_A", "ld_acquire(DONEA + j) < RB_A")
var = var.replace("atomicAdd(p.doneA + jj, b - a);", "atomicAdd(DONEA + jj, b - a);")
var = rep(var, "if (k) { atomicAdd(p.doneA + jj, k);", "if (k) { atomicAdd(DONEA + jj, k);")
var = rep(var, "      if (PB) row = p.inter + (size_t)(st + r) * ID;", "      if (PB) row = INTERP + (size_t)(st + r) * ID;")
var = rep(var, "reinterpret_cast<const uint4*>(p.inter + (size_t)(st + (act ? g : 0)) * ID", "reinterpret_cast<const uint4*>(INTERP + (size_t)(st + (act ? g : 0)) * ID")
var = rep(var, "const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID", "const __nv_bfloat16* row = PB ? INTERP + (size_t)(st + 8 + r) * ID")
var = rep(var, "st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg)", "st_keep_bf16(INTERP + ((size_t)(st + tk) * ID + rb * 8 + gg)", 3)
# --- launch parity
var = rep(var, """    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
""", """#if OKC_TFIN
    // launch parity: every CTA bumps the counter once (wrap 2 G), so it reads G * n (mod 2 G) plus earlier CTAs of this
    // launch; issued here, its (contended) result is first needed after routing
    tf_old = atomicInc(reinterpret_cast<unsigned*>(p.tfl + TF_EXIT), 2u * gridDim.x - 1u);
#else
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
#endif
""")
var = rep(var, """  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
""", """  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
#if OKC_TFIN
    const int gen = tf_old >= gridDim.x ? 1 : 0;
    S.gen = gen;
    S.ssp = reinterpret_cast<float*>(p.tfl + TF_SS + gen * TF_SSN);
    S.dap = p.tfl + TF_DA + gen * JMAX;
#endif
""")
var = rep(var, """  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
""", """#if !OKC_TFIN
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#endif
""")
var = rep(var, """  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {  // padded-row base""", """  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {  // padded-row base""")
# --- tail: barrier -> per-token credits
old_bar = """  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
"""
i = var.index(old_bar)
j = var.index("  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n", i)
old_block = var[i:j]
var = var[:i] + """#if OKC_TFIN
  static_assert(OKC_WS && OKC_FINCOL && OKC_FINPF && !OKC_WGS, "TFIN: skinny WS builds with the column-parallel final pass");
  // every epilogue store of this CTA is done (WS_JOIN above): credit this CTA's phase-B units to the counters of the
  // tokens they cover (release: the stores before the join are ordered before the credit), then only the CTAs that own
  // a final item stay
  if (tid < 32) {
    const int b0 = S.wsr[2], b1 = S.wsr[3], njF = S.njF;
    for (int t = tid; t < p.M; t += 32) {
      int cr = 0;
      for (int jj = njF + b0 / RB_B; b1 > b0 && jj <= njF + (b1 - 1) / RB_B; ++jj) {
        const int n = min(b1, (jj - njF + 1) * RB_B) - max(b0, (jj - njF) * RB_B);
        const int st = S.jS[jj], c = S.jC[jj];
        for (int k = 0; k < c; ++k) cr += (S.ent[st + k] >> 3) == t ? n : 0;
      }
      if (cr) {
        if (OKC_TFR) { __threadfence(); atomicAdd(p.tfl + TF_TK + S.gen * TF_TKN + t, cr); }
        else asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.tfl + TF_TK + S.gen * TF_TKN + t), "r"(cr) : "memory");
      }
    }
  }
  if ((int)blockIdx.x >= p.M * FSEGP) {
    // no final item here: zero the previous launch's (other parity) flags for the next launch, off the critical path
    // (the CTAs without a final item share it; the launcher checks there is at least one)
    const int nf = p.M * FSEGP, o = 1 - S.gen, G = gridDim.x - nf, i0 = ((int)blockIdx.x - nf) * NT + tid;
    float* sso = reinterpret_cast<float*>(p.tfl + TF_SS + o * TF_SSN);
    for (int i = i0; i < TF_SSN; i += G * NT) sso[i] = 0.f;
    for (int i = i0; i < JMAX; i += G * NT) p.tfl[TF_DA + o * JMAX + i] = 0;
    for (int i = i0; i < TF_TKN; i += G * NT) p.tfl[TF_TK + o * TF_TKN + i] = 0;
    // the DF inter rows the previous launch wrote go back to the "not yet written" sentinel
    const int rows = *reinterpret_cast<volatile const int*>(p.tfl + TF_DIRTY + o);
    uint4* ip = reinterpret_cast<uint4*>(p.idf + (size_t)o * INTER_SENT_ROWS * ID);
    for (int i = i0; i < rows * (ID / 8); i += G * NT) ip[i] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    if (i0 == 0) p.tfl[TF_DIRTY + S.gen] = OKC_DF ? min(INTER_SENT_ROWS, p.M * 8) : 0;
    return;
  }
  DBG(4);
#else
""" + old_block + "#endif\n" + var[j:]
# final pass: wait for the item's token before its ss / ybuf loads
var = rep(var, """  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
""", """  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
#if OKC_TFIN
    tf_wait(p, S, tkp, tid);
#endif
""")
var = rep(var, """  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
""", """  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
#if OKC_TFIN
    tf_wait(p, S, tk, tid);
#endif
""")
var = rep(var, "  const int tid = threadIdx.x;\n  DBG(0);\n", "  const int tid = threadIdx.x;\n  DBG(0);\n#if OKC_TFIN\n  unsigned tf_old = 0;   // thread 0: this CTA's launch-counter ticket\n#endif\n")
# tf_wait helper, before moe_fused
var = rep(var, "template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {", """#if OKC_TFIN
// final item of token tk: every phase-B unit of the token's local routes was credited (RB_B per route)
__device__ __forceinline__ void tf_wait(const Params& p, const Smem& S, int tk, int tid) {
  if (tid == 0) {
    int nl = 0;
#pragma unroll
    for (int s = 0; s < 8; ++s) nl += S.pos[tk * 8 + s] >= 0 ? 1 : 0;
    const int* c = p.tfl + TF_TK + S.gen * TF_TKN + tk;
    for (int n = 0; ld_acquire(c) < nl * RB_B && n < 20000000; ++n) { if (OKC_TFW) __nanosleep(OKC_TFW); }
  }
  CTA_SYNC();
}
#endif
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {""")
# launcher
var = rep(var, """  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
""", """  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  p.idf = (__nv_bfloat16*)scratch[2].data_ptr() + (size_t)128 * 4096;
  p.tfl = nullptr;
#if OKC_TFIN
  p.tfl = ib + 8 + 1024 + okc::EMAX + 3200;   // after the WGS MFIN flags (MF_DA + 2 JMAX 16 = 3200)
  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + 3200 + okc::TF_INTS), "TFIN flag region");
  TORCH_CHECK(p.M <= okc::TF_TKN && p.M * 8 <= okc::TF_SSN && p.M * 8 <= INTER_SENT_ROWS, "TFIN: small builds only");
  TORCH_CHECK(p.M * (okc::HD / (okc::NT * 4)) < grid, "TFIN: a CTA without a final item zeroes the next launch's flags");
  TORCH_CHECK((size_t)scratch[2].numel() >= (size_t)(128 + 2 * INTER_SENT_ROWS) * 4096, "TFIN: two DF inter copies");
#endif
""")
# FSEGP must exist before the credit block (it is defined in the FINPF block above the barrier): check
assert var.index("constexpr int FSEGP") < var.index("if ((int)blockIdx.x >= p.M * FSEGP) {")

# --- python side
head = head  # CUDA_SRC untouched
tail = rep(tail, '    inter[128 * HIDDEN:256 * HIDDEN].view(torch.int16).fill_(-1)\n',
           '    inter[128 * HIDDEN:384 * HIDDEN].view(torch.int16).fill_(-1)   # tailstack: two DF parity copies (TFIN)\n')
tail = rep(tail, 'TINY_DEFINES += ("-DOKC_DF=0", "@var")\nTINY2_DEFINES += ("-DOKC_DF=0", "@var")\nMID_DEFINES += ("-DOKC_WSH0=1", "@helper0")\n',
           'TFIN = 1   # tailstack: per-token completion counters instead of the grid barrier (small builds)\n'
           'TINY_DEFINES += ("-DOKC_DF=0", f"-DOKC_TFIN={TFIN}", "@var")\nTINY2_DEFINES += ("-DOKC_DF=0", f"-DOKC_TFIN={TFIN}", "@var")\n'
           'MID_DEFINES += ("-DOKC_WSH0=1", f"-DOKC_TFIN={TFIN}", "@helper0")\n')
tail = rep(tail, "PS_SHAPES = {8: ('tiny2', {}, 't2_wsi')}", "PS_SHAPES = {8: ('tiny2', {'OKC_DF': 1}, 't2_wsi')}   # tailstack: + fence-free DF hand-off at bs8")
hdr = ("# okc: problem=muse_moe_w4a16\n"
       "# tailstack: small builds (TINY / TINY2 / MID) drop the grid barrier before the final pass: launch-parity flag sets\n"
       "#   (zeroed / re-sentineled at the start of the next launch) + per-token phase-B completion counters that only the\n"
       "#   final-item CTAs wait on (OKC_TFIN); bs8 also runs the fence-free data-flag A->B hand-off (OKC_DF=1).\n")
assert head.startswith("# okc: problem=muse_moe_w4a16\n")
head = hdr + head[len("# okc: problem=muse_moe_w4a16\n"):]
out = head + var + tail
(Path(__file__).parent / 'submission.py').write_text(out)
print('ok', len(out))
