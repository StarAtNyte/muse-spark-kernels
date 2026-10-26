"""Split-KV flash-decode partials over a paged FP8 KV cache (CUDA, mma.sync f16, cp.async ring)."""

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

#define HD 64
#define STAGE 16          // tokens per warp stage
#define KP 144            // bytes per converted f16 row (64 halves + 16 pad)
#define RP 80             // bytes per raw fp8 row slot (64 + 16 pad)

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t hmul2_h(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// four e4m3 bytes -> two f16x2 (exact)
__device__ __forceinline__ void fp8x4_to_h2(uint32_t w, uint32_t& lo, uint32_t& hi) {
#if defined(FD_HW_CVT) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n}"
      : "=r"(lo), "=r"(hi) : "r"(w));
  return;
#endif
  uint32_t tl = prmt(w, 0u, 0x1404u), th = prmt(w, 0u, 0x3424u);
  tl = (tl & 0x80008000u) | ((tl >> 1) & 0x3F803F80u);
  th = (th & 0x80008000u) | ((th >> 1) & 0x3F803F80u);
  lo = hmul2_h(tl, 0x5C005C00u);  // x 2^8 undoes the exponent-bias shift
  hi = hmul2_h(th, 0x5C005C00u);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// grid (splits, HKV, B), 128 threads.  Each warp owns every 4th 16-token stage of the CTA's range.
template <int MT, int S, bool QK8>
__global__ void __launch_bounds__(128)
fd_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
            const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
            const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
            int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = MT * 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;  // scores in log2 domain
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;

  const int nstages = end > begin ? (end - begin + STAGE - 1) / STAGE : 0;

  // ---- per-warp shared layout ----
  constexpr int RAW = 2 * 16 * RP;                       // K and V raw slot
  constexpr int PER_WARP = S * RAW + 2 * 16 * KP;        // ring + converted K and V
  uint8_t* wbase = smem + warp * PER_WARP;
  uint8_t* kf = wbase + S * RAW;
  uint8_t* vf = kf + 16 * KP;

  // ---- Q fragments (f16, or raw e4m3 for the fp8 mma path) ----
  uint32_t qa[MT][4][4];
  if constexpr (QK8) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int row = mt * 16 + g + 8 * half;
        const int token = row >> 3, qh = kh * 8 + (row & 7);
        const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          uint32_t x0 = 0, x1 = 0;
          if (token < L) {
            x0 = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
            x1 = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
          }
          qa[mt][ks][half ? 1 : 0] = x0;  // a0 / a1
          qa[mt][ks][half ? 3 : 2] = x1;  // a2 / a3
        }
      }
    }
  } else {
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int row = mt * 16 + g + 8 * half;
      const int token = row >> 3, qh = kh * 8 + (row & 7);
      const bool ok = token < L;
      const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        // elements d = 16 ks + 2t,2t+1 (a0/a1) and 16 ks + 8 + 2t,2t+1 (a2/a3)
        uint32_t lo = 0, hi = 0, x = 0, y = 0;
        if (ok) {
          uint32_t v0 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 2 * t);
          uint32_t v1 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 8 + 2 * t);
          fp8x4_to_h2(v0 | (v1 << 16), lo, hi);
          x = lo;
          y = hi;
        }
        qa[mt][ks][half ? 1 : 0] = x;       // a0 (row g) / a1 (row g+8): first column pair
        qa[mt][ks][half ? 3 : 2] = y;       // a2 / a3: second column pair
      }
    }
  }
  }

  float o[MT][8][4];
  float m_run[MT][2], l_run[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    m_run[mt][0] = m_run[mt][1] = -INFINITY;
    l_run[mt][0] = l_run[mt][1] = 0.f;
#pragma unroll
    for (int d = 0; d < 8; d++)
#pragma unroll
      for (int r = 0; r < 4; r++) o[mt][d][r] = 0.f;
  }

  const int pshift = __ffs(page) - 1, pmask = page - 1;
  auto row_of = [&](int tok) -> long {
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    return phys < 0 ? -1L : ((long)phys << pshift) + (tok & pmask);
  };
  // stage s of this warp covers tokens begin + (warp + 4 s) * STAGE ...
  auto issue = [&](int s, long r0, long r1) {
    if (s < (nstages - warp + 3) / 4) {
      uint8_t* slot = wbase + (s % S) * RAW;
      const int c = lane & 3;
      const int tok0 = lane >> 2;
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const long r = i ? r1 : r0;
        const int tok = tok0 + 8 * i;
        // invalid rows read row 0 (masked later)
        const long rr = r < 0 ? 0 : r;
        cp16(slot + tok * RP + c * 16, KC + (rr * HKV + kh) * HD + c * 16);
        cp16(slot + 16 * RP + tok * RP + c * 16, VC + (rr * HKV + kh) * HD + c * 16);
      }
    }
    cp_commit();
  };

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto stage_tok = [&](int s) { return begin + (warp + 4 * s) * STAGE; };
  // gather-index prefetch for stage s: tokens stage_tok(s) + lane/4 and +8
  long ra0 = -1, ra1 = -1, rb0 = -1, rb1 = -1;
  auto load_rows = [&](int s, long& x0, long& x1) {
    const int tb = stage_tok(s) + (lane >> 2);
    x0 = x1 = -1;
    if (s < my_stages) {
      if (tb < end) x0 = row_of(tb);
      if (tb + 8 < end) x1 = row_of(tb + 8);
    }
  };
  {
    long pr0[S], pr1[S];
#pragma unroll
    for (int s = 0; s < S; s++) load_rows(s, pr0[s], pr1[s]);
#pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, pr0[s], pr1[s]);
    rb0 = pr0[S - 1];
    rb1 = pr1[S - 1];
  }

  const int srow = lane & 7;  // ldmatrix row within an 8x8 tile
  const int smat = lane >> 3; // matrix index within x4

#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1, rb0, rb1);
    load_rows(s + S, rb0, rb1);
    const uint8_t* slot = wbase + (s % S) * RAW;
    // ---- convert K and V rows of this stage to f16 (lane handles one 16B raw chunk x 2) ----
    {
      const int tk = lane >> 1, half16 = (lane & 1) * 32;  // 16 tokens x 2 halves of 32 bytes
#pragma unroll
      for (int which = (QK8 ? 1 : 0); which < 2; which++) {
        const uint8_t* src = slot + which * 16 * RP + tk * RP + half16;
        uint8_t* dst = (which ? vf : kf) + tk * KP + half16 * 2;
        const uint4 u0 = *reinterpret_cast<const uint4*>(src);
        const uint4 u1 = *reinterpret_cast<const uint4*>(src + 16);
        uint4 o0, o1, o2, o3;
        fp8x4_to_h2(u0.x, o0.x, o0.y);
        fp8x4_to_h2(u0.y, o0.z, o0.w);
        fp8x4_to_h2(u0.z, o1.x, o1.y);
        fp8x4_to_h2(u0.w, o1.z, o1.w);
        fp8x4_to_h2(u1.x, o2.x, o2.y);
        fp8x4_to_h2(u1.y, o2.z, o2.w);
        fp8x4_to_h2(u1.z, o3.x, o3.y);
        fp8x4_to_h2(u1.w, o3.z, o3.w);
        *reinterpret_cast<uint4*>(dst) = o0;
        *reinterpret_cast<uint4*>(dst + 16) = o1;
        *reinterpret_cast<uint4*>(dst + 32) = o2;
        *reinterpret_cast<uint4*>(dst + 48) = o3;
      }
    }
    __syncwarp();

    const int tb = stage_tok(s);
    const bool full_vis = (tb + STAGE <= end) && (L == 1 || tb + STAGE - 1 <= prefix) &&
                          (!window || L == 1 || tb > prefix + L - 1 - window);
    // ---- S = Q K^T for each m-tile: 2 n-tiles of 8 tokens ----
    uint32_t kb[2][4][2];
    if constexpr (QK8) {
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          const uint8_t* kr = slot + (nt * 8 + g) * RP + 32 * ks + 4 * t;
          kb[nt][ks][0] = *reinterpret_cast<const uint32_t*>(kr);
          kb[nt][ks][1] = *reinterpret_cast<const uint32_t*>(kr + 16);
        }
    } else {
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4(r0, r1, r2, r3, kf + (nt * 8 + srow) * KP + pr * 64 + smat * 16);
        kb[nt][2 * pr][0] = r0; kb[nt][2 * pr][1] = r1;
        kb[nt][2 * pr + 1][0] = r2; kb[nt][2 * pr + 1][1] = r3;
      }
    }
    uint32_t vb[4][4];
#pragma unroll
    for (int dp = 0; dp < 4; dp++)
      ldsm4t(vb[dp][0], vb[dp][1], vb[dp][2], vb[dp][3],
             vf + ((smat & 1) * 8 + srow) * KP + dp * 32 + (smat >> 1) * 16);

#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float sacc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      if constexpr (QK8) {
#pragma unroll
        for (int ks = 0; ks < 2; ks++)
#pragma unroll
          for (int nt = 0; nt < 2; nt++)
            mma_e4m3(sacc[nt], qa[mt][ks][0], qa[mt][ks][1], qa[mt][ks][2], qa[mt][ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      } else {
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
          mma_f16(sacc[nt], qa[mt][ks][0], qa[mt][ks][1], qa[mt][ks][2], qa[mt][ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      }

      // masking (only on boundary stages), online softmax (two rows per thread: g and g+8)
      float mx[2] = {-INFINITY, -INFINITY};
      float sv[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sv[nt][r] = sacc[nt][r];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = mt * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sv[nt][r] = -INFINITY;
          }
      }
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sv[nt][r]);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;  // running max lives in the scaled log2 domain
      }
      float corr[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        newm[h] = fmaxf(m_run[mt][h], mx[h]);
        const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[mt][h] - safe);
        newm[h] = safe == 0.f && newm[h] == -INFINITY ? -INFINITY : newm[h];
      }
      uint32_t pa[4];
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int h = r >> 1;
          const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
          p[r] = exp2f(fmaf(sv[nt][r], sc2, -safe));
          rs[h] += p[r];
        }
        pa[nt * 2 + 0] = pack_h2(p[0], p[1]);  // row g
        pa[nt * 2 + 1] = pack_h2(p[2], p[3]);  // row g+8
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[mt][h] = l_run[mt][h] * corr[h] + rs[h];
        m_run[mt][h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[mt][d][0] *= corr[0]; o[mt][d][1] *= corr[0];
          o[mt][d][2] *= corr[1]; o[mt][d][3] *= corr[1];
        }
      }
      // PV: A = P (a0: row g tokens 0-7, a1: row g+8 tokens 0-7, a2: row g tokens 8-15, a3: row g+8 tokens 8-15)
#pragma unroll
      for (int dp = 0; dp < 4; dp++) {
        mma_f16(o[mt][2 * dp], pa[0], pa[1], pa[2], pa[3], vb[dp][0], vb[dp][1]);
        mma_f16(o[mt][2 * dp + 1], pa[0], pa[1], pa[2], pa[3], vb[dp][2], vb[dp][3]);
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();

  // ---- merge the four warps' states ----
  float* sm_m = reinterpret_cast<float*>(smem);                 // [4][BM]
  float* sm_l = sm_m + 4 * BM;                                  // [4][BM]
  float* sm_o = sm_l + 4 * BM;                                  // [4][BM][HD]
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = mt * 16 + g + 8 * h;
      if (t == 0) {
        sm_m[warp * BM + row] = m_run[mt][h];
        sm_l[warp * BM + row] = l_run[mt][h];
      }
#pragma unroll
      for (int d = 0; d < 8; d++) {
        sm_o[(warp * BM + row) * HD + d * 8 + 2 * t] = o[mt][d][2 * h];
        sm_o[(warp * BM + row) * HD + d * 8 + 2 * t + 1] = o[mt][d][2 * h + 1];
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
    const int row = idx / HD, d = idx % HD;
    float mm = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mm = fmaxf(mm, sm_m[w * BM + row]);
    float ls = 0.f, acc = 0.f;
    if (mm != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float f = sm_m[w * BM + row] == -INFINITY ? 0.f : exp2f(sm_m[w * BM + row] - mm);
        ls += sm_l[w * BM + row] * f;
        acc += sm_o[(w * BM + row) * HD + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mm + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}


// Verify blocks: 64 query rows (8 heads x 8 tokens) per KV head -> one warp per 16-row m-tile; the four warps share each
// 16-token K/V stage (converted once per CTA), so no cross-warp merge is needed.
template <int S>
__global__ void __launch_bounds__(128)
fd_partials_v(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
              const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
              const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
              int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 64;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
  const int nstages = end > begin ? (end - begin + STAGE - 1) / STAGE : 0;

  constexpr int RAW = 2 * 16 * RP;
  uint8_t* ring = smem;
  uint8_t* kf = smem + S * RAW;            // [2][16 * KP]
  uint8_t* vf = kf + 2 * 16 * KP;          // [2][16 * KP]

  // Q fragments for this warp's m-tile (rows warp*16 + g and +8)
  uint32_t qa[4][4];
#pragma unroll
  for (int half = 0; half < 2; half++) {
    const int row = warp * 16 + g + 8 * half;
    const int token = row >> 3, qh = kh * 8 + (row & 7);
    const bool ok = token < L;
    const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      uint32_t lo = 0, hi = 0;
      if (ok) {
        uint32_t v0 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 2 * t);
        uint32_t v1 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 8 + 2 * t);
        fp8x4_to_h2(v0 | (v1 << 16), lo, hi);
      }
      qa[ks][half ? 1 : 0] = lo;
      qa[ks][half ? 3 : 2] = hi;
    }
  }
  float o[8][4];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int d = 0; d < 8; d++)
#pragma unroll
    for (int r = 0; r < 4; r++) o[d][r] = 0.f;

  const int which = tid >> 6, tk = (tid >> 2) & 15, c4 = tid & 3;
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  auto row_of = [&](int tok) -> long {
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    return phys < 0 ? -1L : ((long)phys << pshift) + (tok & pmask);
  };
  auto load_row = [&](int s) -> long {
    const int tok = begin + s * STAGE + tk;
    return (s < nstages && tok < end) ? row_of(tok) : -1L;
  };
  auto issue = [&](int s, long r) {
    if (s < nstages) {
      const long rr = r < 0 ? 0 : r;
      const uint8_t* base = which ? VC : KC;
      cp16(ring + (s % S) * RAW + which * 16 * RP + tk * RP + c4 * 16, base + (rr * HKV + kh) * HD + c4 * 16);
    }
    cp_commit();
  };
  {
    long pr[S];
#pragma unroll
    for (int s = 0; s < S; s++) pr[s] = load_row(s);
#pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, pr[s]);
    long nxt = pr[S - 1];
    const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
    for (int s = 0; s < nstages; s++) {
      cp_wait<S - 2>();
      issue(s + S - 1, nxt);
      nxt = load_row(s + S);
      // convert this thread's own chunk (K or V row segment) to f16
      {
        const uint8_t* src = ring + (s % S) * RAW + which * 16 * RP + tk * RP + c4 * 16;
        const uint4 u = *reinterpret_cast<const uint4*>(src);
        uint4 o0, o1;
        fp8x4_to_h2(u.x, o0.x, o0.y);
        fp8x4_to_h2(u.y, o0.z, o0.w);
        fp8x4_to_h2(u.z, o1.x, o1.y);
        fp8x4_to_h2(u.w, o1.z, o1.w);
        uint8_t* dst = (which ? vf : kf) + (s & 1) * 16 * KP + tk * KP + c4 * 32;
        *reinterpret_cast<uint4*>(dst) = o0;
        *reinterpret_cast<uint4*>(dst + 16) = o1;
      }
      __syncthreads();
      const uint8_t* kfs = kf + (s & 1) * 16 * KP;
      const uint8_t* vfs = vf + (s & 1) * 16 * KP;
      const int tb = begin + s * STAGE;
      const bool full_vis = (tb + STAGE <= end) && (L == 1 || tb + STAGE - 1 <= prefix) &&
                            (!window || L == 1 || tb > prefix + L - 1 - window);
      uint32_t kb[2][4][2];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int pr2 = 0; pr2 < 2; pr2++) {
          uint32_t r0, r1, r2, r3;
          ldsm4(r0, r1, r2, r3, kfs + (nt * 8 + srow) * KP + pr2 * 64 + smat * 16);
          kb[nt][2 * pr2][0] = r0; kb[nt][2 * pr2][1] = r1;
          kb[nt][2 * pr2 + 1][0] = r2; kb[nt][2 * pr2 + 1][1] = r3;
        }
      uint32_t vb[4][4];
#pragma unroll
      for (int dp = 0; dp < 4; dp++)
        ldsm4t(vb[dp][0], vb[dp][1], vb[dp][2], vb[dp][3], vfs + ((smat & 1) * 8 + srow) * KP + dp * 32 + (smat >> 1) * 16);
      float sacc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
          mma_f16(sacc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      float mx[2] = {-INFINITY, -INFINITY};
      float sv[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sv[nt][r] = sacc[nt][r];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = warp * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sv[nt][r] = -INFINITY;
          }
      }
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sv[nt][r]);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;
      }
      float corr[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        newm[h] = fmaxf(m_run[h], mx[h]);
        const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[h] - safe);
      }
      uint32_t pa[4];
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int h = r >> 1;
          const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
          p[r] = exp2f(fmaf(sv[nt][r], sc2, -safe));
          rs[h] += p[r];
        }
        pa[nt * 2 + 0] = pack_h2(p[0], p[1]);
        pa[nt * 2 + 1] = pack_h2(p[2], p[3]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[h] = l_run[h] * corr[h] + rs[h];
        m_run[h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[d][0] *= corr[0]; o[d][1] *= corr[0];
          o[d][2] *= corr[1]; o[d][3] *= corr[1];
        }
      }
#pragma unroll
      for (int dp = 0; dp < 4; dp++) {
        mma_f16(o[2 * dp], pa[0], pa[1], pa[2], pa[3], vb[dp][0], vb[dp][1]);
        mma_f16(o[2 * dp + 1], pa[0], pa[1], pa[2], pa[3], vb[dp][2], vb[dp][3]);
      }
    }
  }
  cp_wait<0>();
  // rows owned by this warp: no cross-warp merge
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = warp * 16 + g + 8 * h;
    const float ls = l_run[h];
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
    for (int d = 0; d < 8; d++) {
      PART[(base_row + row) * HD + d * 8 + 2 * t] = o[d][2 * h] * inv * value_scale;
      PART[(base_row + row) * HD + d * 8 + 2 * t + 1] = o[d][2 * h + 1] * inv * value_scale;
    }
    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}

template <int S>
static void launch_v(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                     const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                     int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  const size_t dyn = (size_t)S * 2 * 16 * RP + 4 * 16 * KP;
  cudaFuncSetAttribute(fd_partials_v<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  fd_partials_v<S><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                          reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                          table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---------------------------------------------------------------------------------------------------------------
// fd2: decode (one query token, 8 heads per KV head) with 32-token stages.  K and V stay raw FP8 in shared memory:
// K fragments come straight from ldmatrix (e4m3 mma), V fragments from ldmatrix.trans + byte permutes + cvt.
// ---------------------------------------------------------------------------------------------------------------
#define ST2 32
#define RMAX 4096

__device__ __forceinline__ void mma_qk8(float* c, uint32_t qa0, uint32_t qa2, uint32_t b0, uint32_t b1) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(qa0), "r"(0u), "r"(qa2), "r"(0u), "r"(b0), "r"(b1));
#else
  // exact emulation with f16 mma: each fp8 register holds 4 consecutive k, used as two k16 slots
  uint32_t ql, qh, kl, kh2;
  fp8x4_to_h2(qa0, ql, qh);
  fp8x4_to_h2(b0, kl, kh2);
  mma_f16(c, ql, 0u, qh, 0u, kl, kh2);
  fp8x4_to_h2(qa2, ql, qh);
  fp8x4_to_h2(b1, kl, kh2);
  mma_f16(c, ql, 0u, qh, 0u, kl, kh2);
#endif
}
// two e4m3 bytes (low 16 bits of x) -> f16x2
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  uint32_t r;
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %1;\ncvt.rn.f16x2.e4m3x2 %0, a;\n}" : "=r"(r) : "r"(x));
  return r;
#else
  uint32_t lo, hi;
  fp8x4_to_h2(x, lo, hi);
  return lo;
#endif
}

template <int S>
__global__ void __launch_bounds__(128)
fd2_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int HQ, int HKV,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 16;
  constexpr int SLOT = 2 * ST2 * RP;  // K and V raw rows of one stage
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + 1);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
  const int n = end > begin ? end - begin : 0;
  const int nstages = (n + ST2 - 1) / ST2;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + 31) / 32) * 32;
  uint8_t* wbase = smem + rmax_rt * 4 + warp * (S * SLOT);
  // physical cache row of every token of this CTA's range
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  // Q fragments: row g = query head kh*8+g, e4m3 bytes d = 32 ks + 4 t.. and +16
  uint32_t q0[2], q2[2];
  {
    const uint8_t* qp = Q + ((long)b * HQ + kh * 8 + g) * HD;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      q0[ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
      q2[ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
    }
  }
  __syncthreads();  // rows[] ready

  float o[8][2];  // [d-tile (E0,O0,E1,O1,...)][row g, columns 2t, 2t+1]
  float m_run = -INFINITY, l_run = 0.f;
#pragma unroll
  for (int d = 0; d < 8; d++) o[d][0] = o[d][1] = 0.f;

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + 4 * s) * ST2;  // offset within the CTA range
      const int c = lane & 3;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int tk = (lane >> 2) + 8 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = (rr * HKV + kh) * HD + c * 16;
        cp16(slot + tk * RP + c * 16, KC + goff);
        cp16(slot + ST2 * RP + tk * RP + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    const uint8_t* slot = wbase + (s % S) * SLOT;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP;
    const int tb = begin + (warp + 4 * s) * ST2;
    // ---- S = Q K^T over 4 n-tiles of 8 tokens ----
    float sacc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#pragma unroll
      for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      uint32_t k0, k1, k2, k3;
      ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP + smat * 16);
      mma_qk8(sacc[nt], q0[0], q2[0], k0, k1);
      mma_qk8(sacc[nt], q0[1], q2[1], k2, k3);
    }
    // ---- softmax over the 32 columns of row g ----
    const bool full_vis = tb + ST2 <= end;
    float sv[4][2];
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int r = 0; r < 2; r++) {
        float v = sacc[nt][r];
        if (!full_vis && tb + nt * 8 + 2 * t + r >= end) v = -INFINITY;
        sv[nt][r] = v;
        mx = fmaxf(mx, v);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx *= sc2;
    const float newm = fmaxf(m_run, mx);
    const float safe = newm == -INFINITY ? 0.f : newm;
    const float corr = exp2f(m_run - safe);
    float rs = 0.f;
    uint32_t pa[4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
      rs += p0 + p1;
      pa[nt] = pack_h2(p0, p1);
    }
    rs += __shfl_xor_sync(0xffffffffu, rs, 1);
    rs += __shfl_xor_sync(0xffffffffu, rs, 2);
    l_run = l_run * corr + rs;
    m_run = newm;
    if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
      for (int d = 0; d < 8; d++) { o[d][0] *= corr; o[d][1] *= corr; }
    }
    // ---- O += P V: two 16-token k-tiles ----
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        // pr selects d bytes [32 pr, 32 pr + 32): matrices (tok 0-7 | 8-15) x (m = 2 pr, 2 pr + 1)
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP + pr * 32 + (smat >> 1) * 16);
        // r0: (tok 0-7, m=2pr)  r1: (tok 8-15, m=2pr)  r2: (tok 0-7, m=2pr+1)  r3: (tok 8-15, m=2pr+1)
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
        float acc[4];
        acc[0] = o[4 * pr + 0][0]; acc[1] = o[4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
        o[4 * pr + 0][0] = acc[0]; o[4 * pr + 0][1] = acc[1];
        acc[0] = o[4 * pr + 1][0]; acc[1] = o[4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
        o[4 * pr + 1][0] = acc[0]; o[4 * pr + 1][1] = acc[1];
        acc[0] = o[4 * pr + 2][0]; acc[1] = o[4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
        o[4 * pr + 2][0] = acc[0]; o[4 * pr + 2][1] = acc[1];
        acc[0] = o[4 * pr + 3][0]; acc[1] = o[4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
        o[4 * pr + 3][0] = acc[0]; o[4 * pr + 3][1] = acc[1];
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();

  // ---- merge the four warps (rows g only: 8 query heads) ----
  float* sm_m = reinterpret_cast<float*>(smem);   // [4][8]
  float* sm_l = sm_m + 32;                        // [4][8]
  float* sm_o = sm_l + 32;                        // [4][8][64]
  if (t == 0) {
    sm_m[warp * 8 + g] = m_run;
    sm_l[warp * 8 + g] = l_run;
  }
#pragma unroll
  for (int mm = 0; mm < 4; mm++) {
    // tile 2mm = E (d = 16 mm + 2 n), tile 2mm+1 = O (d = 16 mm + 2 n + 1), n = 2t + c
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[2 * mm][cc];
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[2 * mm + 1][cc];
    }
  }
  __syncthreads();
  for (int idx = tid; idx < 8 * HD; idx += 128) {
    const int row = idx / HD, d = idx % HD;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mmx = fmaxf(mmx, sm_m[w * 8 + row]);
    float ls = 0.f, acc = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float f = sm_m[w * 8 + row] == -INFINITY ? 0.f : exp2f(sm_m[w * 8 + row] - mmx);
        ls += sm_l[w * 8 + row] * f;
        acc += sm_o[(w * 8 + row) * 64 + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mmx + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}


// ---- fd3: one CTA serves BOTH kv heads (HKV == 2): rows are fetched as whole 128 B lines ------------------------
#define RP3 144   // bytes per K/V smem row: 128 + 16 pad

template <int S>
__global__ void __launch_bounds__(128)
fd3_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int HQ, int HKV,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 16;
  constexpr int SLOT = 2 * ST2 * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + 1);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int nstages = (n + ST2 - 1) / ST2;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + 31) / 32) * 32;
  uint8_t* wbase = smem + rmax_rt * 4 + warp * (S * SLOT);
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  uint32_t q0[2][2], q2[2][2];
#pragma unroll
  for (int kh = 0; kh < 2; kh++) {
    const uint8_t* qp = Q + ((long)b * HQ + kh * 8 + g) * HD;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      q0[kh][ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
      q2[kh][ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
    }
  }
  __syncthreads();

  float o[2][8][2];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int kh = 0; kh < 2; kh++)
#pragma unroll
    for (int d = 0; d < 8; d++) o[kh][d][0] = o[kh][d][1] = 0.f;

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + 4 * s) * ST2;
      const int c = lane & 7;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int tk = (lane >> 3) + 4 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = rr * 128 + c * 16;
        cp16(slot + tk * RP3 + c * 16, KC + goff);
        cp16(slot + ST2 * RP3 + tk * RP3 + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    const uint8_t* slot = wbase + (s % S) * SLOT;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP3;
    const int tb = begin + (warp + 4 * s) * ST2;
    const bool full_vis = tb + ST2 <= end;
#pragma unroll
    for (int kh = 0; kh < 2; kh++) {
      float sacc[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
        uint32_t k0, k1, k2, k3;
        ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
        mma_qk8(sacc[nt], q0[kh][0], q2[kh][0], k0, k1);
        mma_qk8(sacc[nt], q0[kh][1], q2[kh][1], k2, k3);
      }
      float sv[4][2];
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          float v = sacc[nt][r];
          if (!full_vis && tb + nt * 8 + 2 * t + r >= end) v = -INFINITY;
          sv[nt][r] = v;
          mx = fmaxf(mx, v);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx *= sc2;
      const float newm = fmaxf(m_run[kh], mx);
      const float safe = newm == -INFINITY ? 0.f : newm;
      const float corr = exp2f(m_run[kh] - safe);
      float rs = 0.f;
      uint32_t pa[4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
        rs += p0 + p1;
        pa[nt] = pack_h2(p0, p1);
      }
      rs += __shfl_xor_sync(0xffffffffu, rs, 1);
      rs += __shfl_xor_sync(0xffffffffu, rs, 2);
      l_run[kh] = l_run[kh] * corr + rs;
      m_run[kh] = newm;
      if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) { o[kh][d][0] *= corr; o[kh][d][1] *= corr; }
      }
#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
#pragma unroll
        for (int pr = 0; pr < 2; pr++) {
          uint32_t r0, r1, r2, r3;
          ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
          const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
          const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
          const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
          const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
          float acc[4];
          acc[0] = o[kh][4 * pr + 0][0]; acc[1] = o[kh][4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
          o[kh][4 * pr + 0][0] = acc[0]; o[kh][4 * pr + 0][1] = acc[1];
          acc[0] = o[kh][4 * pr + 1][0]; acc[1] = o[kh][4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
          o[kh][4 * pr + 1][0] = acc[0]; o[kh][4 * pr + 1][1] = acc[1];
          acc[0] = o[kh][4 * pr + 2][0]; acc[1] = o[kh][4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
          o[kh][4 * pr + 2][0] = acc[0]; o[kh][4 * pr + 2][1] = acc[1];
          acc[0] = o[kh][4 * pr + 3][0]; acc[1] = o[kh][4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
          o[kh][4 * pr + 3][0] = acc[0]; o[kh][4 * pr + 3][1] = acc[1];
        }
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();

  // merge the four warps for each kv head
  float* sm_m = reinterpret_cast<float*>(smem);   // [2][4][8]
  float* sm_l = sm_m + 64;                        // [2][4][8]
  float* sm_o = sm_l + 64;                        // [2][4][8][64]
#pragma unroll
  for (int kh = 0; kh < 2; kh++) {
    if (t == 0) {
      sm_m[(kh * 4 + warp) * 8 + g] = m_run[kh];
      sm_l[(kh * 4 + warp) * 8 + g] = l_run[kh];
    }
#pragma unroll
    for (int mm = 0; mm < 4; mm++)
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        sm_o[((kh * 4 + warp) * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[kh][2 * mm][cc];
        sm_o[((kh * 4 + warp) * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[kh][2 * mm + 1][cc];
      }
  }
  __syncthreads();
  for (int idx = tid; idx < 2 * 8 * HD; idx += 128) {
    const int kh = idx / (8 * HD), rem = idx % (8 * HD);
    const int row = rem / HD, d = rem % HD;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mmx = fmaxf(mmx, sm_m[(kh * 4 + w) * 8 + row]);
    float ls = 0.f, acc = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float mw = sm_m[(kh * 4 + w) * 8 + row];
        const float f = mw == -INFINITY ? 0.f : exp2f(mw - mmx);
        ls += sm_l[(kh * 4 + w) * 8 + row] * f;
        acc += sm_o[((kh * 4 + w) * 8 + row) * 64 + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mmx + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}

template <int S>
static void launch3(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                    const float* VS, float* PART, float* LSE, int B, int HQ, int HKV, int table_n, int page, int window,
                    int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP3;
  size_t dyn = (size_t)(((chunk + 31) / 32) * 32) * 4 + (size_t)4 * S * SLOT;
  const size_t merge = (size_t)(128 + 2 * 4 * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd3_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  fd3_partials<S><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                         reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, HQ, HKV,
                                         table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---- fd3v: verify blocks (8 tokens x 8 heads = 64 query rows per KV head), both KV heads per CTA, 128 B row fetches ----
// warp = (kv head, 32-row half); the four warps share each 64-token K/V tile; no cross-warp merge is needed.
#define T3V 64
static int g_fd2_dbg = 0;
extern "C" void fd2_set_dbg(int v) { g_fd2_dbg = v; }
__device__ __forceinline__ void mma_q16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  uint32_t gl, gh, hl, hh, bl, bh;
  fp8x4_to_h2(a0, gl, gh); fp8x4_to_h2(a1, hl, hh); fp8x4_to_h2(b0, bl, bh);
  mma_f16(c, gl, hl, gh, hh, bl, bh);
  fp8x4_to_h2(a2, gl, gh); fp8x4_to_h2(a3, hl, hh); fp8x4_to_h2(b1, bl, bh);
  mma_f16(c, gl, hl, gh, hh, bl, bh);
#endif
}

template <int S>
__global__ void __launch_bounds__(128)
fd3v_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
              const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
              const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
              int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 64;
  constexpr int SLOTB = 2 * T3V * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kh = warp >> 1, mhalf = warp & 1;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int ntiles = (n + T3V - 1) / T3V;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + T3V - 1) / T3V) * T3V;
  uint8_t* ring = smem + rmax_rt * 4;
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  // Q fragments (e4m3 bytes): m-tile mg = 2 mhalf + mt, rows g and g + 8
  // f16 A fragments of the four k16 steps (QK runs on f16 mma: the e4m3 mma.sync is emulated on sm_90)
  uint32_t qa[2][4][4];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    uint32_t xr[2][2][2];  // [half][ks][lo/hi segment]
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int row = (mhalf * 2 + mt) * 16 + g + 8 * half;
      const int token = row >> 3, qh = kh * 8 + (row & 7);
      const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        xr[half][ks][0] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
        xr[half][ks][1] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
      }
    }
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int seg2 = 0; seg2 < 2; seg2++) {
        uint32_t gl, gh, hl, hh;
        fp8x4_to_h2(xr[0][ks][seg2], gl, gh);
        fp8x4_to_h2(xr[1][ks][seg2], hl, hh);
        qa[mt][2 * ks + seg2][0] = gl; qa[mt][2 * ks + seg2][1] = hl;
        qa[mt][2 * ks + seg2][2] = gh; qa[mt][2 * ks + seg2][3] = hh;
      }
  }
  __syncthreads();  // rows[] ready

  float o[2][8][4];
  float m_run[2][2], l_run[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    m_run[mt][0] = m_run[mt][1] = -INFINITY;
    l_run[mt][0] = l_run[mt][1] = 0.f;
#pragma unroll
    for (int d = 0; d < 8; d++) o[mt][d][0] = o[mt][d][1] = o[mt][d][2] = o[mt][d][3] = 0.f;
  }

  auto issue = [&](int s) {
    if (s < ntiles) {
      uint8_t* slot = ring + (s % S) * SLOTB;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int ci = tid + 128 * i;
        const int which = ci >> 9, rem = ci & 511;
        const int row = rem >> 3, c = rem & 7;
        const int li = s * T3V + row;
        const long rr = li < n ? rows[li] : 0;
        if (!(dbg & 8)) cp16(slot + which * (T3V * RP3) + row * RP3 + c * 16, (which ? VC : KC) + rr * 128 + c * 16);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < ntiles; s++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(s + S - 1);
    const uint8_t* slot = ring + (s % S) * SLOTB;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + T3V * RP3;
    const int tb = begin + s * T3V;
    const bool full_vis = (tb + T3V <= end) && (tb + T3V - 1 <= prefix) && (!window || tb > prefix + L - 1 - window);
    uint32_t pa[2][8][2];
    float sall[2][8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      uint32_t kk4[4], bl[4], bh[4];
      if (!(dbg & 1)) {
      ldsm4(kk4[0], kk4[1], kk4[2], kk4[3], kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
#pragma unroll
      for (int j = 0; j < 4; j++) fp8x4_to_h2(kk4[j], bl[j], bh[j]);
      }
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int r = 0; r < 4; r++) sall[mt][nt][r] = (dbg & 1) ? (float)(nt + r) : 0.f;
        if (!(dbg & 1))
#pragma unroll
        for (int j = 0; j < 4; j++) mma_f16(sall[mt][nt], qa[mt][j][0], qa[mt][j][1], qa[mt][j][2], qa[mt][j][3], bl[j], bh[j]);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
      float (&sacc)[8][4] = sall[mt];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = (mhalf * 2 + mt) * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sacc[nt][r] = -INFINITY;
          }
      }
      float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sacc[nt][r]);
      float corr[2], safe[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;
        newm[h] = fmaxf(m_run[mt][h], mx[h]);
        safe[h] = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[mt][h] - safe[h]);
      }
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          p[r] = (dbg & 2) ? fmaf(sacc[nt][r], sc2, -safe[r >> 1]) : exp2f(fmaf(sacc[nt][r], sc2, -safe[r >> 1]));
          rs[r >> 1] += p[r];
        }
        pa[mt][nt][0] = pack_h2(p[0], p[1]);  // row g
        pa[mt][nt][1] = pack_h2(p[2], p[3]);  // row g + 8
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[mt][h] = l_run[mt][h] * corr[h] + rs[h];
        m_run[mt][h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[mt][d][0] *= corr[0]; o[mt][d][1] *= corr[0];
          o[mt][d][2] *= corr[1]; o[mt][d][3] *= corr[1];
        }
      }
    }
    // ---- O += P V: four 16-token k-steps; the V fragments are shared by both m-tiles ----
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      if (dbg & 4) break;
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
          const uint32_t a0 = pa[mt][2 * kk][0], a1 = pa[mt][2 * kk][1], a2 = pa[mt][2 * kk + 1][0], a3 = pa[mt][2 * kk + 1][1];
          mma_f16(o[mt][4 * pr + 0], a0, a1, a2, a3, bE0, bE1);
          mma_f16(o[mt][4 * pr + 1], a0, a1, a2, a3, bO0, bO1);
          mma_f16(o[mt][4 * pr + 2], a0, a1, a2, a3, cE0, cE1);
          mma_f16(o[mt][4 * pr + 3], a0, a1, a2, a3, cO0, cO1);
        }
      }
    }
  }
  cp_wait<0>();
  // rows owned by this warp: no cross-warp merge.  Column mapping of the E/O d-tile pairs: d = 16 mm + 2 (2 t + c) + {0: E, 1: O}.
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = (mhalf * 2 + mt) * 16 + g + 8 * h;
      const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
      const float ls = l_run[mt][h];
      const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
      for (int mm = 0; mm < 4; mm++)
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
          PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc)] = o[mt][2 * mm][2 * h + cc] * inv * value_scale;
          PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc) + 1] = o[mt][2 * mm + 1][2 * h + cc] * inv * value_scale;
        }
      if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[mt][h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
    }
}

template <int S>
static void launch3v(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                     const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                     int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  const size_t dyn = (size_t)(((chunk + T3V - 1) / T3V) * T3V) * 4 + (size_t)S * 2 * T3V * RP3;
  cudaFuncSetAttribute(fd3v_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  fd3v_partials<S><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                          reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                          table_n, page, window, splits, chunk, sm_scale, scaled, g_fd2_dbg);
}

template <int S>
static void launch2(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                    const float* VS, float* PART, float* LSE, int B, int HQ, int HKV, int table_n, int page, int window,
                    int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP;
  size_t dyn = (size_t)(((chunk + 31) / 32) * 32) * 4 + (size_t)4 * S * SLOT;
  const size_t merge = (size_t)(64 + 4 * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd2_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  fd2_partials<S><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                         reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, HQ, HKV,
                                         table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---------------------------------------------------------------------------------------------------------------
// slide1: the whole sliding-window decode attention in ONE kernel (one query token, 8 heads per KV head):
// q/k RMS norm + RoPE, fp8 conversion, cache write of the new k/v, streamed attention over the window,
// cross-warp merge, per-head RMS norm and sigmoid gate.  grid (HKV, B), 256 threads.
// ---------------------------------------------------------------------------------------------------------------
#include <cuda_fp8.h>
#define SW 8                 // warps per CTA
#define SRMAX 2112           // max streamed tokens (window 2048 + slack)

__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

template <int S>
__global__ void __launch_bounds__(SW * 32)
slide1_kernel(const bf16* __restrict__ QKV, const long* __restrict__ POS, const long* __restrict__ LOC,
              uint8_t* __restrict__ KC, uint8_t* __restrict__ VC, const int* __restrict__ TABLE,
              const int* __restrict__ SEQLEN, const float* __restrict__ KS, const float* __restrict__ VS,
              bf16* __restrict__ OUT, int HQ, int HKV, int table_n, int page, int window, float theta, float sm_scale,
              int scaled, int qkv_stride) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int SLOT = 2 * ST2 * RP;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kh = blockIdx.x, b = blockIdx.y;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int n = prefix - first;            // streamed tokens (the new token is handled separately)
  const int nst = (n + ST2 - 1) / ST2;
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const bf16* qkv = QKV + (long)b * qkv_stride;
  const float pos = (float)POS[b];

  // shared layout: rows[SRMAX] | qs[8][64] | newslot (K row 0 / V row 0, zero filled) | per-warp rings | merge area
  int* rows = reinterpret_cast<int*>(smem);
  uint8_t* qs = smem + SRMAX * 4;
  uint8_t* newslot = qs + 512;                       // SLOT bytes
  uint8_t* rings = newslot + SLOT;                   // SW * S * SLOT bytes
  uint8_t* wbase = rings + warp * (S * SLOT);

  // ---- zero the new-token slot, fill the row table ----
  for (int i = tid; i < SLOT / 16; i += SW * 32) reinterpret_cast<uint4*>(newslot)[i] = make_uint4(0, 0, 0, 0);
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += SW * 32) {
    const int tok = first + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  // ---- q: 8 query heads of this KV head (threads 0..127), k_new (warp 4), v_new (warp 5) ----
  if (tid < 128) {
    const int j = tid >> 4, part = tid & 15;
    const bf16* src = qkv + (kh * 8 + j) * HD + 4 * part;
    float v[4];
    {
      const uint2 raw = *reinterpret_cast<const uint2*>(src);
      const bf162* p = reinterpret_cast<const bf162*>(&raw);
      const float2 a = __bfloat1622float2(p[0]), c = __bfloat1622float2(p[1]);
      v[0] = a.x; v[1] = a.y; v[2] = c.x; v[3] = c.y;
    }
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
    for (int d = 1; d <= 8; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float inv = rsqrtf(ss / 64.f + 1e-5f);
    float o4[4];
#pragma unroll
    for (int k = 0; k < 4; k++) o4[k] = v[k] * inv;
    if (theta != 0.f) {
#pragma unroll
      for (int pp = 0; pp < 2; pp++) {
        const int i = 2 * part + pp;
        const float freq = expf((-2.f * (float)i / 64.f) * logf(theta));
        float sn, cs;
        sincosf(pos * freq, &sn, &cs);
        const float x0 = o4[2 * pp], x1 = o4[2 * pp + 1];
        o4[2 * pp] = x0 * cs - x1 * sn;
        o4[2 * pp + 1] = x0 * sn + x1 * cs;
      }
    }
    uint32_t w4 = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) w4 |= (uint32_t)f2e4m3(o4[k]) << (8 * k);
    *reinterpret_cast<uint32_t*>(qs + j * 64 + 4 * part) = w4;
  } else if (warp == 4 || warp == 5) {
    const int is_v = warp == 5;
    const bf16* src = qkv + (HQ + (is_v ? HKV : 0) + kh) * HD + 2 * lane;
    const bf162 raw = *reinterpret_cast<const bf162*>(src);
    float2 v = __bfloat1622float2(raw);
    if (!is_v) {
      float ss = v.x * v.x + v.y * v.y;
#pragma unroll
      for (int d = 1; d <= 16; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
      const float inv = rsqrtf(ss / 64.f + 1e-5f);
      v.x *= inv; v.y *= inv;
      if (theta != 0.f) {
        const float freq = expf((-2.f * (float)lane / 64.f) * logf(theta));
        float sn, cs;
        sincosf(pos * freq, &sn, &cs);
        const float x0 = v.x, x1 = v.y;
        v.x = x0 * cs - x1 * sn;
        v.y = x0 * sn + x1 * cs;
      }
      v.x /= key_scale; v.y /= key_scale;
    } else {
      v.x /= value_scale; v.y /= value_scale;
    }
    const uint8_t b0 = f2e4m3(v.x), b1 = f2e4m3(v.y);
    const long loc = LOC[b];
    uint8_t* dst = (is_v ? VC : KC) + (loc * HKV + kh) * HD + 2 * lane;
    *reinterpret_cast<uint16_t*>(dst) = (uint16_t)b0 | ((uint16_t)b1 << 8);
    uint8_t* ns = newslot + (is_v ? ST2 * RP : 0) + 2 * lane;     // row 0 of the new-token slot
    *reinterpret_cast<uint16_t*>(ns) = (uint16_t)b0 | ((uint16_t)b1 << 8);
  }
  __syncthreads();

  uint32_t q0[2], q2[2];
#pragma unroll
  for (int ks = 0; ks < 2; ks++) {
    q0[ks] = *reinterpret_cast<const uint32_t*>(qs + g * 64 + 32 * ks + 4 * t);
    q2[ks] = *reinterpret_cast<const uint32_t*>(qs + g * 64 + 32 * ks + 16 + 4 * t);
  }
  float o[8][2];
  float m_run = -INFINITY, l_run = 0.f;
#pragma unroll
  for (int d = 0; d < 8; d++) o[d][0] = o[d][1] = 0.f;

  const int my_stages = (nst - warp + SW - 1) / SW > 0 ? (nst - warp + SW - 1) / SW : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + SW * s) * ST2;
      const int c = lane & 3;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int tk = (lane >> 2) + 8 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = (rr * HKV + kh) * HD + c * 16;
        cp16(slot + tk * RP + c * 16, KC + goff);
        cp16(slot + ST2 * RP + tk * RP + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);
  const int srow = lane & 7, smat = lane >> 3;

  auto process = [&](const uint8_t* slot, int tb, int end_tok) {
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP;
    float sacc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#pragma unroll
      for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      uint32_t k0, k1, k2, k3;
      ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP + smat * 16);
      mma_qk8(sacc[nt], q0[0], q2[0], k0, k1);
      mma_qk8(sacc[nt], q0[1], q2[1], k2, k3);
    }
    const bool full_vis = tb + ST2 <= end_tok;
    float sv[4][2];
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int r = 0; r < 2; r++) {
        float v = sacc[nt][r];
        if (!full_vis && tb + nt * 8 + 2 * t + r >= end_tok) v = -INFINITY;
        sv[nt][r] = v;
        mx = fmaxf(mx, v);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx *= sc2;
    const float newm = fmaxf(m_run, mx);
    const float safe = newm == -INFINITY ? 0.f : newm;
    const float corr = exp2f(m_run - safe);
    float rs = 0.f;
    uint32_t pa[4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
      rs += p0 + p1;
      pa[nt] = pack_h2(p0, p1);
    }
    rs += __shfl_xor_sync(0xffffffffu, rs, 1);
    rs += __shfl_xor_sync(0xffffffffu, rs, 2);
    l_run = l_run * corr + rs;
    m_run = newm;
    if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
      for (int d = 0; d < 8; d++) { o[d][0] *= corr; o[d][1] *= corr; }
    }
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP + pr * 32 + (smat >> 1) * 16);
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
        float acc[4];
        acc[0] = o[4 * pr + 0][0]; acc[1] = o[4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
        o[4 * pr + 0][0] = acc[0]; o[4 * pr + 0][1] = acc[1];
        acc[0] = o[4 * pr + 1][0]; acc[1] = o[4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
        o[4 * pr + 1][0] = acc[0]; o[4 * pr + 1][1] = acc[1];
        acc[0] = o[4 * pr + 2][0]; acc[1] = o[4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
        o[4 * pr + 2][0] = acc[0]; o[4 * pr + 2][1] = acc[1];
        acc[0] = o[4 * pr + 3][0]; acc[1] = o[4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
        o[4 * pr + 3][0] = acc[0]; o[4 * pr + 3][1] = acc[1];
      }
    }
  };

#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    process(wbase + (s % S) * SLOT, first + (warp + SW * s) * ST2, prefix);
    __syncwarp();
  }
  cp_wait<0>();
  if (warp == 0) process(newslot, prefix, prefix + 1);   // the new token (position prefix), fp8-rounded as stored
  __syncthreads();

  // ---- merge the warps ----
  float* sm_m = reinterpret_cast<float*>(smem);   // reuses the row table: [SW][8], [SW][8], [SW][8][64]
  float* sm_l = sm_m + SW * 8;
  float* sm_o = sm_l + SW * 8;
  if (t == 0) { sm_m[warp * 8 + g] = m_run; sm_l[warp * 8 + g] = l_run; }
#pragma unroll
  for (int mm = 0; mm < 4; mm++)
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[2 * mm][cc];
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[2 * mm + 1][cc];
    }
  __syncthreads();
  // warp h finalizes query head h of this KV head; lane owns dims 2 lane, 2 lane + 1
  {
    const int h = warp;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < SW; w++) mmx = fmaxf(mmx, sm_m[w * 8 + h]);
    float ls = 0.f, a0 = 0.f, a1 = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < SW; w++) {
        const float f = sm_m[w * 8 + h] == -INFINITY ? 0.f : exp2f(sm_m[w * 8 + h] - mmx);
        ls += sm_l[w * 8 + h] * f;
        a0 += sm_o[(w * 8 + h) * 64 + 2 * lane] * f;
        a1 += sm_o[(w * 8 + h) * 64 + 2 * lane + 1] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    // bf16 round, RMS norm over the 64 dims, sigmoid gate
    const float u0 = __bfloat162float(__float2bfloat16(a0 * inv * value_scale));
    const float u1 = __bfloat162float(__float2bfloat16(a1 * inv * value_scale));
    float ss = u0 * u0 + u1 * u1;
#pragma unroll
    for (int d = 1; d <= 16; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float rinv = rsqrtf(ss / 64.f + 1e-5f);
    const int qh = kh * 8 + h;
    const bf162 gate = *reinterpret_cast<const bf162*>(qkv + (HQ + 2 * HKV) * HD + qh * HD + 2 * lane);
    const float2 gf = __bfloat1622float2(gate);
    const float s0 = 1.f / (1.f + expf(-gf.x)), s1 = 1.f / (1.f + expf(-gf.y));
    *reinterpret_cast<bf162*>(OUT + ((long)b * HQ + qh) * HD + 2 * lane) = __floats2bfloat162_rn(u0 * rinv * s0, u1 * rinv * s1);
  }
}

template <int S>
static void launch_slide(const void* QKV, const void* POS, const void* LOC, void* KC, void* VC, const int* TABLE,
                         const int* SEQLEN, const float* KS, const float* VS, void* OUT, int B, int HQ, int HKV,
                         int table_n, int page, int window, float theta, float sm_scale, int scaled, int qkv_stride,
                         cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP;
  size_t dyn = (size_t)SRMAX * 4 + 512 + SLOT + (size_t)SW * S * SLOT;
  const size_t merge = (size_t)(2 * SW * 8 + SW * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(slide1_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(HKV, B);
  slide1_kernel<S><<<grid, SW * 32, dyn, st>>>(reinterpret_cast<const bf16*>(QKV), reinterpret_cast<const long*>(POS),
                                              reinterpret_cast<const long*>(LOC), reinterpret_cast<uint8_t*>(KC),
                                              reinterpret_cast<uint8_t*>(VC), TABLE, SEQLEN, KS, VS,
                                              reinterpret_cast<bf16*>(OUT), HQ, HKV, table_n, page, window, theta,
                                              sm_scale, scaled, qkv_stride);
}

extern "C" int slide_run(int stages, const void* QKV, const void* POS, const void* LOC, void* KC, void* VC,
                         const int* TABLE, const int* SEQLEN, const float* KS, const float* VS, void* OUT, int B, int HQ,
                         int HKV, int table_n, int page, int window, float theta, float sm_scale, int scaled,
                         int qkv_stride, cudaStream_t st) {
#define SL(S_) if (stages == S_) { launch_slide<S_>(QKV, POS, LOC, KC, VC, TABLE, SEQLEN, KS, VS, OUT, B, HQ, HKV, table_n, page, window, theta, sm_scale, scaled, qkv_stride, st); return 0; }
  SL(2) SL(3) SL(4)
#undef SL
  return 1;
}

__global__ void pf_kernel(const char* p, size_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const size_t chunk = 32768;
  for (size_t off = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * chunk; off < bytes; off += (size_t)gridDim.x * blockDim.x * chunk) {
    const size_t n = bytes - off < chunk ? bytes - off : chunk;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p + off), "r"((uint32_t)n) : "memory");
  }
#endif
}
extern "C" void pf_run(const void* p, size_t bytes, cudaStream_t st) { pf_kernel<<<4, 128, 0, st>>>(reinterpret_cast<const char*>(p), bytes); }

template <int MT, int S, bool QK8>
static void launch(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                   const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                   int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int BM = MT * 16;
  constexpr int PER_WARP = S * 2 * 16 * RP + 2 * 16 * KP;
  size_t dyn = 4 * PER_WARP;
  const size_t merge = (size_t)(8 * BM + 4 * BM * HD) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd_partials<MT, S, QK8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  fd_partials<MT, S, QK8><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                             reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ,
                                             HKV, table_n, page, window, splits, chunk, sm_scale, scaled);
}

extern "C" int fd_run(int mt, int stages, int qk8, const void* Q, const void* KC, const void* VC, const int* TABLE,
                      const int* SEQLEN, const float* KS, const float* VS, float* PART, float* LSE, int B, int L, int HQ,
                      int HKV, int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled,
                      cudaStream_t st) {
#define FD(MT_, S_) if (mt == MT_ && stages == S_ && qk8 == 0) { launch<MT_, S_, false>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
#define FD8(MT_, S_) if (mt == MT_ && stages == S_ && qk8 == 1) { launch<MT_, S_, true>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD(1, 3) FD(1, 4) FD(1, 6) FD(1, 8) FD(4, 3) FD(4, 4) FD(4, 6)
  FD8(1, 3) FD8(1, 4) FD8(1, 6) FD8(4, 3) FD8(4, 4) FD8(4, 6)
#undef FD8
#define FD2(S_) if (mt == 1 && stages == 200 + S_) { launch2<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD2(2) FD2(3) FD2(4)
#undef FD2
#define FD3V(S_) if (mt == 4 && HKV == 2 && L == 8 && stages == 400 + S_) { launch3v<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3V(2) FD3V(3) FD3V(4)
#undef FD3V
#define FD3(S_) if (mt == 1 && HKV == 2 && stages == 300 + S_) { launch3<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3(2) FD3(3) FD3(4)
#undef FD3
#define FDV(S_) if (mt == 4 && stages == 100 + S_) { launch_v<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FDV(3) FDV(4) FDV(6) FDV(8)
#undef FDV
#undef FD
  return 1;
}
"""
