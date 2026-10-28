"""Weight-major NVFP4 MoE kernels (CUDA source + weight repack), appended to the v5 base."""

CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

#define HD 4096
#define HD_ 4096
#define ID 4096
#define KC 64            // k64 chunks along K = 4096
#define KCG 16           // groups of 4 chunks (scale words)
#define RT0 512          // 16-row tiles of w13 (gate 256 + up 256)
#define RT1 256          // 16-row tiles of w2

constexpr uint32_t MASK = 0x81C081C0u;

// Weight word -> bf16x2 holding the E2M1 value times 2^-126 (denormal-linear encoding).
template <int J>
__device__ __forceinline__ uint32_t dec(uint32_t w) {
  if constexpr (J == 0) {
    return w & MASK;
  } else if constexpr (J < 3) {
    return __funnelshift_r(w, w, 3 * J) & MASK;
  } else {
    return ((w & 0x001B001Bu) * 0x1084u) & MASK;  // sources m0@4 m1@0 m2@1 sign@3 (+16 for the high half)
  }
}

__device__ __forceinline__ uint32_t scale2(uint32_t q, int c) {
  // byte c of q holds b'; scale = bf16 bits (b'<<4) + 0x7600 = S * 2^118
  uint32_t bits = ((q >> (8 * c)) & 0xFFu) << 4;
  return (bits + 0x7600u) * 0x10001u;
}

__device__ __forceinline__ uint32_t hmul(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

static int g_pdl = 0;
static int g_persist = 0;
extern "C" void wm_set_persist(int v) { g_persist = v; }
extern "C" void wm_set_pdl(int v) { g_pdl = v; }

template <typename... KArgs, typename... Args>
static void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, bool pdl,
                     Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

__global__ void route_kernel(const int* __restrict__ ids, int npair, const int* __restrict__ ls_ptr, int ls_val, int lmax,
                             int* __restrict__ cnt, int* __restrict__ lst, int* __restrict__ act,
                             int* __restrict__ nact, int* __restrict__ local_ids, int* __restrict__ woff, int grp, int nslots) {
  __shared__ int s_cnt[32];
  pdl_wait();
  const int ls = ls_ptr ? *ls_ptr : ls_val;
  pdl_trigger();
  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;
  for (int i = threadIdx.x; i < nslots; i += blockDim.x) reinterpret_cast<int4*>(act)[i] = make_int4(0, 0, 0, 0);  // work items {expert, offset, tokens, 0}
  __syncthreads();
  for (int p = threadIdx.x; p < npair; p += blockDim.x) {
    int e = (ids[p] - ls) & 255;  // experts are ordered mod 256 starting at this GPU's first expert
    bool ok = e < 32;
    local_ids[p] = ok ? e : -1;
    if (ok) {
      int pos = atomicAdd(&s_cnt[e], 1);
      lst[e * lmax + pos] = p;
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    int lane = threadIdx.x;
    int c = s_cnt[lane];
    cnt[lane] = c;
    // work items: one per group of `grp` tokens of an expert, so a hot expert spreads over several CTAs
    const int groups = (c + grp - 1) / grp;
    // longest-processing-time-first: experts with the most tokens (largest NCH) get the earliest work items
    int start = 0, total = 0;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      const int cj = __shfl_sync(0xffffffffu, c, j), gj = __shfl_sync(0xffffffffu, groups, j);
      if (cj > c || (cj == c && j < lane)) start += gj;
      total += gj;
    }
    for (int j = 0; j < groups; j++)
      reinterpret_cast<int4*>(act)[start + j] = make_int4(lane, j * grp, min(c - j * grp, grp), 1);
    if (lane == 31) *nact = total;
  }
}

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc));
}
__device__ __forceinline__ void cp16z(void* sdst, const void* gsrc, uint32_t nbytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc), "r"(nbytes));
}
template <int NTB_, int NCH_>
__host__ __device__ constexpr int ring2_stages() {
  return (14336 / (512 * (NTB_ + 2 * NCH_))) > 6 ? 6 : ((14336 / (512 * (NTB_ + 2 * NCH_))) < 3 ? 3 : (14336 / (512 * (NTB_ + 2 * NCH_))));
}
__device__ __forceinline__ void cp8(void* sdst, const void* gsrc) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(sdst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(d), "l"(gsrc));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int NTB, int NCH, int NW>
__device__ __forceinline__ void wm_epilogue(float (&acc)[NTB][NCH][4], float* red, const float gs, const int* lst,
                                            int ntok, int tbase, int mode, int nt, bf16* OUT, const int bx) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++)
        red[(((warp * NTB + n) * NCH + c) * 4 + r) * 32 + lane] = acc[n][c][r];
  __syncthreads();
  for (int item = threadIdx.x; item < NCH * 4 * 32; item += NW * 32) {
    const int li = item & 31, r = (item >> 5) & 3, c = item >> 7;
    const int row = (li >> 2) + 8 * (r >> 1);
    const int col = c * 8 + 2 * (li & 3) + (r & 1);
    if (col >= ntok) continue;
    const int p = __ldcg(lst + col);
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      if (mode == 0 && (n & 1)) continue;
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) s += red[(((w * NTB + n) * NCH + c) * 4 + r) * 32 + li];
      if (mode == 0) {
        float s2 = 0.f;
#pragma unroll
        for (int w = 0; w < NW; w++) s2 += red[(((w * NTB + n + 1) * NCH + c) * 4 + r) * 32 + li];
        const float gate = bfr(bfr(s * 256.0f) * gs), up = bfr(bfr(s2 * 256.0f) * gs);
        const float sil = gate / (1.0f + __expf(-gate));
        OUT[(size_t)p * ID + (bx * (nt / 2) + ((tbase + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);
      } else {
        OUT[(size_t)p * HD + (bx * nt + tbase + n) * 16 + row] = __float2bfloat16(bfr(s * 256.0f) * gs);
      }
    }
  }
  __syncthreads();
}

// RING == false: every lane issues its weight loads up front, group by group (best for small problems).
// RING == true : weights stream through a per-warp cp.async ring in shared memory (lane-private slots).
template <int MODE, int NT, int NTB, int NCH, int NW, int RING, int S, int DBG>
__device__ __forceinline__ void wm_body(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq,
                                        const float* __restrict__ GS, const bf16* __restrict__ X,
                                        bf16* __restrict__ OUT, const int* __restrict__ lst, int e, int ntok,
                                        int tbase, uint4* smem, const int bx) {
  constexpr int RT = MODE == 0 ? RT0 : RT1;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int CPW = 64 / NW;  // chunks of k64 per warp
  constexpr int GPW = CPW / 4;  // scale groups per warp
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  float* red = reinterpret_cast<float*>(smem);

  int tile[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) {
    const int nn = tbase + n;
    if (MODE == 0) tile[n] = (nn & 1) ? RT0 / 2 + bx * (NT / 2) + (nn >> 1) : bx * (NT / 2) + (nn >> 1);
    else tile[n] = bx * NT + nn;
  }

  const bf16* xp[NCH];
#pragma unroll
  for (int c = 0; c < NCH; c++) {
    int col = c * 8 + g;
    xp[c] = nullptr;
    if (col < ntok) {
      int p = __ldcg(lst + col);
      xp[c] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + 16 * t + warp * CPW * 64;
    }
  }

  float acc[NTB][NCH][4];
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;

  uint32_t sink = 0;
  const uint4* wsrc[NTB];
  const uint2* ssrc[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) {
    wsrc[n] = Wq + ((size_t)e * RT + tile[n]) * KC * 32 + warp * CPW * 32 + lane;
    ssrc[n] = Sq + ((size_t)e * RT + tile[n]) * KCG * 32 + warp * GPW * 32 + lane;
  }

  if (RING == 0) {
#pragma unroll
    for (int q = 0; q < GPW; q++) {
      uint2 sc[NTB];
      uint4 wv[NTB][4];
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        sc[n] = __ldg(ssrc[n] + q * 32);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          if constexpr (DBG & 8) wv[n][c] = make_uint4(lane * 7 + c, lane * 13 + q, lane + n, lane * 3);
          else wv[n][c] = __ldg(wsrc[n] + (q * 4 + c) * 32);
        }
      }
#pragma unroll
      for (int c = 0; c < 4; c++) {
        uint4 xa[NCH], xb[NCH];
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          if (xp[h] && !(DBG & 4)) {
            const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (q * 4 + c) * 64);
            xa[h] = __ldcg(xv);
            xb[h] = __ldcg(xv + 1);
          } else {
            xa[h] = make_uint4(0, 0, 0, 0);
            xb[h] = make_uint4(0, 0, 0, 0);
          }
        }
#pragma unroll
        for (int n = 0; n < NTB; n++) {
          const uint32_t s0 = scale2(sc[n].x, c);
          const uint32_t s1 = scale2(sc[n].y, c);
          const uint32_t ww[4] = {wv[n][c].x, wv[n][c].y, wv[n][c].z, wv[n][c].w};
#pragma unroll
          for (int ks = 0; ks < 4; ks++) {
            const uint32_t w = ww[ks];
            uint32_t a0, a1, a2, a3;
            if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }
            else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }
#pragma unroll
            for (int h = 0; h < NCH; h++) {
              const uint4 xs = ks < 2 ? xa[h] : xb[h];
              if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
              else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
            }
          }
        }
      }
    }
  } else if (RING == 2) {
    constexpr int XS = NCH * 2;
    constexpr int STG = NTB + XS;
    uint4* ring = smem + warp * (S * STG * 32);
    auto issue = [&](int c) {
      if (c < CPW && !(DBG & 8)) {
#pragma unroll
        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * STG + n) * 32 + lane], wsrc[n] + c * 32);
      }
      if (c < CPW) {
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const bool ok = xp[h] != nullptr && !(DBG & 4);
          const bf16* src = ok ? xp[h] + c * 64 : X;
          cp16z(&ring[((c % S) * STG + NTB + 2 * h) * 32 + lane], src, ok ? 16u : 0u);
          cp16z(&ring[((c % S) * STG + NTB + 2 * h + 1) * 32 + lane], src + 8, ok ? 16u : 0u);
        }
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);
    uint2 scur[NTB], snxt[NTB];
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      scur[n] = __ldg(ssrc[n]);
      snxt[n] = GPW > 1 ? __ldg(ssrc[n] + 32) : scur[n];
    }
#pragma unroll 1
    for (int i = 0; i < CPW; i++) {
      cp_wait<S - 2>();
      issue(i + S - 1);
      const int cq = i & 3;
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        const uint4 wv = ring[((i % S) * STG + n) * 32 + lane];
        const uint32_t s0 = scale2(scur[n].x, cq);
        const uint32_t s1 = scale2(scur[n].y, cq);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint32_t w = ww[ks];
          uint32_t a0, a1, a2, a3;
          if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }
          else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = ring[((i % S) * STG + NTB + 2 * h + (ks >> 1)) * 32 + lane];
            if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
            else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
          }
        }
      }
      if (cq == 3) {
#pragma unroll
        for (int n = 0; n < NTB; n++) {
          scur[n] = snxt[n];
          if ((i >> 2) + 2 < GPW) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);
        }
      }
    }
    cp_wait<0>();
  } else {
    uint4* ring = smem + warp * (S * NTB * 32);
    constexpr int GRP = RING == 8 ? 2 : RING == 9 ? 4 : 1;   // chunks fetched back to back (larger contiguous DRAM bursts per weight stream)
    auto issue = [&](int c) {
      if (c < CPW && !(DBG & 8)) {
#pragma unroll
        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * NTB + n) * 32 + lane], wsrc[n] + c * 32);
      }
      cp_commit();
    };
    auto issue_group = [&](int gi) {
#pragma unroll
      for (int g = 0; g < GRP; g++) {
        const int c = gi * GRP + g;
        if (c < CPW && !(DBG & 8)) {
#pragma unroll
          for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * NTB + n) * 32 + lane], wsrc[n] + c * 32);
        }
      }
      cp_commit();
    };
    if constexpr (GRP > 1) {
      issue_group(0);
      issue_group(1);
    } else {
#pragma unroll
      for (int c = 0; c < S - 1; c++) issue(c);
    }

    uint2 scur[NTB], snxt[NTB];
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      scur[n] = __ldg(ssrc[n]);
      snxt[n] = GPW > 1 ? __ldg(ssrc[n] + 32) : scur[n];
    }
    uint4 xcur[NCH][2], xnxt[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      if (xp[h] && !(DBG & 4)) {
        const uint4* xv = reinterpret_cast<const uint4*>(xp[h]);
        xcur[h][0] = __ldcg(xv);
        xcur[h][1] = __ldcg(xv + 1);
      } else {
        xcur[h][0] = make_uint4(0, 0, 0, 0);
        xcur[h][1] = make_uint4(0, 0, 0, 0);
      }
      xnxt[h][0] = xcur[h][0];
      xnxt[h][1] = xcur[h][1];
    }

#pragma unroll 1
    for (int i = 0; i < CPW; i++) {
      if constexpr (GRP > 1) {
        if ((i % GRP) == 0) { cp_wait<1>(); issue_group(i / GRP + 2); }
      } else {
        cp_wait<S - 2>();
        issue(i + S - 1);
      }
#pragma unroll
      for (int h = 0; h < NCH; h++) {
        if (xp[h] && i + 1 < CPW && !(DBG & 4)) {
          const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (i + 1) * 64);
          xnxt[h][0] = __ldcg(xv);
          xnxt[h][1] = __ldcg(xv + 1);
        }
      }
      const int cq = i & 3;
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        const uint4 wv = ring[((i % S) * NTB + n) * 32 + lane];
        const uint32_t s0 = scale2(scur[n].x, cq);
        const uint32_t s1 = scale2(scur[n].y, cq);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint32_t w = ww[ks];
          uint32_t a0, a1, a2, a3;
          if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }
          else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = xcur[h][ks >> 1];
            if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
            else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
          }
        }
      }
      if (cq == 3) {
#pragma unroll
        for (int n = 0; n < NTB; n++) {
          scur[n] = snxt[n];
          if ((i >> 2) + 2 < GPW) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);
        }
      }
#pragma unroll
      for (int h = 0; h < NCH; h++) {
        xcur[h][0] = xnxt[h][0];
        xcur[h][1] = xnxt[h][1];
      }
    }
    cp_wait<0>();
  }

  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);
  __syncthreads();  // the ring is dead; the same smem holds the cross-warp reduction
  wm_epilogue<NTB, NCH, NW>(acc, red, GS[e], lst, ntok, tbase, MODE, NT, OUT, bx);
}

#ifndef WM_MINB
#define WM_MINB 4
#endif
template <int MODE, int NT, int NMAX, int NW, int RING, int DBG>
__global__ void __launch_bounds__(NW * 32, (RING == 3 ? 2 : WM_MINB))
wm_gemm(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,
        const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ cnt,
        const int* __restrict__ lst, const int* __restrict__ act, const int* __restrict__ nact_p,
        const int* __restrict__ woff, int lmax, int persist) {
  extern __shared__ uint4 smem[];
  constexpr int TILES = (MODE == 0 ? 512 : 256) / NT;
  pdl_wait();  // everything below (items, token lists, activations) comes from earlier kernels
  pdl_trigger();
  const int nitems = persist ? __ldcg(nact_p) : 1;
  for (int u = blockIdx.x;; u += gridDim.x) {
  int slot, bx;
  if (persist) {
    if (u >= nitems * TILES) break;
    slot = u / TILES; bx = u - slot * TILES;
  } else { slot = blockIdx.y; bx = blockIdx.x; }
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + slot);  // {expert, offset, tokens, valid}
  if (item.z != 0) {
  const int e = item.x;
  const int off = item.y;
  {
    const int ntok = item.z;
    const int* l = lst + e * lmax + off;
    if (ntok <= 8) {
      wm_body<MODE, NT, NT, 1, NW, RING, (RING == 2 ? ring2_stages<NT, 1>() : RING == 8 ? 6 : RING == 9 ? 12 : (NT == 8 ? 3 : NT == 4 ? 4 : 6)), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, 0, smem, bx);
    } else if (NMAX >= 2 && ntok <= 16) {
#pragma unroll 1
      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))
        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 2 ? 2 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 2 ? 2 : 1)>() : RING == 3 ? 4 : RING == 8 ? 6 : RING == 9 ? 12 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);
    } else if (NMAX >= 4) {
#pragma unroll 1
      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))
        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 4 ? 4 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 4 ? 4 : 1)>() : RING == 3 ? 4 : RING == 8 ? 6 : RING == 9 ? 12 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);
    }
  }
  }
  if (!persist) break;
  }
}



// ---- row-split variant: the four warps of a CTA own different weight rows but the same activations, which are
// staged once per CTA through shared memory.  No cross-warp reduction; x traffic from L2 drops 4x. ----
template <int MODE, int NTB, int NCH, int S, int DBG>
__device__ __forceinline__ void wm_body_rs(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq,
                                           const float* __restrict__ GS, const bf16* __restrict__ X,
                                           bf16* __restrict__ OUT, const int* __restrict__ lst, int e, int ntok,
                                           uint4* smem, const int bx) {
  constexpr int RT = MODE == 0 ? RT0 : RT1;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int NTC = 4 * NTB;  // tiles per CTA
  constexpr int XSLOTS = NCH * 2 * 32;
  constexpr int RPT = (XSLOTS + 127) / 128;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  constexpr int SOFF = 3 * 4 * NTB * 16;     // scale ring: [3][4][NTB][32] uint2, in uint4 units
  uint2* sring = reinterpret_cast<uint2*>(smem);
  uint4* wring = smem + SOFF;                // [S][4][NTB][32]
  uint4* xring = wring + S * 4 * NTB * 32;   // [S][XSLOTS]

  const uint4* wsrc[NTB];
  const uint2* ssrc[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) {
    const int nn = warp * NTB + n;
    const int tile = MODE == 0 ? ((nn & 1) ? RT0 / 2 + bx * (NTC / 2) + (nn >> 1) : bx * (NTC / 2) + (nn >> 1)) : bx * NTC + nn;
    wsrc[n] = Wq + ((size_t)e * RT + tile) * KC * 32 + lane;
    ssrc[n] = Sq + ((size_t)e * RT + tile) * KCG * 32 + lane;
  }
  const bf16* xb[RPT];
  bool xok[RPT];
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    const int sl = tid + 128 * r;
    xok[r] = false;
    xb[r] = X;
    if (sl < XSLOTS) {
      const int h = sl >> 6, j = (sl >> 5) & 1, L = sl & 31;
      const int col = h * 8 + (L >> 2);
      if (col < ntok) {
        const int p = __ldcg(lst + col);
        xb[r] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + 16 * (L & 3) + 8 * j;
        xok[r] = true;
      }
    }
  }

  uint32_t sink = 0;
  float acc[NTB][NCH][4];
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;

  auto issue_sc = [&](int g) {
    if (g < KCG && !(DBG & 32)) {
#pragma unroll
      for (int n = 0; n < NTB; n++) cp8(&sring[(((g % 3) * 4 + warp) * NTB + n) * 32 + lane], ssrc[n] + g * 32);
    }
  };
  auto issue = [&](int c) {
    if ((c & 3) == 0) issue_sc((c >> 2) + 1);
    if (c < KC) {
#pragma unroll
      for (int n = 0; n < NTB; n++) { if constexpr (!(DBG & 8)) cp16(&wring[(((c % S) * 4 + warp) * NTB + n) * 32 + lane], wsrc[n] + c * 32); }
#pragma unroll
      for (int r = 0; r < RPT; r++)
        if (tid + 128 * r < XSLOTS && !(DBG & 4)) cp16z(&xring[(c % S) * XSLOTS + tid + 128 * r], xb[r] + c * 64, xok[r] ? 16u : 0u);
    }
    cp_commit();
  };
  issue_sc(0);
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
#pragma unroll 1
  for (int i = 0; i < KC; i++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(i + S - 1);
    const int cq = i & 3;
    uint4 xr[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      xr[h][0] = xring[(i % S) * XSLOTS + (h * 2) * 32 + lane];
      xr[h][1] = xring[(i % S) * XSLOTS + (h * 2 + 1) * 32 + lane];
    }
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      const uint4 wv = wring[(((i % S) * 4 + warp) * NTB + n) * 32 + lane];
      const uint2 sc = (DBG & 32) ? make_uint2(0x38383838u, 0x38383838u) : sring[((((i >> 2) % 3) * 4 + warp) * NTB + n) * 32 + lane];
      const uint32_t s0 = scale2(sc.x, cq);
      const uint32_t s1 = scale2(sc.y, cq);
      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        const uint32_t w = ww[ks];
        uint32_t a0, a1, a2, a3;
        if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }
        else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const uint4 xs = xr[h][ks >> 1];
          if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
          else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
        }
      }
    }
  }
  cp_wait<0>();
  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);

  const float gs = GS[e];
#pragma unroll
  for (int c = 0; c < NCH; c++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int col = c * 8 + 2 * t + (r & 1);
      if (col >= ntok || (DBG & 16)) continue;
      const int p = (DBG & 64) ? col * 7 : __ldcg(lst + col);
      const int row = g + 8 * (r >> 1);
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        if (MODE == 0) {
          if (n & 1) continue;
          const float gate = bfr(bfr(acc[n][c][r] * 256.0f) * gs), up = bfr(bfr(acc[n + 1][c][r] * 256.0f) * gs);
          const float sil = gate / (1.0f + __expf(-gate));
          OUT[(size_t)p * ID + (bx * (NTC / 2) + ((warp * NTB + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);
        } else {
          OUT[(size_t)p * HD + (bx * NTC + warp * NTB + n) * 16 + row] = __float2bfloat16(bfr(acc[n][c][r] * 256.0f) * gs);
        }
      }
    }
}

template <int MODE, int NTB, int NMAX, int S, int DBG>
__global__ void __launch_bounds__(128, (S <= 3 ? 4 : 3))
wm_gemm_rs(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,
           const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst,
           const int* __restrict__ act, int lmax) {
  extern __shared__ uint4 smem[];
  pdl_wait();
  pdl_trigger();
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);
  if (item.z == 0) return;
  const int e = item.x, ntok = item.z;
  const int* l = lst + e * lmax + item.y;
  if (ntok <= 8) wm_body_rs<MODE, NTB, 1, S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
  else if (NMAX >= 2 && ntok <= 16) wm_body_rs<MODE, NTB, (NMAX >= 2 ? 2 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
  else if (NMAX >= 4 && (NMAX < 8 || ntok <= 32)) wm_body_rs<MODE, NTB, (NMAX >= 4 ? 4 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
  else if (NMAX >= 8) wm_body_rs<MODE, NTB, (NMAX >= 8 ? 8 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
}

template <int MODE, int NTB, int NMAX, int S, int DBG>
static void launch_rs(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst,
                      const int* act, int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid((MODE == 0 ? 512 : 256) / (4 * NTB), (unsigned)maxslots);
  const int dyn = S * (4 * NTB * 512 + NMAX * 2 * 32 * 16) + 3 * 4 * NTB * 256;
  cudaFuncSetAttribute(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const uint2*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);
}


// ---- wgmma (Hopper) building blocks ----
__device__ __forceinline__ uint64_t wg_desc(const void* p, uint32_t lbo, uint32_t sbo) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  return (uint64_t)((a >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit_wait() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void proxy_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
template <int N> struct WgN;

template <> struct WgN<32> {
  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
  }
};
template <> struct WgN<64> {
  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
  }
};
template <> struct WgN<128> {
  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
  }
};
// position of natural k inside its 64-chunk after the 'fragment order' permutation the wgmma path reads its activations in
__device__ __forceinline__ int wg_perm_k(int k) {
  const int c = k >> 6, kl = k & 63;
  const int t = kl >> 4, ks = (kl >> 2) & 3, p = (kl >> 1) & 1, h = kl & 1;
  return c * 64 + (ks * 2 + p) * 8 + t * 2 + h;
}
__global__ void wm_permx_kernel(const bf16* __restrict__ X, bf16* __restrict__ Xp, int ld) {
  pdl_wait();
  pdl_trigger();
  const size_t base = (size_t)blockIdx.x * ld;
  for (int k = threadIdx.x; k < ld; k += blockDim.x) Xp[base + wg_perm_k(k)] = X[base + k];
}
extern "C" void wm_permx_c(const void* X, void* Xp, int M, cudaStream_t st) {
  launch_k(wm_permx_kernel, dim3(M), dim3(256), 0, st, g_pdl != 0, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(Xp), 4096);
}

// ---- wgmma row-split variant: a warpgroup owns 64 weight rows (one 16-row tile per warp, decoded into registers) and up to
// 128 tokens; B (activations) lives in shared memory in core-matrix layout ----
template <int MODE, int NCH, int S>
__device__ __forceinline__ void wm_body_wg(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,
                                           const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst, int e,
                                           int ntok, uint8_t* smem_raw, const int bx) {
  constexpr int RT = MODE == 0 ? RT0 : RT1;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int N = NCH * 8;
  constexpr int NG = MODE == 0 ? 2 : 1;                  // weight tiles per warp: gate + up (MODE 0)
  constexpr int XST = N * 128;                           // bytes of one x stage (N tokens x 64 k bf16)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  uint8_t* xring = smem_raw;                             // [S][N tokens in core layout]
  uint4* wring = reinterpret_cast<uint4*>(smem_raw + S * XST);   // [S][4 warps][NG][32 lanes]: one uint4 (4 ks words) per lane
  const uint4* wsrc[NG];
  const uint2* ssrc[NG];
#pragma unroll
  for (int q = 0; q < NG; q++) {
    // MODE 0: gate tile bx*4+warp, up tile RT0/2 + same; MODE 1: tile bx*4+warp
    const int tile = MODE == 0 ? (q == 0 ? bx * 4 + warp : RT0 / 2 + bx * 4 + warp) : bx * 4 + warp;
    wsrc[q] = Wq + ((size_t)e * RT + tile) * KC * 32 + lane;
    ssrc[q] = Sq + ((size_t)e * RT + tile) * KCG * 32 + lane;
  }
  // x staging: granule id = tid + 128 * u : token = id >> 3, k chunk = id & 7
  const bf16* xb[NCH / 2];
  bool xok[NCH / 2];
#pragma unroll
  for (int u = 0; u < NCH / 2; u++) {
    const int id = tid + 128 * u;
    const int col = id >> 3;
    xok[u] = col < ntok;
    xb[u] = X;
    if (xok[u]) {
      const int p = __ldcg(lst + col);
      xb[u] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + (id & 7) * 8;
    }
  }
  auto issue = [&](int c) {
    if (c < KC) {
#pragma unroll
      for (int q = 0; q < NG; q++) cp16(&wring[(((c % S) * 4 + warp) * NG + q) * 32 + lane], wsrc[q] + c * 32);
#pragma unroll
      for (int u = 0; u < NCH / 2; u++) {
        const int id = tid + 128 * u;
        cp16z(xring + (c % S) * XST + (id >> 6) * 1024 + (id & 7) * 128 + ((id >> 3) & 7) * 16, xb[u] + c * 64, xok[u] ? 16u : 0u);
      }
    }
    cp_commit();
  };
  float acc[NG][N / 2];
#pragma unroll
  for (int q = 0; q < NG; q++)
#pragma unroll
    for (int i = 0; i < N / 2; i++) acc[q][i] = 0.f;
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
  uint2 scur[NG], snxt[NG];
#pragma unroll
  for (int q = 0; q < NG; q++) {
    scur[q] = __ldg(ssrc[q]);
    snxt[q] = __ldg(ssrc[q] + 32);
  }
#pragma unroll 1
  for (int i = 0; i < KC; i++) {
    cp_wait<S - 2>();
    proxy_fence();
    __syncthreads();
    issue(i + S - 1);
    const int cq = i & 3;
    uint32_t a[NG][4][4];
#pragma unroll
    for (int q = 0; q < NG; q++) {
      const uint32_t s0 = scale2(scur[q].x, cq);
      const uint32_t s1 = scale2(scur[q].y, cq);
      const uint4 wv = wring[(((i % S) * 4 + warp) * NG + q) * 32 + lane];
      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        a[q][ks][0] = hmul(dec<0>(ww[ks]), s0);
        a[q][ks][1] = hmul(dec<1>(ww[ks]), s1);
        a[q][ks][2] = hmul(dec<2>(ww[ks]), s0);
        a[q][ks][3] = hmul(dec<3>(ww[ks]), s1);
      }
    }
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
#pragma unroll
      for (int q = 0; q < NG; q++) WgN<N>::mma(acc[q], a[q][ks][0], a[q][ks][1], a[q][ks][2], a[q][ks][3], wg_desc(xring + (i % S) * XST + ks * 256, 128, 1024));
    wg_commit_wait();
    if (cq == 3) {
#pragma unroll
      for (int q = 0; q < NG; q++) {
        scur[q] = snxt[q];
        if ((i >> 2) + 2 < KCG) snxt[q] = __ldg(ssrc[q] + ((i >> 2) + 2) * 32);
      }
    }
  }
  cp_wait<0>();
  const float gs = GS[e];
#pragma unroll
  for (int jb = 0; jb < N / 8; jb++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int col = jb * 8 + 2 * t + (r & 1);
      if (col >= ntok) continue;
      const int p = __ldcg(lst + col);
      const int row = warp * 16 + g + 8 * (r >> 1);
      if (MODE == 0) {
        const float gate = bfr(bfr(acc[0][jb * 4 + r] * 256.0f) * gs), up = bfr(bfr(acc[1][jb * 4 + r] * 256.0f) * gs);
        const float sil = gate / (1.0f + __expf(-gate));
        OUT[(size_t)p * ID + wg_perm_k(bx * 64 + row)] = __float2bfloat16(sil * up);
      } else {
        OUT[(size_t)p * HD + bx * 64 + row] = __float2bfloat16(bfr(acc[0][jb * 4 + r] * 256.0f) * gs);
      }
    }
}

template <int MODE, int NMAX, int S>
__global__ void __launch_bounds__(128, 2)
wm_gemm_wg(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS, const bf16* __restrict__ X,
           bf16* __restrict__ OUT, const int* __restrict__ lst, const int* __restrict__ act, int lmax) {
  extern __shared__ __align__(16) uint8_t smem_wg[];
  pdl_wait();
  pdl_trigger();
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);
  if (item.z == 0) return;
  const int e = item.x, ntok = item.z;
  const int* l = lst + e * lmax + item.y;
  if (ntok <= 32) wm_body_wg<MODE, 4, S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);
  else if (NMAX >= 8 && ntok <= 64) wm_body_wg<MODE, (NMAX >= 8 ? 8 : 4), S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);
  else if (NMAX >= 16) wm_body_wg<MODE, (NMAX >= 16 ? 16 : 4), S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);
}

template <int MODE, int NMAX, int S>
static void launch_wg(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst, const int* act,
                      int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid(64, (unsigned)maxslots);
  constexpr int NG = MODE == 0 ? 2 : 1;
  const int dyn = S * (NMAX * 8 * 128 + 4 * NG * 512);
  cudaFuncSetAttribute(wm_gemm_wg<MODE, NMAX, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(wm_gemm_wg<MODE, NMAX, S>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const uint2*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);
}

// Post-expert RMSNorm + weighted sum: out[m] = sum_k w[m,k] * rmsnorm_k(y[m,k] * nw), skipping slots routed elsewhere.
// One CTA per token; warp k owns slot k for the sum of squares, then all threads accumulate their columns.
__global__ void __launch_bounds__(256)
wm_finish_kernel(const bf16* __restrict__ Y, const float* __restrict__ TW, const int* __restrict__ lids,
                 const bf16* __restrict__ NW_, bf16* __restrict__ OUT, float eps) {
  __shared__ float s_scale[8];
  pdl_wait();
  pdl_trigger();
  const int m = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool live = lids[m * 8 + warp] >= 0;
  float ss = 0.f;
  if (live) {
    const uint4* yv = reinterpret_cast<const uint4*>(Y + ((size_t)m * 8 + warp) * HD_ * 1);
    const uint4* wv = reinterpret_cast<const uint4*>(NW_);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const uint4 a = __ldcg(yv + i * 32 + lane), w = __ldg(wv + i * 32 + lane);
      const bf162* ap = reinterpret_cast<const bf162*>(&a);
      const bf162* wp = reinterpret_cast<const bf162*>(&w);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float2 af = __bfloat1622float2(ap[j]), wf = __bfloat1622float2(wp[j]);
        const float v0 = af.x * wf.x, v1 = af.y * wf.y;
        ss += v0 * v0 + v1 * v1;
      }
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
  }
  if (lane == 0) s_scale[warp] = live ? TW[m * 8 + warp] * rsqrtf(ss / HD_ + eps) : 0.f;
  __syncthreads();
  // each thread: 16 columns = two uint4 of bf16
  float acc[16];
#pragma unroll
  for (int j = 0; j < 16; j++) acc[j] = 0.f;
  const uint4* wv = reinterpret_cast<const uint4*>(NW_) + tid * 2;
  const uint4 w0 = __ldg(wv), w1 = __ldg(wv + 1);
  float wf[16];
  {
    const bf162* p0 = reinterpret_cast<const bf162*>(&w0);
    const bf162* p1 = reinterpret_cast<const bf162*>(&w1);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float2 a = __bfloat1622float2(p0[j]), b = __bfloat1622float2(p1[j]);
      wf[2 * j] = a.x; wf[2 * j + 1] = a.y; wf[8 + 2 * j] = b.x; wf[8 + 2 * j + 1] = b.y;
    }
  }
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const float sc = s_scale[k];
    if (sc != 0.f) {
      const uint4* yv = reinterpret_cast<const uint4*>(Y + ((size_t)m * 8 + k) * HD_) + tid * 2;
      const uint4 a0 = __ldcg(yv), a1 = __ldcg(yv + 1);
      const bf162* p0 = reinterpret_cast<const bf162*>(&a0);
      const bf162* p1 = reinterpret_cast<const bf162*>(&a1);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float2 a = __bfloat1622float2(p0[j]), b = __bfloat1622float2(p1[j]);
        acc[2 * j] += sc * (a.x * wf[2 * j]);
        acc[2 * j + 1] += sc * (a.y * wf[2 * j + 1]);
        acc[8 + 2 * j] += sc * (b.x * wf[8 + 2 * j]);
        acc[8 + 2 * j + 1] += sc * (b.y * wf[8 + 2 * j + 1]);
      }
    }
  }
  uint4 o0, o1;
  bf162* q0 = reinterpret_cast<bf162*>(&o0);
  bf162* q1 = reinterpret_cast<bf162*>(&o1);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    q0[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
    q1[j] = __floats2bfloat162_rn(acc[8 + 2 * j], acc[8 + 2 * j + 1]);
  }
  uint4* ov = reinterpret_cast<uint4*>(OUT + (size_t)m * HD_) + tid * 2;
  ov[0] = o0;
  ov[1] = o1;
}
extern "C" void wm_finish_c(const void* Y, const float* TW, const int* lids, const void* NWT, void* OUT, int M, float eps,
                            cudaStream_t st) {
  launch_k(wm_finish_kernel, dim3(M), dim3(256), 0, st, g_pdl != 0, reinterpret_cast<const bf16*>(Y), TW, lids,
           reinterpret_cast<const bf16*>(NWT), reinterpret_cast<bf16*>(OUT), eps);
}

extern "C" void wm_route_c(const int* ids, int npair, const int* ls_ptr, int ls_val, int lmax, int* cnt, int* lst,
                           int* act, int* nact, int* local_ids, int* woff, int grp, int nslots, cudaStream_t st) {
  const int pdl = g_pdl;
  launch_k(route_kernel, dim3(1), dim3(1024), 0, st, pdl != 0, ids, npair, ls_ptr, ls_val, lmax, cnt, lst, act, nact, local_ids, woff, grp, nslots);
}

template <int MODE, int NT, int NMAX, int NW, int RING, int DBG>
static void launch(const void* W, const void* S, const float* GS, const void* X, void* OUT, const int* cnt,
                   const int* lst, const int* act, const int* nact, const int* woff, int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid((MODE == 0 ? 512 : 256) / NT, (unsigned)maxslots);
  if (g_persist) { const unsigned cap = 132u * (unsigned)g_persist; const unsigned tot = grid.x * grid.y; grid = dim3(tot < cap ? tot : cap, 1); }
  int dyn = NW * NT * 6 * 512 > 49152 ? 49152 : NW * NT * 6 * 512;
  if (NT == 8) dyn = NW * 3 * 8 * 512 + 1024;
  if (RING == 8) dyn = NW * NT * 6 * 512;
  if (RING == 9) dyn = NW * NT * 12 * 512;
  if (RING == 3) { const int d3 = NW * 4 * 4 * 512; dyn = d3 > dyn ? d3 : dyn; if (NW * NT * 4 * 512 > dyn) dyn = NW * NT * 4 * 512; }
  if (RING == 2) {
    int per = ring2_stages<NT, 1>() * (NT + 2);
    if (NMAX >= 2) { const int p2 = ring2_stages<2, 2>() * (2 + 4); per = per > p2 ? per : p2; }
    if (NMAX >= 4) { const int p4 = ring2_stages<2, 4>() * (2 + 8); per = per > p4 ? per : p4; }
    dyn = NW * per * 512;
  }
  cudaFuncSetAttribute(wm_gemm<MODE, NT, NMAX, NW, RING, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(wm_gemm<MODE, NT, NMAX, NW, RING, DBG>, grid, dim3(NW * 32), (size_t)dyn, st, pdl != 0,
           reinterpret_cast<const uint4*>(W), reinterpret_cast<const uint2*>(S), GS, reinterpret_cast<const bf16*>(X),
           reinterpret_cast<bf16*>(OUT), cnt, lst, act, nact, woff, lmax, g_persist);
}


// ---- L2 prefetch of the down-projection weights of the active experts (issued while the gate/up GEMM streams) ----
__device__ __forceinline__ uint64_t pol_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void pf_l2(const void* ptr, uint32_t bytes, uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(ptr), "r"(bytes), "l"(pol) : "memory");
#endif
}
// W2 codes of one expert: RT1 * KC * 512 B = 8 MiB; scales RT1 * KCG * 256 B = 1 MiB. Slab = 8 KiB.
__global__ void __launch_bounds__(128)
wm_pf_kernel(const uint8_t* __restrict__ W2, const uint8_t* __restrict__ S2, const int* __restrict__ act,
             const int* __restrict__ nact_p, int max_items, int slabs_code, int slabs_scale, size_t code_stride, size_t scale_stride) {
  pdl_wait();
  pdl_trigger();
  const int nitems = min(__ldcg(nact_p), max_items);
  const uint64_t pol = pol_evict_last();
  const int per = slabs_code + slabs_scale;
  const long total = (long)nitems * per;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int it = (int)(i / per), r = (int)(i % per);
    const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + it);
    if (item.z == 0 || item.y != 0) continue;   // one prefetch per expert (its first work item)
    const size_t e = item.x;
    if (r < slabs_code) pf_l2(W2 + e * code_stride + (size_t)r * 8192, 8192, pol);
    else pf_l2(S2 + e * scale_stride + (size_t)(r - slabs_code) * 8192, 8192, pol);
  }
}
extern "C" void wm_pf_c(const void* W2, const void* S2, const int* act, const int* nact, int max_items, int pdl, cudaStream_t st) {
  constexpr size_t CODE = (size_t)RT1 * KC * 512, SCALE = (size_t)RT1 * KCG * 256;
  launch_k(wm_pf_kernel, dim3(132), dim3(128), 0, st, pdl != 0, reinterpret_cast<const uint8_t*>(W2),
           reinterpret_cast<const uint8_t*>(S2), act, nact, max_items, (int)(CODE / 8192), (int)((SCALE + 8191) / 8192), CODE, SCALE);
}

// cfg bits: nmax (1,2,4) | nt (2,4) | nw (4,8) | ring (0/1)
extern "C" int wm_gemm_c(int mode, int nmax, int nt, int nw, int ring, int dbg, const void* W, const void* S, const float* GS,
                         const void* X, void* OUT, const int* cnt, const int* lst, const int* act, const int* nact,
                         const int* woff, int maxslots, int lmax, cudaStream_t st) {
  const int pdl = g_pdl;
#define WM_WG(M, N_, S_) if (mode == M && ring == 7 && nmax == N_ && nt == S_ && dbg == 0) { launch_wg<M, N_, S_>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
#undef WM_WG
#define WM_RS(M, T, N, S_) \
  if (mode == M && ring == 4 && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RS(0, 4, 4, 4) WM_RS(1, 4, 4, 4) WM_RS(0, 4, 2, 4) WM_RS(1, 4, 2, 4) WM_RS(0, 4, 1, 4) WM_RS(1, 4, 1, 4)
  WM_RS(0, 2, 4, 4) WM_RS(1, 2, 4, 4) WM_RS(0, 2, 2, 4) WM_RS(1, 2, 2, 4) WM_RS(0, 2, 1, 4) WM_RS(1, 2, 1, 4)
  WM_RS(0, 2, 8, 4) WM_RS(1, 2, 8, 4) WM_RS(0, 2, 8, 3) WM_RS(1, 2, 8, 3) WM_RS(0, 4, 8, 3) WM_RS(1, 4, 8, 3)
#undef WM_RS
#define WM_RS(M, T, N, S_, R_) \
  if (mode == M && ring == R_ && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RS(0, 4, 4, 6, 5) WM_RS(1, 4, 4, 6, 5) WM_RS(0, 2, 4, 6, 5) WM_RS(1, 2, 4, 6, 5) WM_RS(0, 4, 4, 8, 6) WM_RS(1, 4, 4, 8, 6) WM_RS(0, 2, 4, 8, 6) WM_RS(1, 2, 4, 8, 6)
  WM_RS(0, 4, 2, 6, 5) WM_RS(1, 4, 2, 6, 5) WM_RS(0, 4, 1, 6, 5) WM_RS(1, 4, 1, 6, 5)
  WM_RS(0, 4, 4, 3, 8) WM_RS(1, 4, 4, 3, 8) WM_RS(0, 2, 4, 3, 8) WM_RS(1, 2, 4, 3, 8) WM_RS(0, 2, 8, 3, 9) WM_RS(1, 2, 8, 3, 9) WM_RS(0, 2, 4, 3, 9) WM_RS(1, 2, 4, 3, 9)
#undef WM_RS
#define WM_RSD(M, D) if (mode == M && ring == 4 && nt == 4 && nmax == 4 && dbg == D) { launch_rs<M, 4, 4, 4, D>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RSD(0, 1) WM_RSD(1, 1) WM_RSD(0, 2) WM_RSD(1, 2) WM_RSD(0, 8) WM_RSD(1, 8) WM_RSD(0, 4) WM_RSD(1, 4) WM_RSD(0, 15) WM_RSD(1, 15) WM_RSD(0, 3) WM_RSD(1, 3) WM_RSD(0, 16) WM_RSD(1, 16) WM_RSD(0, 32) WM_RSD(1, 32) WM_RSD(0, 48) WM_RSD(1, 48) WM_RSD(0, 63) WM_RSD(1, 63) WM_RSD(0, 31) WM_RSD(1, 31) WM_RSD(0, 64) WM_RSD(1, 64)
#undef WM_RSD
#define WM_D(M, T, N, W_, R)                                                                             \
  if (mode == M && nt == T && nmax == N && nw == W_ && ring == R && dbg == 0) {                           \
    launch<M, T, N, W_, R, 0>(W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax, pdl, st);          \
    return 0;                                                                                             \
  }
#define WM_DBG(M, T, N, W_, R, D)                                                                         \
  if (mode == M && nt == T && nmax == N && nw == W_ && ring == R && dbg == D) {                           \
    launch<M, T, N, W_, R, D>(W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax, pdl, st);          \
    return 0;                                                                                             \
  }
#define WM_DBGS(M, T, N, W_, R) WM_DBG(M, T, N, W_, R, 1) WM_DBG(M, T, N, W_, R, 2) WM_DBG(M, T, N, W_, R, 8) WM_DBG(M, T, N, W_, R, 3) WM_DBG(M, T, N, W_, R, 11) WM_DBG(M, T, N, W_, R, 4) WM_DBG(M, T, N, W_, R, 9)
#define WM_ALL(T, N, W_, R) WM_D(0, T, N, W_, R) WM_D(1, T, N, W_, R)
  WM_ALL(2, 1, 4, 0) WM_ALL(2, 2, 4, 0) WM_ALL(2, 1, 8, 0) WM_ALL(2, 2, 8, 0) WM_ALL(4, 1, 4, 0) WM_ALL(4, 2, 4, 0)
  WM_ALL(2, 1, 4, 1) WM_ALL(2, 2, 4, 1) WM_ALL(2, 4, 4, 1) WM_ALL(4, 1, 4, 1) WM_ALL(4, 2, 4, 1) WM_ALL(4, 4, 4, 1)
  WM_ALL(2, 1, 8, 1) WM_ALL(2, 2, 8, 1) WM_ALL(2, 4, 8, 1)
  WM_ALL(8, 1, 4, 1) WM_ALL(8, 2, 4, 1) WM_ALL(8, 4, 4, 1)
  WM_ALL(4, 4, 4, 3) WM_ALL(4, 2, 4, 3) WM_ALL(4, 1, 4, 3) WM_ALL(8, 4, 4, 3) WM_ALL(8, 2, 4, 3)
  WM_ALL(4, 1, 2, 1) WM_ALL(4, 2, 2, 1) WM_ALL(4, 4, 2, 1) WM_ALL(2, 1, 2, 0) WM_ALL(2, 2, 2, 0) WM_ALL(2, 1, 2, 1)
  WM_ALL(2, 1, 4, 2) WM_ALL(2, 2, 4, 2) WM_ALL(2, 4, 4, 2) WM_ALL(4, 1, 4, 2) WM_ALL(4, 2, 4, 2) WM_ALL(4, 4, 4, 2)
  WM_ALL(4, 1, 4, 8) WM_ALL(4, 2, 4, 8) WM_ALL(4, 4, 4, 8) WM_ALL(2, 1, 4, 8) WM_ALL(2, 2, 4, 8) WM_ALL(2, 4, 4, 8) WM_ALL(2, 1, 8, 8) WM_ALL(2, 2, 8, 8) WM_ALL(2, 1, 2, 8)
  WM_ALL(4, 1, 4, 9) WM_ALL(4, 2, 4, 9) WM_ALL(4, 4, 4, 9) WM_ALL(2, 1, 4, 9) WM_ALL(2, 2, 4, 9) WM_ALL(2, 4, 4, 9)
  WM_ALL(2, 1, 8, 2) WM_ALL(2, 2, 8, 2) WM_ALL(2, 4, 8, 2)
#ifdef WM_DBG_BUILD
  WM_DBGS(0, 2, 1, 4, 0) WM_DBGS(1, 2, 1, 4, 0) WM_DBGS(0, 2, 1, 4, 2) WM_DBGS(1, 2, 1, 4, 2) WM_DBGS(0, 2, 2, 4, 2) WM_DBGS(1, 2, 2, 4, 2)
  WM_DBGS(0, 4, 4, 4, 2) WM_DBGS(1, 4, 4, 4, 2) WM_DBGS(0, 2, 4, 4, 2) WM_DBGS(1, 2, 4, 4, 2)
  WM_DBGS(0, 4, 4, 4, 1) WM_DBGS(1, 4, 4, 4, 1)
#endif
#undef WM_ALL
#undef WM_D
  return 1;
}
"""

TORCH_GLUE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" void wm_route_c(const int*, int, const int*, int, int, int*, int*, int*, int*, int*, int*, int, int, cudaStream_t);
extern "C" void wm_set_pdl(int);
void wm_set_pdl_t(int64_t v) { wm_set_pdl((int)v); }
extern "C" void wm_set_persist(int);
void wm_set_persist_t(int64_t v) { wm_set_persist((int)v); }
extern "C" void wm_finish_c(const void*, const float*, const int*, const void*, void*, int, float, cudaStream_t);
void wm_finish(torch::Tensor y, torch::Tensor tw, torch::Tensor lids, torch::Tensor nw, torch::Tensor out, double eps) {
  wm_finish_c(y.data_ptr(), tw.data_ptr<float>(), lids.data_ptr<int>(), nw.data_ptr(), out.data_ptr(), (int)out.size(0), (float)eps,
              at::cuda::getCurrentCUDAStream());
}
extern "C" void wm_permx_c(const void*, void*, int, cudaStream_t);
void wm_permx(torch::Tensor x, torch::Tensor xp) { wm_permx_c(x.data_ptr(), xp.data_ptr(), (int)x.size(0), at::cuda::getCurrentCUDAStream()); }
extern "C" int wm_gemm_c(int, int, int, int, int, int, const void*, const void*, const float*, const void*, void*, const int*,
                         const int*, const int*, const int*, const int*, int, int, cudaStream_t);
extern "C" void wm_pf_c(const void*, const void*, const int*, const int*, int, int, cudaStream_t);
void wm_pf(torch::Tensor w2, torch::Tensor s2, torch::Tensor act, torch::Tensor nact, int64_t max_items, int64_t pdl) {
  wm_pf_c(w2.data_ptr(), s2.data_ptr(), act.data_ptr<int>(), nact.data_ptr<int>(), (int)max_items, (int)pdl, at::cuda::getCurrentCUDAStream());
}

void wm_route(torch::Tensor ids, torch::Tensor ls, int64_t ls_val, int64_t lmax, torch::Tensor cnt, torch::Tensor lst,
              torch::Tensor act, torch::Tensor nact, torch::Tensor local_ids, torch::Tensor woff, int64_t grp) {
  wm_route_c(ids.data_ptr<int>(), (int)ids.numel(), ls.numel() ? ls.data_ptr<int>() : nullptr, (int)ls_val, (int)lmax, cnt.data_ptr<int>(),
             lst.data_ptr<int>(), act.data_ptr<int>(), nact.data_ptr<int>(), local_ids.data_ptr<int>(), woff.data_ptr<int>(), (int)grp,
             (int)(act.numel() / 4), at::cuda::getCurrentCUDAStream());
}

void wm_gemm_run(int64_t mode, int64_t nmax, int64_t nt, int64_t nw, int64_t ring, int64_t dbg, torch::Tensor W, torch::Tensor S, torch::Tensor GS, torch::Tensor X,
                 torch::Tensor OUT, torch::Tensor cnt, torch::Tensor lst, torch::Tensor act, torch::Tensor nact,
                 torch::Tensor woff, int64_t maxslots, int64_t lmax) {
  int r = wm_gemm_c((int)mode, (int)nmax, (int)nt, (int)nw, (int)ring, (int)dbg, W.data_ptr(), S.data_ptr(), GS.data_ptr<float>(), X.data_ptr(),
                    OUT.data_ptr(), cnt.data_ptr<int>(), lst.data_ptr<int>(), act.data_ptr<int>(),
                    nact.data_ptr<int>(), woff.data_ptr<int>(), (int)maxslots, (int)lmax, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported config");
}
"""

CPP_SRC = r"""
void wm_set_pdl_t(int64_t v);
void wm_set_persist_t(int64_t v);
void wm_pf(torch::Tensor w2, torch::Tensor s2, torch::Tensor act, torch::Tensor nact, int64_t max_items, int64_t pdl);
void wm_permx(torch::Tensor x, torch::Tensor xp);
void wm_finish(torch::Tensor y, torch::Tensor tw, torch::Tensor lids, torch::Tensor nw, torch::Tensor out, double eps);
void wm_route(torch::Tensor ids, torch::Tensor ls, int64_t ls_val, int64_t lmax, torch::Tensor cnt, torch::Tensor lst,
              torch::Tensor act, torch::Tensor nact, torch::Tensor local_ids, torch::Tensor woff, int64_t grp);
void wm_gemm_run(int64_t mode, int64_t nmax, int64_t nt, int64_t nw, int64_t ring, int64_t dbg, torch::Tensor W, torch::Tensor S, torch::Tensor GS, torch::Tensor X,
                 torch::Tensor OUT, torch::Tensor cnt, torch::Tensor lst, torch::Tensor act, torch::Tensor nact,
                 torch::Tensor woff, int64_t maxslots, int64_t lmax);
"""


def repack(w, sf):
    """w: uint8 [E, R, K/2] (low nibble first), sf: float8_e4m3fn [E, R, K/16].
    Returns (words int32 [E, R/16, K/64, 32, 4], scales int32-viewable uint8 [E, R/16, K/256, 32, 8])."""
    import torch

    E, R, K2 = w.shape
    K = K2 * 2
    RT, KCn = R // 16, K // 64
    out_w = torch.empty((E, RT, KCn, 32, 4), dtype=torch.int32, device=w.device)
    out_s = torch.empty((E, RT, KCn // 4, 32, 8), dtype=torch.uint8, device=w.device)
    # per-reg source bit positions for (magnitude bits m0..m2, sign), low half; high half is +16
    src = {
        0: (6, 7, 8, 15),
        1: (9, 10, 11, 18),
        2: (12, 13, 14, 21),
        3: (4, 0, 1, 3),
    }
    for e in range(E):
        t = w[e]
        codes = torch.stack((t & 15, t >> 4), dim=-1).reshape(R, K).to(torch.int64)
        # [rt, rh, g, kc, t, ks, p, h]
        c8 = codes.view(RT, 2, 8, KCn, 4, 4, 2, 2)
        word = torch.zeros((RT, 8, KCn, 4, 4), dtype=torch.int64, device=w.device)  # [rt,g,kc,t,ks]
        for rh in range(2):
            for p in range(2):
                j = rh + 2 * p
                for h in range(2):
                    c = c8[:, rh, :, :, :, :, p, h]  # [rt, g, kc, t, ks]
                    for bit in range(4):
                        pos = (src[j][bit] + 16 * h) % 32
                        word |= ((c >> bit) & 1) << pos
        word = word.permute(0, 2, 1, 3, 4)  # [rt, kc, g, t, ks]
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        out_w[e] = word.reshape(RT, KCn, 32, 4)
        # scales: raw e4m3 -> b' with S*2^118 = bf16 bits ((b'<<4) + 0x7600)
        s = sf[e].float() * (2.0 ** 118)
        bits = (s.view(torch.int32) >> 16).clamp(min=0x7600)
        bp = ((bits - 0x7600) >> 4).to(torch.uint8)  # [R, K/16]
        bp = bp.view(RT, 2, 8, KCn // 4, 4, 4)  # [rt, rh, g, kcg, c, t]; group = kcg*16 + c*4 + t
        out_s[e] = bp.permute(0, 3, 2, 5, 1, 4).reshape(RT, KCn // 4, 32, 8)  # [rt, kcg, (g,t), (rh,c)]
    return out_w, out_s
