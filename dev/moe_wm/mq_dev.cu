
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
                             int* __restrict__ nact, int* __restrict__ local_ids, int* __restrict__ woff, int grp) {
  __shared__ int s_cnt[32];
  pdl_wait();
  const int ls = ls_ptr ? *ls_ptr : ls_val;
  pdl_trigger();
  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;
  if (threadIdx.x < 128) reinterpret_cast<int4*>(act)[threadIdx.x] = make_int4(0, 0, 0, 0);  // work items {expert, offset, tokens, 0}
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
    int scan = groups;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, scan, d);
      if (lane >= d) scan += v;
    }
    const int start = scan - groups;
    for (int j = 0; j < groups; j++)
      reinterpret_cast<int4*>(act)[start + j] = make_int4(lane, j * grp, min(c - j * grp, grp), 1);
    if (lane == 31) *nact = scan;
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
        const float gate = bfr(s * gs), up = bfr(s2 * gs);
        const float sil = gate / (1.0f + __expf(-gate));
        OUT[(size_t)p * ID + (bx * (nt / 2) + ((tbase + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);
      } else {
        OUT[(size_t)p * HD + (bx * nt + tbase + n) * 16 + row] = __float2bfloat16(s * gs);
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
    auto issue = [&](int c) {
      if (c < CPW && !(DBG & 8)) {
#pragma unroll
        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * NTB + n) * 32 + lane], wsrc[n] + c * 32);
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
      cp_wait<S - 2>();
      issue(i + S - 1);
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
  wm_epilogue<NTB, NCH, NW>(acc, red, GS[e] * 256.0f, lst, ntok, tbase, MODE, NT, OUT, bx);
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
      wm_body<MODE, NT, NT, 1, NW, RING, (RING == 2 ? ring2_stages<NT, 1>() : (NT == 8 ? 3 : NT == 4 ? 4 : 6)), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, 0, smem, bx);
    } else if (NMAX >= 2 && ntok <= 16) {
#pragma unroll 1
      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))
        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 2 ? 2 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 2 ? 2 : 1)>() : RING == 3 ? 4 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);
    } else if (NMAX >= 4) {
#pragma unroll 1
      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))
        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 4 ? 4 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 4 ? 4 : 1)>() : RING == 3 ? 4 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);
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
  uint4* wring = smem;                       // [S][4][NTB][32]
  uint4* xring = smem + S * 4 * NTB * 32;    // [S][XSLOTS]

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

  auto issue = [&](int c) {
    if (c < KC) {
#pragma unroll
      for (int n = 0; n < NTB; n++) { if constexpr (!(DBG & 8)) cp16(&wring[(((c % S) * 4 + warp) * NTB + n) * 32 + lane], wsrc[n] + c * 32); }
#pragma unroll
      for (int r = 0; r < RPT; r++)
        if (tid + 128 * r < XSLOTS && !(DBG & 4)) cp16z(&xring[(c % S) * XSLOTS + tid + 128 * r], xb[r] + c * 64, xok[r] ? 16u : 0u);
    }
    cp_commit();
  };
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
  uint2 scur[NTB], snxt[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) {
    scur[n] = __ldg(ssrc[n]);
    snxt[n] = __ldg(ssrc[n] + 32);
  }
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
          const uint4 xs = xr[h][ks >> 1];
          if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
          else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
        }
      }
    }
    if (cq == 3) {
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        scur[n] = snxt[n];
        if ((i >> 2) + 2 < KCG) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);
      }
    }
  }
  cp_wait<0>();
  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);

  const float gs = GS[e] * 256.0f;
#pragma unroll
  for (int c = 0; c < NCH; c++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int col = c * 8 + 2 * t + (r & 1);
      if (col >= ntok) continue;
      const int p = __ldcg(lst + col);
      const int row = g + 8 * (r >> 1);
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        if (MODE == 0) {
          if (n & 1) continue;
          const float gate = bfr(acc[n][c][r] * gs), up = bfr(acc[n + 1][c][r] * gs);
          const float sil = gate / (1.0f + __expf(-gate));
          OUT[(size_t)p * ID + (bx * (NTC / 2) + ((warp * NTB + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);
        } else {
          OUT[(size_t)p * HD + (bx * NTC + warp * NTB + n) * 16 + row] = __float2bfloat16(acc[n][c][r] * gs);
        }
      }
    }
}

template <int MODE, int NTB, int NMAX, int S, int DBG>
__global__ void __launch_bounds__(128, 3)
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
  else if (NMAX >= 4) wm_body_rs<MODE, NTB, (NMAX >= 4 ? 4 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
}

template <int MODE, int NTB, int NMAX, int S, int DBG>
static void launch_rs(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst,
                      const int* act, int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid((MODE == 0 ? 512 : 256) / (4 * NTB), (unsigned)maxslots);
  const int dyn = S * (4 * NTB * 512 + NMAX * 2 * 32 * 16);
  cudaFuncSetAttribute(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),
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
                           int* act, int* nact, int* local_ids, int* woff, int grp, cudaStream_t st) {
  const int pdl = g_pdl;
  launch_k(route_kernel, dim3(1), dim3(1024), 0, st, pdl != 0, ids, npair, ls_ptr, ls_val, lmax, cnt, lst, act, nact, local_ids, woff, grp);
}

template <int MODE, int NT, int NMAX, int NW, int RING, int DBG>
static void launch(const void* W, const void* S, const float* GS, const void* X, void* OUT, const int* cnt,
                   const int* lst, const int* act, const int* nact, const int* woff, int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid((MODE == 0 ? 512 : 256) / NT, (unsigned)maxslots);
  if (g_persist) { const unsigned cap = 132u * (unsigned)g_persist; const unsigned tot = grid.x * grid.y; grid = dim3(tot < cap ? tot : cap, 1); }
  int dyn = NW * NT * 6 * 512 > 49152 ? 49152 : NW * NT * 6 * 512;
  if (NT == 8) dyn = NW * 3 * 8 * 512 + 1024;
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

// cfg bits: nmax (1,2,4) | nt (2,4) | nw (4,8) | ring (0/1)
extern "C" int wm_gemm_c(int mode, int nmax, int nt, int nw, int ring, int dbg, const void* W, const void* S, const float* GS,
                         const void* X, void* OUT, const int* cnt, const int* lst, const int* act, const int* nact,
                         const int* woff, int maxslots, int lmax, cudaStream_t st) {
  const int pdl = g_pdl;
#define WM_RS(M, T, N, S_) \
  if (mode == M && ring == 4 && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RS(0, 4, 4, 4) WM_RS(1, 4, 4, 4) WM_RS(0, 4, 2, 4) WM_RS(1, 4, 2, 4) WM_RS(0, 4, 1, 4) WM_RS(1, 4, 1, 4)
  WM_RS(0, 2, 4, 4) WM_RS(1, 2, 4, 4) WM_RS(0, 2, 2, 4) WM_RS(1, 2, 2, 4) WM_RS(0, 2, 1, 4) WM_RS(1, 2, 1, 4)
#undef WM_RS
#define WM_RS(M, T, N, S_, R_) \
  if (mode == M && ring == R_ && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RS(0, 4, 4, 6, 5) WM_RS(1, 4, 4, 6, 5) WM_RS(0, 2, 4, 6, 5) WM_RS(1, 2, 4, 6, 5) WM_RS(0, 4, 4, 8, 6) WM_RS(1, 4, 4, 8, 6) WM_RS(0, 2, 4, 8, 6) WM_RS(1, 2, 4, 8, 6)
  WM_RS(0, 4, 2, 6, 5) WM_RS(1, 4, 2, 6, 5) WM_RS(0, 4, 1, 6, 5) WM_RS(1, 4, 1, 6, 5)
#undef WM_RS
#define WM_RSD(M, D) if (mode == M && ring == 4 && nt == 4 && nmax == 4 && dbg == D) { launch_rs<M, 4, 4, 4, D>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RSD(0, 1) WM_RSD(1, 1) WM_RSD(0, 2) WM_RSD(1, 2) WM_RSD(0, 8) WM_RSD(1, 8) WM_RSD(0, 4) WM_RSD(1, 4) WM_RSD(0, 15) WM_RSD(1, 15) WM_RSD(0, 3) WM_RSD(1, 3)
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
  WM_ALL(2, 1, 8, 2) WM_ALL(2, 2, 8, 2) WM_ALL(2, 4, 8, 2)
#ifdef WM_DBG_BUILD
  WM_DBGS(0, 2, 1, 4, 0) WM_DBGS(1, 2, 1, 4, 0) WM_DBGS(0, 2, 1, 4, 2) WM_DBGS(1, 2, 1, 4, 2) WM_DBGS(0, 2, 2, 4, 2) WM_DBGS(1, 2, 2, 4, 2)
  WM_DBGS(0, 4, 4, 4, 2) WM_DBGS(1, 4, 4, 4, 2) WM_DBGS(0, 2, 4, 4, 2) WM_DBGS(1, 2, 4, 4, 2)
#endif
#undef WM_ALL
#undef WM_D
  return 1;
}

// ---------------------------------------------------------------------------------------------------------------------
// mq: weights/scales exactly as SGLang's gptq_marlin_repack / marlin scale processing leave them.
//   weights: int32 [E, K/16, NTILES*128]; word index ((kt * NTILES + nt) * 128 + lane * 4 + wq): the 8 nibbles of
//            (16 rows n = nt*64 + wq*16 + {g, g+8}) x (k16 step kt) = one mma A fragment, nibble pairs (j, j+4) per register.
//   scales : uint8 [E, K/16, N]; position within a 64-row block is permuted so that lane g finds its 8 scales (rows g/g+8 of
//            the 4 sub-tiles) in the 8 contiguous bytes at offset 8 g.  byte B -> bf16 bits ((B << 4) + 0x6F80) = S * 2^118.
//   global scale: stored as gs * 2^119 in bf16; the accumulator is rescaled by stored * 2^-111 (undoes the 2^-8 of the above).
// ---------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t mq_dq(uint32_t q, int j) {
  const uint32_t t = j == 3 ? q : (q << (4 * (3 - j)));
  const uint32_t m = (t & 0x70007000u) >> 6;
  uint32_t o;
  asm("lop3.b32 %0, %1, %2, %3, 0xea;\n" : "=r"(o) : "r"(t), "r"(0x80008000u), "r"(m));
  return o;
}
__device__ __forceinline__ void mq_ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ uint32_t mq_scale(uint32_t w32, int b) {
  const uint32_t x = __byte_perm(w32, 0u, 0x4040u | (b << 8) | b);   // byte b in both 16-bit lanes
  return x * 16u + 0x6F806F80u;
}

template <int MODE, int NCH, int S, int DBG>
__device__ __forceinline__ void mq_body(const uint4* __restrict__ Wm, const uint8_t* __restrict__ Sm, const float* __restrict__ GS,
                                        const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst, int e,
                                        int ntok, uint8_t* smem_raw, const int bx) {
  constexpr int NTILES = MODE == 0 ? 128 : 64;   // 64-row tiles of the layer
  constexpr int NROWS = NTILES * 64;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int XSLOTS = NCH * 64;      // 16-byte granules of the x tile: NCH*8 token rows x 8
  constexpr int XPITCH = 144;           // bytes per token row (128 + pad) -> conflict-free ldmatrix
  constexpr int XSTAGE = NCH * 8 * XPITCH;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int nt = MODE == 0 ? ((warp < 4) ? (bx * 4 + warp) : (64 + bx * 4 + (warp - 4))) : (bx * 8 + warp);
  uint4* wring = reinterpret_cast<uint4*>(smem_raw);            // [S][8 warps][4 ks][32 lanes]
  uint8_t* xring = reinterpret_cast<uint8_t*>(wring + S * 8 * 4 * 32);   // [S][NCH*8][XPITCH]

  const uint4* wsrc = Wm + (size_t)e * (256 * NTILES * 32) + (size_t)nt * 32 + lane;
  const uint8_t* ssrc = Sm + (size_t)e * ((size_t)256 * NROWS) + (size_t)nt * 64 + 8 * g;

  const bf16* xb = X;
  bool xok = false;
  if (tid < XSLOTS) {
    const int col = tid >> 3, gr = tid & 7;
    if (col < ntok) {
      const int p = __ldcg(lst + col);
      xb = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + gr * 8;
      xok = true;
    }
  }
  float acc[4][NCH][4];
#pragma unroll
  for (int n = 0; n < 4; n++)
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;

  auto issue = [&](int c) {
    if (c < KC) {
#pragma unroll
      for (int ks = 0; ks < 4; ks++) cp16(&wring[(((c % S) * 8 + warp) * 4 + ks) * 32 + lane], wsrc + (size_t)(4 * c + ks) * (NTILES * 32));
      if (tid < XSLOTS) cp16z(xring + (c % S) * XSTAGE + (tid >> 3) * XPITCH + (tid & 7) * 16, xb + c * 64, xok ? 16u : 0u);
    }
    cp_commit();
  };
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
  uint2 scur[4], snxt[4];
#pragma unroll
  for (int ks = 0; ks < 4; ks++) {
    scur[ks] = __ldg(reinterpret_cast<const uint2*>(ssrc + (size_t)ks * NROWS));
    snxt[ks] = __ldg(reinterpret_cast<const uint2*>(ssrc + (size_t)(4 + ks) * NROWS));
  }
#pragma unroll 1
  for (int i = 0; i < KC; i++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(i + S - 1);
    uint32_t xr[NCH][4][2];   // [token group][ks][b0, b1]
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        const uint8_t* ap = xring + (i % S) * XSTAGE + (h * 8 + (lane & 7)) * XPITCH + (hf * 32 + (lane >> 3) * 8) * 2;
        mq_ldsm4(xr[h][2 * hf][0], xr[h][2 * hf][1], xr[h][2 * hf + 1][0], xr[h][2 * hf + 1][1], ap);
      }
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const uint4 q4 = wring[(((i % S) * 8 + warp) * 4 + ks) * 32 + lane];
      const uint32_t qw[4] = {q4.x, q4.y, q4.z, q4.w};
      const uint2 sc = scur[ks];
#pragma unroll
      for (int n = 0; n < 4; n++) {
        // scale bytes of (sub-tile n, row g / g+8): offsets 0,2 | 1,3 | 4,6 | 5,7
        const int bl = (n < 2) ? (n == 0 ? 0 : 1) : (n == 2 ? 4 : 5);
        const uint32_t wl = bl < 4 ? sc.x : sc.y;
        const int bsh = bl & 3;
        const uint32_t s_lo = mq_scale(wl, bsh);
        const uint32_t s_hi = mq_scale(wl, bsh + 2);
        const uint32_t a0 = hmul(mq_dq(qw[n], 0), s_lo);
        const uint32_t a1 = hmul(mq_dq(qw[n], 2), s_hi);
        const uint32_t a2 = hmul(mq_dq(qw[n], 1), s_lo);
        const uint32_t a3 = hmul(mq_dq(qw[n], 3), s_hi);
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          mma16816(acc[n][h], a0, a1, a2, a3, xr[h][ks][0], xr[h][ks][1]);
        }
      }
    }
    // next chunk's scales (two chunks ahead of use)
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      scur[ks] = snxt[ks];
      if (i + 2 < KC) snxt[ks] = __ldg(reinterpret_cast<const uint2*>(ssrc + (size_t)(4 * (i + 2) + ks) * NROWS));
    }
  }
  cp_wait<0>();
  __syncthreads();

  const float gs = GS[e];
  if (MODE == 0) {
    float* ex = reinterpret_cast<float*>(smem_raw);   // [NCH*8][260]
    if (warp >= 4) {
#pragma unroll
      for (int n = 0; n < 4; n++)
#pragma unroll
        for (int c = 0; c < NCH; c++)
#pragma unroll
          for (int r = 0; r < 4; r++) ex[(c * 8 + 2 * t + (r & 1)) * 260 + (warp - 4) * 64 + n * 16 + g + 8 * (r >> 1)] = acc[n][c][r];
    }
    __syncthreads();
    if (warp < 4) {
#pragma unroll
      for (int c = 0; c < NCH; c++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int col = c * 8 + 2 * t + (r & 1);
          if (col >= ntok) continue;
          const int p = __ldcg(lst + col);
#pragma unroll
          for (int n = 0; n < 4; n++) {
            const int fl = warp * 64 + n * 16 + g + 8 * (r >> 1);
            const float gate = bfr(acc[n][c][r] * gs), up = bfr(ex[col * 260 + fl] * gs);
            const float sil = gate / (1.0f + __expf(-gate));
            OUT[(size_t)p * ID + bx * 256 + fl] = __float2bfloat16(sil * up);
          }
        }
    }
  } else {
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int col = c * 8 + 2 * t + (r & 1);
        if (col >= ntok) continue;
        const int p = __ldcg(lst + col);
#pragma unroll
        for (int n = 0; n < 4; n++) OUT[(size_t)p * HD + nt * 64 + n * 16 + g + 8 * (r >> 1)] = __float2bfloat16(acc[n][c][r] * gs);
      }
  }
}

template <int MODE, int NMAX, int S, int DBG>
__global__ void __launch_bounds__(256, 2)
mq_gemm(const uint4* __restrict__ Wm, const uint8_t* __restrict__ Sm, const float* __restrict__ GS, const bf16* __restrict__ X,
        bf16* __restrict__ OUT, const int* __restrict__ lst, const int* __restrict__ act, int lmax) {
  extern __shared__ __align__(16) uint8_t smem_mq[];
  pdl_wait();
  pdl_trigger();
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);
  if (item.z == 0) return;
  const int e = item.x, ntok = item.z;
  const int* l = lst + e * lmax + item.y;
  if (ntok <= 8) mq_body<MODE, 1, S, DBG>(Wm, Sm, GS, X, OUT, l, e, ntok, smem_mq, blockIdx.x);
  else if (NMAX >= 2 && ntok <= 16) mq_body<MODE, (NMAX >= 2 ? 2 : 1), S, DBG>(Wm, Sm, GS, X, OUT, l, e, ntok, smem_mq, blockIdx.x);
  else if (NMAX >= 4) mq_body<MODE, (NMAX >= 4 ? 4 : 1), S, DBG>(Wm, Sm, GS, X, OUT, l, e, ntok, smem_mq, blockIdx.x);
}

template <int MODE, int NMAX, int S, int DBG>
static void launch_mq(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst, const int* act,
                      int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid(MODE == 0 ? 16 : 8, (unsigned)maxslots);
  int dyn = S * (8 * 4 * 32 * 16 + NMAX * 8 * 144);
  const int ex = NMAX * 8 * 260 * 4;
  if (ex > dyn) dyn = ex;
  cudaFuncSetAttribute(mq_gemm<MODE, NMAX, S, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(mq_gemm<MODE, NMAX, S, DBG>, grid, dim3(256), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const uint8_t*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);
}

extern "C" int mq_gemm_c(int mode, int nmax, int s, const void* W, const void* Sc, const float* GS, const void* X, void* OUT,
                         const int* lst, const int* act, int maxslots, int lmax, cudaStream_t st) {
  const int pdl = g_pdl;
#define MQ_D(M, N, S_) if (mode == M && nmax == N && s == S_) { launch_mq<M, N, S_, 0>(W, Sc, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  MQ_D(0, 4, 4) MQ_D(1, 4, 4) MQ_D(0, 2, 4) MQ_D(1, 2, 4) MQ_D(0, 1, 4) MQ_D(1, 1, 4)
  MQ_D(0, 4, 3) MQ_D(1, 4, 3) MQ_D(0, 4, 6) MQ_D(1, 4, 6) MQ_D(0, 2, 6) MQ_D(1, 2, 6) MQ_D(0, 1, 6) MQ_D(1, 1, 6)
#undef MQ_D
  return 1;
}
