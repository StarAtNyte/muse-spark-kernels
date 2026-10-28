#include <cooperative_groups.h>

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
  wm_epilogue<NTB, NCH, NW>(acc, red, GS[e], lst, ntok, tbase, MODE, NT, OUT, bx);
}

#ifndef WM_MINB
#define WM_MINB 4
#endif

__global__ void __launch_bounds__(128,4) moe_coop(
 const uint4* W13,const uint2* S13,const float* GS13,const uint4* W2,const uint2* S2,const float* GS2,
 const bf16* X,const int* IDS,const float* TW,const bf16* NWT,const int* PIDS,
 bf16* INTER,bf16* Y,float* NSQ,bf16* OUT,float eps,int prefetch){
 extern __shared__ uint4 scratch[];
 __shared__ int experts[8],pairs[8],nlocal;
 __shared__ float scales[8];
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 if(tid==0){int n=0;
#pragma unroll
  for(int p=0;p<8;p++){int e=IDS[p];if((unsigned)e<32){experts[n]=e;pairs[n]=p;n++;}}
  nlocal=n;
 }
 __syncthreads();
 auto grid=cooperative_groups::this_grid();
 // Fixed resident workers distribute only live experts' work. Nothing exits
 // before the grid barriers, including workers with no local work.
 for(int job=blockIdx.x;job<nlocal*256;job+=gridDim.x){
  int a=job>>8,bx=job&255,e=experts[a],p=pairs[a];
  if(prefetch && tid==0){
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(policy));
    const void* wp=W2+((size_t)e*256+bx)*64*32;
    const void* sp=S2+((size_t)e*256+bx)*16*32;
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"::"l"(wp),"r"(32768),"l"(policy):"memory");
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"::"l"(sp),"r"(4096),"l"(policy):"memory");
  }
  wm_body<0,2,2,1,4,0,6,0>(W13,S13,GS13,X,INTER,PIDS+p,e,1,0,scratch,bx);
  __syncthreads();
 }
 grid.sync();
 for(int job=blockIdx.x;job<nlocal*256;job+=gridDim.x){
  int a=job>>8,bx=job&255,e=experts[a],p=pairs[a];
  wm_body<1,1,1,1,4,0,6,0>(W2,S2,GS2,INTER,Y,PIDS+p,e,1,0,scratch,bx);
  __syncthreads();
  if(tid<16){
   int d=bx*16+tid;float v=__bfloat162float(Y[p*4096+d])*__bfloat162float(NWT[d]);
   float square=v*v;
#pragma unroll
   for(int k=8;k;k>>=1)square+=__shfl_xor_sync(0xffff,square,k);
   if(tid==0)NSQ[p*256+bx]=square;
  }
  __syncthreads();
 }
 grid.sync();
 // Two blocks finish disjoint halves of the output. Norm sufficient
 // statistics come from the down-projection producers, after BF16 rounding.
 if(blockIdx.x<2){
#pragma unroll
  for(int pass=0;pass<2;pass++){
   int p=warp+pass*4;float ss=0.f;bool live=(unsigned)IDS[p]<32;
   if(live){
#pragma unroll
    for(int j=0;j<8;j++)ss+=NSQ[p*256+lane+j*32];
#pragma unroll
    for(int k=16;k;k>>=1)ss+=__shfl_xor_sync(0xffffffff,ss,k);
   }
   if(lane==0)scales[p]=live?TW[p]*rsqrtf(ss/4096.f+eps):0.f;
  }
  __syncthreads();
  int column=(blockIdx.x*128+tid)*16;
  float acc[16]={},nw[16];
#pragma unroll
  for(int j=0;j<16;j++)nw[j]=__bfloat162float(NWT[column+j]);
#pragma unroll
  for(int p=0;p<8;p++){
   float scale=scales[p];
   if(scale!=0.f){
#pragma unroll
    for(int j=0;j<16;j++)acc[j]+=scale*(__bfloat162float(Y[p*4096+column+j])*nw[j]);
   }
  }
#pragma unroll
  for(int j=0;j<16;j++)OUT[column+j]=__float2bfloat16_rn(acc[j]);
 }
}
extern "C" int coop_launch(const void* w13,const void* s13,const float* gs13,const void* w2,const void* s2,const float* gs2,
 const void* x,const int* ids,const float* tw,const void* nw,const int* pids,void* inter,void* y,float* nsq,void* out,float eps,int cap,int prefetch,cudaStream_t stream){
 int dev=0,sms=0,supported=0,resident=0;
 cudaError_t e=cudaGetDevice(&dev);if(e)return e;
 e=cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,dev);if(e)return e;
 if(!supported)return cudaErrorNotSupported;
 e=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);if(e)return e;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,moe_coop,128,4096);if(e)return e;
 int blocks=sms*resident;if(cap>0&&blocks>cap)blocks=cap;
 if(blocks<2)return cudaErrorCooperativeLaunchTooLarge;
 const uint4* W13=(const uint4*)w13;const uint2* S13=(const uint2*)s13;
 const uint4* W2=(const uint4*)w2;const uint2* S2=(const uint2*)s2;
 const bf16* X=(const bf16*)x;const bf16* NWT=(const bf16*)nw;
 bf16* INTER=(bf16*)inter;bf16* Y=(bf16*)y;bf16* OUT=(bf16*)out;
 void* args[]={&W13,&S13,&gs13,&W2,&S2,&gs2,&X,&ids,&tw,&NWT,&pids,&INTER,&Y,&nsq,&OUT,&eps,&prefetch};
 return cudaLaunchCooperativeKernel((void*)moe_coop,dim3(blocks),dim3(128),args,4096,stream);
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int coop_launch(const void*,const void*,const float*,const void*,const void*,const float*,const void*,const int*,const float*,const void*,const int*,void*,void*,float*,void*,float,int,int,cudaStream_t);
torch::Tensor cooperative_moe(torch::Tensor w13,torch::Tensor s13,torch::Tensor gs13,torch::Tensor w2,torch::Tensor s2,torch::Tensor gs2,torch::Tensor x,torch::Tensor ids,torch::Tensor tw,torch::Tensor nw,torch::Tensor pids,double eps,int64_t cap,int64_t prefetch){
 TORCH_CHECK(x.size(0)==1,"single-token prototype");
 auto inter=torch::empty({8,4096},x.options()),y=torch::empty_like(inter),out=torch::empty_like(x);
 auto nsq=torch::empty({8,256},x.options().dtype(torch::kFloat32));
 int e=coop_launch(w13.data_ptr(),s13.data_ptr(),gs13.data_ptr<float>(),w2.data_ptr(),s2.data_ptr(),gs2.data_ptr<float>(),x.data_ptr(),ids.data_ptr<int>(),tw.data_ptr<float>(),nw.data_ptr(),pids.data_ptr<int>(),inter.data_ptr(),y.data_ptr(),nsq.data_ptr<float>(),out.data_ptr(),eps,cap,prefetch,at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));return out;
}
