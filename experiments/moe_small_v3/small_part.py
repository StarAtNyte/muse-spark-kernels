

# ======================================================================================= moe_small_v3: M <= 8 kernel
# A separate kernel for the small batches (M <= SMALL_MAX_M); every other size keeps the dispatch above, unchanged.
# Design (experiments/moe_small_v3/FINDINGS.md):
#  * one 132-CTA persistent grid; the work list is every (phase, expert, 16-row block) unit of the touched experts,
#    expert-major, A (gate/up) units first, then B (down) units: L = A(0) .. A(D-1), B(0) .. B(D-1).
#  * CTA c takes units c, c + G, c + 2G, ... (strided). Every CTA's count differs by at most one (the byte-balanced
#    optimum) and all CTAs sweep the experts in the same order, so expert k's inter is complete after about
#    (k + 1) / D of phase A instead of at the end of A: phase B's first experts never wait, and only the last
#    expert's B (processed last) depends on the end of A.
#  * one weight stream per CTA: a producer warp issues every unit's 36 KB TMA copy into an NS-slot ring with no
#    dependency on the compute (A and B weights alike); A -> B readiness gates only the compute.
#  * readiness: one credit per (CTA, expert) (fence + atomicAdd of the CTA's unit count when it finishes its last
#    A unit of that expert), polled by one poller lane per CTA, which flags it in shared memory.
#  * compute warps keep the current segment's x / inter fragments in registers and prefetch the next segment's into
#    a second register set as soon as it is ready, so expert transitions do not stall on L2.
#  * an epilogue warp reduces the 8 warps' partials (SwiGLU -> inter bf16; B: y * (1 + w) fp32 -> ybuf, per-route
#    sum of squares via fp32 atomics); one grid barrier at the end, then a distributed norm + weighted combine.
# Weights: the parent's unit-order copies (state["w"][10], [11]): lossless bit/byte permutations of the NVFP4 codes
# and e4m3 scales. FP32 accumulation; inter bf16 (as SGLang); y and its sums of squares FP32 until the final combine.

SMALL_MAX_M = 8
SMALL_DEFINES = ()   # extra -D flags of the production small build

SMALL_SRC = r'''
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#ifndef SM_NS
#define SM_NS 3
#endif
#ifndef SM_DBG
#define SM_DBG 0
#endif
#ifndef SM_L2H
#define SM_L2H 1     // evict-first L2 hint on the weight copies
#endif
#ifndef SM_NRB
#define SM_NRB 4
#endif
#ifndef SM_NWBF
#define SM_NWBF 0    // keep post_norm_w as bf16 in shared memory (8 KB instead of 16)
#endif
#ifndef SM_SLEEP
#define SM_SLEEP 64   // ns backoff in the non-compute warps' waits (0: busy spin)
#endif
#ifndef SM_NACC
#define SM_NACC 4     // independent MMA accumulator chains per unit (the parent's NACC 4)
#endif
#ifndef SM_SROWS
#define SM_SROWS 10  // shared-memory staging rows (x rows, then the B inter rows when M + pairs fits)
#endif
#ifndef SM_NOCMP
#define SM_NOCMP 0   // ablation: skip the decode + MMA (timing only, wrong output)
#endif
#ifndef SM_ORDER
#define SM_ORDER 0   // 0: strided over the expert-major list; 1: contiguous ranges (the parent's style, for A/B)
#endif

namespace sm3 {
constexpr int HD = 4096;
#ifndef SM_NWC
#define SM_NWC 8
#endif
constexpr int NWC = SM_NWC;                  // compute warps (8: 512 K each; 16: 256 K each)
constexpr int NCH = 64 / NWC;                // 64-K chunks per compute warp
#ifndef SM_CREDW
#define SM_CREDW 1   // a credit warp does the phase-A fence + atomicAdd (off the epilogue warp)
#endif
#ifndef SM_MBAR
#define SM_MBAR 1    // monotonic grid barrier: waiters poll the arrival count itself (no release hop)
#endif
constexpr int W_PROD = NWC, W_EPI = NWC + 1, W_POLL = NWC + 2, W_CRED = NWC + 3;
constexpr int NTH = (NWC + (SM_CREDW ? 4 : 3)) * 32;
constexpr int NS = SM_NS;
constexpr int NRB = SM_NRB;
constexpr int UNIT = 36864, UCODES = 32768;  // one 16-row block x full K: 32 KB codes + 4 KB scales
constexpr int MAXE = 64;                     // M * 8 entries, M <= 8
constexpr float TWO62 = 4.611686018427388e18f;
constexpr int SROWS = SM_SROWS, ROWP = 8192 + 16;   // staging row pitch: + 16 B keeps the fragment loads conflict-free

struct Params {
  const unsigned char* wAu; const unsigned char* wBu;
  const float* gsA; const float* gsB; const __nv_bfloat16* normw;
  const __nv_bfloat16* x; const int* ids; const float* tw;
  __nv_bfloat16* out; __nv_bfloat16* inter; float* ybuf; float* ss; int* ctl;
  unsigned long long* dbg;
  int M, lo, nloc;
  uint32_t zero;
};

struct __align__(16) Smem {
  unsigned char ring[NS][UNIT];
  float4 red[NRB][NWC][32];
  unsigned char stage[SROWS][ROWP];  // rows 0..M-1: x; rows M..M+P-1: inter (B), when M + P <= SROWS
#if SM_NWBF
  __nv_bfloat16 nwb[HD];
#else
  float nw1[HD];                 // 1 + post_norm_w (FP32)
#endif
  uint64_t full[NS], empty[NS], rfull[NRB], rempty[NRB], nwbar, xbar, ibar[32], cbar[32];
  int ccnt[32];                  // units of each A segment this CTA did (for the credit warp)
  int ek[32], nk[32], pk[32];    // per list index k: local expert, tokens, first pair
  int c0[32];
  int ptok[MAXE];                // pair -> token
  int pos[MAXE];                 // entry (token * 8 + slot) -> pair, -1 if not local
  float twv[MAXE];
  float gA[32], gB[32];
  volatile int ready[32];
  int D, U_A, U, gen, P, istaged;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t;
}
#define DBG(k) do { if (SM_DBG && p.dbg) p.dbg[blockIdx.x * 16 + (k)] = gtimer(); } while (0)
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
// a wait for the helper warps: back off between probes so they do not take issue slots from the compute warps
__device__ __forceinline__ void mbar_wait_slow(uint64_t* bar, uint32_t parity) {
  uint32_t ok = 0;
  while (true) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
    if (ok) break;
    if (SM_SLEEP) __nanosleep(SM_SLEEP);
  }
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// the parent's DQ7 decode: one word (8 E2M1 codes in the prepacked bit layout) -> 4 bf16x2 A registers (e2m1 * 2^-126)
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
}
// e4m3 scale byte b of word -> bf16x2(scale * 2^64) (positive normal bytes only; prepare() checks)
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)

__device__ __forceinline__ int nth_bit(uint32_t m, int k) {  // position of the (k+1)-th set bit
  for (int j = 0; j < k; ++j) m &= m - 1u;
  return __ffs(m) - 1;
}
// list position -> segment (A: k, B: 32 + k) and 16-row block
__device__ __forceinline__ void decode(int pos, int U_A, int& seg, int& rb) {
  if (pos < U_A) { seg = pos >> 9; rb = pos & 511; }
  else { const int q = pos - U_A; seg = 32 + (q >> 8); rb = q & 255; }
}
// the CTA's unit sequence. ORDER 0: strided, pos = c + G i over the whole list. ORDER 2: the parent's style, a
// contiguous range of the A list, then a contiguous range of the B list.
struct Seq {
  int n, nA, a0, b0, c, G;
  __device__ __forceinline__ void init(int c_, int G_, int U_A, int U) {
    c = c_; G = G_;
#if SM_ORDER == 0
    n = U > c ? (U - c - 1) / G + 1 : 0; nA = n; a0 = c; b0 = 0;
#else
    const int UB = U - U_A;
    a0 = (int)(((long long)U_A * c) / G); nA = (int)(((long long)U_A * (c + 1)) / G) - a0;
    b0 = U_A + (int)(((long long)UB * c) / G); n = nA + U_A + (int)(((long long)UB * (c + 1)) / G) - b0;
#endif
  }
  __device__ __forceinline__ int at(int i) const {
#if SM_ORDER == 0
    return c + G * i;
#else
    return i < nA ? a0 + i : b0 + (i - nA);
#endif
  }
};
__device__ __forceinline__ int seg_of(int pos, int U_A) { int s, r; decode(pos, U_A, s, r); return s; }

__device__ __forceinline__ void wait_ready(const Smem& S, int k) {
  while (S.ready[k] == 0) { }
  __threadfence_block();
}
// x / inter fragments of segment seg for this warp's 512-K slice: lane (g, t) holds row g's k = 64 c + 16 t .. + 15
__device__ __forceinline__ void load_x(const Params& p, Smem& S, int seg, int warp, int lane, uint4 (&xr)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const bool B = seg >= 32;
  const int k = seg & 31;
  const int n = S.nk[k], pk = S.pk[k];
  const bool staged = !B || S.istaged;
  if (staged) mbar_wait(B ? &S.ibar[k] : &S.xbar, 0u);
  else wait_ready(S, k);
  if (g < n) {
    if (staged) {
      const int row = B ? p.M + pk + g : S.ptok[pk + g];
      const unsigned char* src = &S.stage[row][(warp * (64 * NCH) + 16 * t) * 2];
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        xr[2 * c] = *reinterpret_cast<const uint4*>(src + c * 128);
        xr[2 * c + 1] = *reinterpret_cast<const uint4*>(src + c * 128 + 16);
      }
    } else {
      const uint4* src = reinterpret_cast<const uint4*>(p.inter + (size_t)(pk + g) * HD + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) { xr[2 * c] = __ldcg(src + c * 8); xr[2 * c + 1] = __ldcg(src + c * 8 + 1); }
    }
  } else {
#pragma unroll
    for (int c = 0; c < 2 * NCH; ++c) xr[c] = make_uint4(0u, 0u, 0u, 0u);
  }
}


extern __shared__ __align__(1024) unsigned char sm3_raw[];

__global__ void __launch_bounds__(NTH, 1) moe_small(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(sm3_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, cta = blockIdx.x;
  const int E = p.M * 8;

  if (warp == W_PROD) {
    // ---------------------------------------------------------------- producer: routing mask, then the weight stream
    if (lane == 0) DBG(0);
    const int e0 = lane < E ? __ldg(p.ids + lane) - p.lo : -1;
    const int e1 = lane + 32 < E ? __ldg(p.ids + lane + 32) - p.lo : -1;
    const uint32_t bits = ((e0 >= 0 && e0 < p.nloc) ? (1u << e0) : 0u) | ((e1 >= 0 && e1 < p.nloc) ? (1u << e1) : 0u);
    const uint32_t mask = __reduce_or_sync(0xffffffffu, bits);
    const int D = __popc(mask);
    const int U_A = 512 * D, U = 768 * D;
    Seq sq; sq.init(cta, G, U_A, U);
    int curk = -1, cure = 0, nI = 0;
    auto issue = [&](int i, int pos) {
      int seg, rb; decode(pos, U_A, seg, rb);
      const int k = seg & 31;
      if (k != curk) { curk = k; cure = nth_bit(mask, k); }
      const unsigned char* src = seg < 32 ? p.wAu + ((size_t)cure * 512 + rb) * UNIT : p.wBu + ((size_t)cure * 256 + rb) * UNIT;
      const int sl = i % NS;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&S.full[sl])), "r"(UNIT) : "memory");
#if SM_L2H
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                   ::"r"(smem_u32(S.ring[sl])), "l"(src), "r"(UNIT), "r"(smem_u32(&S.full[sl])), "l"(0x12F0000000000000ull) : "memory");
#else
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(S.ring[sl])), "l"(src), "r"(UNIT), "r"(smem_u32(&S.full[sl])) : "memory");
#endif
    };
    if (lane == 0) {
      for (int s = 0; s < NS; ++s) { mbar_init(&S.full[s], 1); mbar_init(&S.empty[s], NWC); }
      for (int s = 0; s < NRB; ++s) { mbar_init(&S.rfull[s], NWC); mbar_init(&S.rempty[s], 1); }
      mbar_init(&S.nwbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      for (; nI < sq.n && nI < NS; ++nI) issue(nI, sq.at(nI));
      DBG(1);
    }
    __syncwarp();
    asm volatile("bar.sync 1, %0;\n" ::"n"(NTH) : "memory");
    if (lane == 0) {
      for (int i = nI; i < sq.n; ++i) {
        mbar_wait_slow(&S.empty[i % NS], (uint32_t)(((i / NS) - 1) & 1));
        issue(i, sq.at(i));
      }
    }
    __syncwarp();
  } else if (warp == W_POLL) {
    // ---------------------------------------------------------------- poller: routing tables, readiness, grid release
    const int e0r = lane < E ? __ldg(p.ids + lane) - p.lo : -1;
    const int e1r = lane + 32 < E ? __ldg(p.ids + lane + 32) - p.lo : -1;
    const float tw0 = lane < E ? __ldg(p.tw + lane) : 0.f, tw1 = lane + 32 < E ? __ldg(p.tw + lane + 32) : 0.f;
    const float ga = lane < p.nloc ? __ldg(p.gsA + lane) : 0.f, gb = lane < p.nloc ? __ldg(p.gsB + lane) : 0.f;
    const int gen = ld_acquire(p.ctl + 65);
    const int e0 = (e0r >= 0 && e0r < p.nloc) ? e0r : -1, e1 = (e1r >= 0 && e1r < p.nloc) ? e1r : -1;
    const uint32_t mask = __reduce_or_sync(0xffffffffu, (e0 >= 0 ? (1u << e0) : 0u) | (e1 >= 0 ? (1u << e1) : 0u));
    const int D = __popc(mask);
    uint32_t lt; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lt));
    const int k0 = e0 >= 0 ? __popc(mask & ((1u << e0) - 1u)) : -1;
    const int k1 = e1 >= 0 ? __popc(mask & ((1u << e1) - 1u)) : -1;
    const uint32_t pe0 = __match_any_sync(0xffffffffu, k0), pe1 = __match_any_sync(0xffffffffu, k1);
    S.c0[lane] = 0; S.nk[lane] = 0; S.ready[lane] = 0;
    S.gA[lane] = ga; S.gB[lane] = gb;
    __syncwarp();
    if (k0 >= 0 && (pe0 & lt) == 0) S.c0[k0] = __popc(pe0);
    __syncwarp();
    if (k1 >= 0 && (pe1 & lt) == 0) S.nk[k1] = __popc(pe1);   // batch-1 count, for now
    __syncwarp();
    int n = S.c0[lane] + S.nk[lane];
    int inc = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += v; }
    __syncwarp();
    S.nk[lane] = n; S.pk[lane] = inc - n;
    S.ek[lane] = lane < D ? nth_bit(mask, lane) : 0;
    __syncwarp();
    const int pr0 = k0 >= 0 ? S.pk[k0] + __popc(pe0 & lt) : -1;
    const int pr1 = k1 >= 0 ? S.pk[k1] + S.c0[k1] + __popc(pe1 & lt) : -1;
    S.pos[lane] = pr0; S.pos[lane + 32] = pr1;
    S.twv[lane] = tw0; S.twv[lane + 32] = tw1;
    if (pr0 >= 0) S.ptok[pr0] = lane >> 3;
    if (pr1 >= 0) S.ptok[pr1] = (lane + 32) >> 3;
    const int P = __shfl_sync(0xffffffffu, inc, 31);
    if (lane == 0) {
      S.D = D; S.U_A = 512 * D; S.U = 768 * D; S.gen = gen; S.P = P; S.istaged = (p.M + P <= SROWS) ? 1 : 0;
      mbar_init(&S.xbar, 1);
      for (int k = 0; k < 32; ++k) { mbar_init(&S.ibar[k], 1); mbar_init(&S.cbar[k], 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&S.xbar)), "r"(p.M * 8192) : "memory");
      for (int t = 0; t < p.M; ++t)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     ::"r"(smem_u32(S.stage[t])), "l"(p.x + (size_t)t * HD), "r"(8192), "r"(smem_u32(&S.xbar)) : "memory");
    }
    if (cta == 0) {  // the next launch's counters and sums (the previous launch's copies)
      p.ctl[((gen + 1) & 1) * 32 + lane] = 0;
      p.ss[((gen + 1) & 1) * 64 + lane] = 0.f; p.ss[((gen + 1) & 1) * 64 + 32 + lane] = 0.f;
    }
    __syncwarp();
    asm volatile("bar.sync 1, %0;\n" ::"n"(NTH) : "memory");
    // 1 + post_norm_w into shared memory (needed from the first B unit on)
    {
      const uint4* src = reinterpret_cast<const uint4*>(p.normw);
#pragma unroll 4
      for (int j = lane; j < HD / 8; j += 32) {
        const uint4 v = __ldg(src + j);
#if SM_NWBF
        reinterpret_cast<uint4*>(S.nwb)[j] = v;
#else
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float2 f = __bfloat1622float2(h[q]);
          S.nw1[8 * j + 2 * q] = 1.f + f.x; S.nw1[8 * j + 2 * q + 1] = 1.f + f.y;
        }
#endif
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.nwbar);
    }
    if (lane == 0) {
      const int* cnt = p.ctl + (gen & 1) * 32;
      const bool stg = p.M + P <= SROWS;
      for (int k = 0; k < D; ++k) {
        while (ld_acquire(cnt + k) < 512) { if (SM_SLEEP) __nanosleep(SM_SLEEP); }
        if (stg) {  // the expert's inter rows -> shared memory (other SMs' generic stores -> this async-proxy read)
          asm volatile("fence.proxy.async.global;\n" ::: "memory");
          const int n = S.nk[k], pk = S.pk[k];
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&S.ibar[k])), "r"(n * 8192) : "memory");
          for (int j = 0; j < n; ++j)
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                         ::"r"(smem_u32(S.stage[p.M + pk + j])), "l"(p.inter + (size_t)(pk + j) * HD), "r"(8192), "r"(smem_u32(&S.ibar[k])) : "memory");
        }
        __threadfence_block();
        S.ready[k] = 1;
      }
#if SM_MBAR
      while (ld_acquire(p.ctl + 64) < (gen + 1) * G) { if (SM_SLEEP) __nanosleep(32); }
#else
      while (ld_acquire(p.ctl + 65) == gen) { }
#endif
      DBG(5);
    }
    __syncwarp();
  } else if (warp == W_EPI) {
    // ---------------------------------------------------------------- epilogue warp
    asm volatile("bar.sync 1, %0;\n" ::"n"(NTH) : "memory");
    const int U_A = S.U_A, U = S.U, gen = S.gen;
    float* ss = p.ss + (gen & 1) * 64;
    int* cnt = p.ctl + (gen & 1) * 32;
    const int g = lane >> 2, t = lane & 3;
    Seq sq; sq.init(cta, G, U_A, U);
    bool nwok = false;
    int segcnt = 0;
    for (int i = 0; i < sq.n; ++i) {
      const int pos = sq.at(i);
      int seg, rb; decode(pos, U_A, seg, rb);
      const int k = seg & 31;
      const int b = i % NRB;
      mbar_wait_slow(&S.rfull[b], (uint32_t)((i / NRB) & 1));
      float4 s = S.red[b][0][lane];
#pragma unroll
      for (int w = 1; w < NWC; ++w) { const float4 v = S.red[b][w][lane]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.rempty[b]);
      const int n = S.nk[k], pk = S.pk[k];
      if (seg < 32) {
        const float gsc = S.gA[k] * TWO62;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 2 * t + h;
          if (tk < n) {
            const float gt = (h ? s.y : s.x) * gsc, up = (h ? s.w : s.z) * gsc;
            p.inter[(size_t)(pk + tk) * HD + rb * 8 + g] = __float2bfloat16(__fdividef(gt, 1.f + __expf(-gt)) * up);
          }
        }
        ++segcnt;
        if (i + 1 == sq.n || seg_of(sq.at(i + 1), U_A) != seg) {  // this CTA's last unit of expert k's phase A
#if SM_CREDW
          __syncwarp();
          if (lane == 0) { S.ccnt[k] = segcnt; mbar_arrive(&S.cbar[k]); }
#else
          __threadfence();
          __syncwarp();
          if (lane == 0) atomicAdd(cnt + k, segcnt);
#endif
          segcnt = 0;
          if (SM_DBG && lane == 0 && (i + 1 == sq.n || sq.at(i + 1) >= U_A)) DBG(3);
        }
      } else {
        if (!nwok) { mbar_wait(&S.nwbar, 0); nwok = true; }
        const float gsc = S.gB[k] * TWO62;
        const int n0 = rb * 16 + g;
        #if SM_NWBF
        const float f0 = gsc * (1.f + __bfloat162float(S.nwb[n0])), f1 = gsc * (1.f + __bfloat162float(S.nwb[n0 + 8]));
#else
        const float f0 = gsc * S.nw1[n0], f1 = gsc * S.nw1[n0 + 8];
#endif
        float sqv[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 2 * t + h;
          const float v0 = (h ? s.y : s.x) * f0, v1 = (h ? s.w : s.z) * f1;
          if (tk < n) {
            p.ybuf[(size_t)(pk + tk) * HD + n0] = v0;
            p.ybuf[(size_t)(pk + tk) * HD + n0 + 8] = v1;
          }
          sqv[h] = tk < n ? v0 * v0 + v1 * v1 : 0.f;
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          sqv[h] += __shfl_xor_sync(0xffffffffu, sqv[h], 4);
          sqv[h] += __shfl_xor_sync(0xffffffffu, sqv[h], 8);
          sqv[h] += __shfl_xor_sync(0xffffffffu, sqv[h], 16);
        }
        if (g == 0) {
#pragma unroll
          for (int h = 0; h < 2; ++h) if (2 * t + h < n) atomicAdd(ss + pk + 2 * t + h, sqv[h]);
        }
      }
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) {
      DBG(4);
#if SM_MBAR
      // monotonic: the count reaches (gen + 1) G at this launch's end; the last arriver bumps gen for the next launch
      const int old = atomicAdd(p.ctl + 64, 1);
      if (old == (gen + 1) * G - 1) atomicExch(p.ctl + 65, gen + 1);
#else
      const int old = atomicAdd(p.ctl + 64, 1);
      if (old == G - 1) {
        p.ctl[64] = 0;
        __threadfence();
        atomicExch(p.ctl + 65, gen + 1);
      }
#endif
    }
    __syncwarp();
  } else if (warp >= NWC + 3) {
    // ---------------------------------------------------------------- credit warp: phase-A credits (fence + atomic)
    asm volatile("bar.sync 1, %0;\n" ::"n"(NTH) : "memory");
    if (lane == 0) {
      const int D = S.D;
      int* cnt = p.ctl + (S.gen & 1) * 32;
      Seq sq; sq.init(cta, G, S.U_A, S.U);
      int kmin = sq.n > 0 ? seg_of(sq.at(0), S.U_A) : 0;   // this CTA's A segments are kmin .. (last A seg)
      for (int k = kmin; k < D; ++k) {
        bool has = false;   // does this CTA have a unit of A segment k?
        for (int i = 0; i < sq.nA && i < sq.n; ++i) { const int pp = sq.at(i); if (pp < S.U_A && (pp >> 9) == k) { has = true; break; } if (pp >= 512 * (k + 1)) break; }
        if (!has) continue;
        mbar_wait_slow(&S.cbar[k], 0u);
        __threadfence();
        atomicAdd(cnt + k, S.ccnt[k]);
      }
    }
    __syncwarp();
  } else {
    // ---------------------------------------------------------------- compute warps 0..7 (each its own 512-K slice)
    asm volatile("bar.sync 1, %0;\n" ::"n"(NTH) : "memory");
    if (SM_DBG && tid == 0) DBG(2);
    const int U_A = S.U_A, U = S.U;
    Seq sq; sq.init(cta, G, U_A, U);
    const uint32_t k16 = 16u + p.zero;
    uint4 xc[2 * NCH];
    int curseg = -1;
    unsigned long long tw0 = 0, twait = 0;
    bool bst = false;
    for (int i = 0; i < sq.n; ++i) {
      const int pos = sq.at(i);
      int seg, rb; decode(pos, U_A, seg, rb);
      if (seg != curseg) {
        if (SM_DBG && tid == 0) tw0 = gtimer();
        load_x(p, S, seg, warp, lane, xc);
        curseg = seg;
        if (SM_DBG && tid == 0) { twait += gtimer() - tw0; if (seg >= 32 && !bst) { bst = true; DBG(9); } }
      }
      const int sl = i % NS;
      mbar_wait(&S.full[sl], (uint32_t)((i / NS) & 1));
      if (SM_DBG && tid == 0 && i == 0) DBG(6);
      const unsigned char* slot = S.ring[sl];
#if SM_NWC == 16
      const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UCODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
      const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UCODES + warp * 512 + lane * 16);
#endif
      float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
      float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < (SM_NOCMP ? 0 : NCH); ++c) {
        const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
        const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
        const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        const uint32_t b[8] = {xc[2 * c].x, xc[2 * c].y, xc[2 * c].z, xc[2 * c].w,
                               xc[2 * c + 1].x, xc[2 * c + 1].y, xc[2 * c + 1].z, xc[2 * c + 1].w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          float* ac = SM_NACC == 4 ? (st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4) : ((st & 1) ? acc2 : acc);
          mma_bf16(ac, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
        }
      }
      const int rbf = i % NRB;
      if (i >= NRB) mbar_wait(&S.rempty[rbf], (uint32_t)(((i / NRB) - 1) & 1));
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[q] += acc2[q] + acc3[q] + acc4[q];
      S.red[rbf][warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
      __syncwarp();
      if (lane == 0) { mbar_arrive(&S.empty[sl]); mbar_arrive(&S.rfull[rbf]); }
    }
    if (SM_DBG && tid == 0 && p.dbg) { p.dbg[cta * 16 + 8] = twait; DBG(7); }
  }

  // ------------------------------------------------------------------ grid released: norm + weighted combine
  asm volatile("bar.sync 2, %0;\n" ::"n"(NTH) : "memory");
  {
    const float* ss = p.ss + (S.gen & 1) * 64;
    const int Q = p.M * 1024;
    for (int q = cta + G * tid; q < Q; q += G * NTH) {
      const int t = q >> 10, r4 = (q & 1023) << 2;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int pr = S.pos[t * 8 + s];
        if (pr >= 0) {
          const float f = S.twv[t * 8 + s] * rsqrtf(__ldcg(ss + pr) * (1.f / HD) + 1e-8f);
          const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)pr * HD + r4));
          a0 += f * v.x; a1 += f * v.y; a2 += f * v.z; a3 += f * v.w;
        }
      }
      __nv_bfloat162 lo = __floats2bfloat162_rn(a0, a1), hi = __floats2bfloat162_rn(a2, a3);
      uint2 o; o.x = *reinterpret_cast<uint32_t*>(&lo); o.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(p.out + (size_t)t * HD + r4) = o;
    }
  }
  if (SM_DBG && tid == W_POLL * 32) DBG(10);
  if (SM_DBG && tid == 0 && p.dbg) {
    unsigned int smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); p.dbg[cta * 16 + 15] = smid;
  }
}
}  // namespace sm3

void sm3_run(torch::Tensor wAu, torch::Tensor wBu, torch::Tensor gsA, torch::Tensor gsB, torch::Tensor normw,
             torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out,
             torch::Tensor ctl, torch::Tensor fbuf, torch::Tensor inter, int64_t lo, int64_t nloc, int64_t grid,
             torch::Tensor dbg) {
  sm3::Params p;
  p.wAu = (const unsigned char*)wAu.data_ptr(); p.wBu = (const unsigned char*)wBu.data_ptr();
  p.gsA = (const float*)gsA.data_ptr(); p.gsB = (const float*)gsB.data_ptr();
  p.normw = (const __nv_bfloat16*)normw.data_ptr();
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.ids = (const int*)ids.data_ptr(); p.tw = (const float*)tw.data_ptr();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.inter = (__nv_bfloat16*)inter.data_ptr();
  p.ss = (float*)fbuf.data_ptr(); p.ybuf = p.ss + 128;
  p.ctl = (int*)ctl.data_ptr();
  p.dbg = dbg.numel() > 0 ? (unsigned long long*)dbg.data_ptr() : nullptr;
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc; p.zero = 0;
  TORCH_CHECK(p.M >= 1 && p.M <= 8 && p.M <= SM_SROWS && nloc <= 32, "sm3: M <= min(8, SROWS), nloc <= 32");
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sm3::moe_small, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(sm3::Smem));
    attr = true;
  }
  sm3::moe_small<<<(int)grid, sm3::NTH, sizeof(sm3::Smem), at::cuda::getCurrentCUDAStream()>>>(p);
}

std::vector<int64_t> sm3_attrs() {
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, sm3::moe_small);
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)sizeof(sm3::Smem)};
}
'''

SMALL_CPP = r'''
#include <torch/extension.h>
void sm3_run(torch::Tensor wAu, torch::Tensor wBu, torch::Tensor gsA, torch::Tensor gsB, torch::Tensor normw,
             torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out,
             torch::Tensor ctl, torch::Tensor fbuf, torch::Tensor inter, int64_t lo, int64_t nloc, int64_t grid,
             torch::Tensor dbg);
std::vector<int64_t> sm3_attrs();
'''

_SMALL_EXTS = {}


def _small_ext(defines=()):
    """Compile (once per define tuple) the small-batch kernel."""
    defines = tuple(defines)
    if defines not in _SMALL_EXTS:
        import hashlib
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = hashlib.sha1((SMALL_SRC + "|".join(defines)).encode()).hexdigest()[:12]
        _SMALL_EXTS[defines] = load_inline(
            name=f"okc_moe_small_v3_{tag}_sm{arch}", cpp_sources=SMALL_CPP, cuda_sources=SMALL_SRC,
            functions=["sm3_run", "sm3_attrs"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo", *defines],
            verbose=False)
    return _SMALL_EXTS[defines]


def _small_state(dev):
    """Scratch of the small kernel: ctl int32 [0..63] two copies of the 32 per-expert phase-A credits, [64] grid
    barrier count, [65] barrier generation; fbuf fp32: two copies of 64 sums of squares, then ybuf [64, 4096];
    inter bf16 [64, 4096]. Zeroed once here; the kernel re-zeroes the copy the next launch uses."""
    return {"ctl": torch.zeros(128, dtype=torch.int32, device=dev),
            "fbuf": torch.zeros(128 + 64 * HIDDEN, dtype=torch.float32, device=dev),
            "inter": torch.zeros(64 * HIDDEN, dtype=torch.bfloat16, device=dev),
            "dbg": torch.zeros(0, dtype=torch.int64, device=dev)}


def _small_launch(state, data, out, defines=None, scr=None):
    v3 = state["v3"]
    ext = _small_ext(v3["defines"] if defines is None else defines)
    scr = scr or v3["scr"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    w = state["w"]
    ext.sm3_run(w[10], w[11], w[2], w[5], w[7], x, ids, tw, out, scr["ctl"], scr["fbuf"], scr["inter"],
                state["lo"], state["cnt"], state["grid"], scr["dbg"])
    return out


_v3_parent_prepare = prepare


def prepare(weights):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_small_ext, SMALL_DEFINES)   # compiles while the parent prepares
        state = _v3_parent_prepare(weights)
        try:
            fut.result()
            ok = ("-DOKC_PRMTSC=1" in tuple(state.get("sfx", ()))) and not SC6_SKINNY and state["cnt"] <= 32
        except Exception as ex:
            print(f"[v3] small build failed: {str(ex)[-400:]}", flush=True)
            ok = False
    if ok:
        dev = weights["w13"].device
        state["v3"] = {"defines": tuple(SMALL_DEFINES), "scr": _small_state(dev)}
        print(f"[v3] small kernel on for M <= {SMALL_MAX_M}: regs/local/smem {_small_ext(SMALL_DEFINES).sm3_attrs()}", flush=True)
    return state


_v3_parent_kernel = kernel_fn


def kernel_fn(data, state):
    M = data["hidden_states"].shape[0]
    if state.get("v3") is None or M > SMALL_MAX_M:
        return _v3_parent_kernel(data, state)
    _bind_context(data["hidden_states"])
    assert data["config"]["local_start"] == state["lo"] and data["config"]["local_count"] == state["cnt"]
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=data["hidden_states"].device)
    return _small_launch(state, data, out)
