// Single-token routed experts without a gate/up -> down dependency between CTAs.
// The K dimension of the down projection is split into 16-wide chunks c: a unit (expert e, chunk c) computes
// inter[16c .. 16c+15] from gate/up rows (two 16-row blocks of the unit-order copy, 36 KB each) and immediately
// multiplies it into W2[:, 16c .. 16c+15] (one 36 KB chunk of the chunk-major down copy). Partial y rows accumulate
// in shared memory across a CTA's units of one expert and are flushed with vector float atomics; one grid barrier
// then precedes the post-expert RMSNorm and the weighted sum. Decode and scale math are those of the main kernel
// (dq4 DQ7 layout, e4m3 -> bf16x2(scale * 2^64) LUT, results * gs * 2^62).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace sk1 {
#ifndef SK_NW
#define SK_NW 8
#endif
constexpr int HD = 4096, NW = SK_NW, NT = NW * 32, NTL = NT + 32;
constexpr int CPW = 64 / NW;   // gate/up: 64-K chunks per warp (8 warps x 8 chunks in the stored layout)
constexpr int RBW = 256 / NW;  // down: 16-row blocks per warp
constexpr int STAGE = 36864, CODES = 32768;
constexpr float TWO62 = 4.611686018427388e18f;
#ifndef SK_PROF
#define SK_PROF 0
#endif
#ifndef SK_NS
#define SK_NS 4
#endif
constexpr int NS = SK_NS;

struct Params {
  const unsigned char* wAu; const unsigned char* wBc;
  const float* gsA; const float* gsB; const uint32_t* lut; const __nv_bfloat16* normw;
  const __nv_bfloat16* x; const int* ids; const float* tw;
  int lo, nloc;
  int* ctr;      // [0] generation, [1..2] grid barrier counters by generation parity
  float* ybuf;   // [2 generations][8 route slots][HD]
  __nv_bfloat16* out;
  long long* prof;  // SK_PROF: per CTA [wait_A, comp_A, wait_B, comp_B, total, stages, t0, t_loop, t_bar, t_end] (t*: globaltimer ns)
};

struct Smem {
  uint64_t full[NS], empty[NS];
  float4 red[2][NW][32];
  float inter[16];
  float rs[8];
  float wsum[NW];
  int E, eid[8];
  float ew[8];
  int u0, u1, gen;
  uint32_t lut[256];
  float yacc[HD];
  __nv_bfloat16 xs[HD];
};

__device__ __forceinline__ uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mwait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n .reg .pred q;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n @!q bra W_%=;\n}\n"
               ::"r"(su(b)), "r"(ph) : "memory");
}
__device__ __forceinline__ void marrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(su(b)) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#if SK_PROF
#define STAMP(k) do { if (threadIdx.x == 0 && p.prof) p.prof[blockIdx.x * 10 + 6 + (k)] = gt(); } while (0)
#else
#define STAMP(k) do {} while (0)
#endif
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory"); }
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// the main kernel's DQ7 decode: 8 E2M1 codes -> 4 bf16x2 A-fragment registers (values * 2^-126)
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
}

__device__ __forceinline__ void flush(const Params& p, Smem& S, float* ycur, int k, int tid) {
  cbar();
  const float gsc = __ldg(p.gsB + S.eid[k]) * TWO62;
  float* dst = ycur + (size_t)k * HD;
  for (int i = tid * 4; i < HD; i += NT * 4) {
    float4 v = *reinterpret_cast<float4*>(&S.yacc[i]);
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n"
                 ::"l"(dst + i), "f"(v.x * gsc), "f"(v.y * gsc), "f"(v.z * gsc), "f"(v.w * gsc) : "memory");
  }
  cbar();
}

__global__ void __launch_bounds__(NTL, 1) sk1_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  unsigned char* ring = smraw + ((sizeof(Smem) + 127) / 128) * 128;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x;
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  STAMP(0);
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(su(&S.full[s])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(su(&S.empty[s])), "r"(NW));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    int E = 0;
    for (int s = 0; s < 8; ++s) {
      const int e = __ldg(p.ids + s) - p.lo;
      if (e >= 0 && e < p.nloc) { S.eid[E] = e; S.ew[E] = __ldg(p.tw + s); ++E; }
    }
    S.E = E;
    const int U = E * 256;
    S.u0 = (int)((long long)U * blockIdx.x / G);
    S.u1 = (int)((long long)U * (blockIdx.x + 1) / G);
    S.gen = ld_acquire(p.ctr);
  }
  for (int i = tid; i < 256; i += NTL) S.lut[i] = __ldg(p.lut + i);
  for (int i = tid; i < HD; i += NTL) S.yacc[i] = 0.f;
  for (int i = tid; i < HD / 8; i += NTL) reinterpret_cast<uint4*>(S.xs)[i] = __ldg(reinterpret_cast<const uint4*>(p.x) + i);
  __syncthreads();
  const int gen = S.gen;
  {  // the next launch's accumulators and barrier counter (the previous launch has finished with them)
    float* yn = p.ybuf + (size_t)((gen + 1) & 1) * 8 * HD;
    for (int i = blockIdx.x * NTL + tid; i < 8 * HD; i += G * NTL) yn[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[1 + ((gen + 1) & 1)] = 0;
  }
  float* ycur = p.ybuf + (size_t)(gen & 1) * 8 * HD;
  const int u0 = S.u0, u1 = S.u1, nst = 3 * (u1 - u0);
  STAMP(1);
  if (warp == NW) {  // producer warp: one 36 KB bulk copy per stage
    if (lane == 0) {
      for (int i = 0; i < nst; ++i) {
        const int s = i % NS;
        if (i >= NS) mwait(&S.empty[s], (uint32_t)((i / NS) - 1) & 1u);
        const int u = u0 + i / 3, part = i % 3, k = u >> 8, c = u & 255, e = S.eid[k];
        const unsigned char* src = part < 2 ? p.wAu + ((size_t)e * 512 + 2 * c + part) * STAGE
                                            : p.wBc + ((size_t)e * 256 + c) * STAGE;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(su(&S.full[s])), "r"(STAGE) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     ::"r"(su(ring + s * STAGE)), "l"(src), "r"(STAGE), "r"(su(&S.full[s])) : "memory");
      }
    }
  } else {
    const int g = lane >> 2, t = lane & 3;
    int curk = -1;
    float yr[RBW][2];  // this lane's down rows (16rb + g, 16rb + g + 8) for its warp's 32 row blocks; column 0 lanes (t == 0) carry the token
#pragma unroll
    for (int r = 0; r < RBW; ++r) { yr[r][0] = 0.f; yr[r][1] = 0.f; }
#define SK1_STAGE_Y()                                                                         \
    do {                                                                                        \
      if (t == 0) {                                                                             \
        _Pragma("unroll") for (int r = 0; r < RBW; ++r) {                                       \
          const int rb = warp * RBW + r;                                                         \
          S.yacc[rb * 16 + g] = yr[r][0]; S.yacc[rb * 16 + g + 8] = yr[r][1];                   \
        }                                                                                       \
      }                                                                                         \
      _Pragma("unroll") for (int r = 0; r < RBW; ++r) { yr[r][0] = 0.f; yr[r][1] = 0.f; }        \
    } while (0)
#if SK_PROF
    long long pw[4] = {0, 0, 0, 0};
    const long long pt0 = clock64();
#endif
    for (int i = 0; i < nst; ++i) {
      const int s = i % NS, u = u0 + i / 3, part = i % 3, k = u >> 8;
      if (part == 0 && k != curk) { if (curk >= 0) { SK1_STAGE_Y(); flush(p, S, ycur, curk, tid); } curk = k; }
#if SK_PROF
      long long tw0 = clock64();
#endif
      mwait(&S.full[s], (uint32_t)(i / NS) & 1u);
#if SK_PROF
      long long tw1 = clock64();
#endif
      const unsigned char* slot = ring + s * STAGE;
      if (part < 2) {  // gate/up block: 8 gate rows (g) and 8 up rows (g + 8), this warp's K slice of 512
        const int w8 = warp * CPW / 8, cb = (warp * CPW) % 8;
        const uint4 sc = *reinterpret_cast<const uint4*>(slot + CODES + w8 * 512 + lane * 16);
        float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int cc = 0; cc < CPW; ++cc) {
          const int c = cb + cc;
          const uint4 w = *reinterpret_cast<const uint4*>(slot + (w8 * 8 + c) * 512 + lane * 16);
          const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
          const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
          const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
          uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;  // B column 0 is the token; the other seven columns are zero
          if (g == 0) {
            const uint4* xp = reinterpret_cast<const uint4*>(S.xs + w8 * 512 + c * 64 + 16 * t);
            x0 = xp[0]; x1 = xp[1];
          }
          const uint32_t b[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
          for (int st = 0; st < 4; ++st) {
            uint32_t r0, r1, r2, r3;
            dq4(ww[st], r0, r1, r2, r3);
            mma_bf16((st & 1) ? acc2 : acc, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
          }
        }
        S.red[part][warp][lane] = make_float4(acc[0] + acc2[0], acc[1] + acc2[1], acc[2] + acc2[2], acc[3] + acc2[3]);
        __syncwarp();
        if (lane == 0) marrive(&S.empty[s]);
        cbar();
        if (tid < 8) {
          float gt = 0.f, up = 0.f;
#pragma unroll
          for (int w2 = 0; w2 < NW; ++w2) { const float4 q = S.red[part][w2][tid * 4]; gt += q.x; up += q.z; }
          const float gsc = __ldg(p.gsA + S.eid[k]) * TWO62;
          gt *= gsc; up *= gsc;
          S.inter[part * 8 + tid] = __fdividef(gt, 1.f + __expf(-gt)) * up;
        }
        if (part == 1) cbar();  // inter[0..15] complete before the down chunk
      } else {  // down chunk: rows 16rb .. 16rb+15 for this warp's 32 row blocks, K = inter[0..15]
        uint32_t b0 = 0, b1 = 0;
        if (g == 0) { b0 = pack2(S.inter[4 * t], S.inter[4 * t + 1]); b1 = pack2(S.inter[4 * t + 2], S.inter[4 * t + 3]); }
#pragma unroll
        for (int r0 = 0; r0 < RBW; r0 += 8) {  // 8 row blocks per batch: loads first, then independent MMAs
          uint32_t qv[8], sv[8];
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int rb = warp * RBW + r0 + r;
            qv[r] = *reinterpret_cast<const uint32_t*>(slot + (rb * 32 + lane) * 4);
            sv[r] = *reinterpret_cast<const uint16_t*>(slot + CODES + rb * 16 + g * 2);
          }
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const uint32_t sg = S.lut[sv[r] & 0xff], sg8 = S.lut[sv[r] >> 8];
            uint32_t a0, a1, a2, a3;
            dq4(qv[r], a0, a1, a2, a3);
            float a[4] = {yr[r0 + r][0], 0.f, yr[r0 + r][1], 0.f};
            mma_bf16(a, bmul2(a0, sg), bmul2(a1, sg8), bmul2(a2, sg), bmul2(a3, sg8), b0, b1);
            yr[r0 + r][0] = a[0]; yr[r0 + r][1] = a[2];
          }
        }
        __syncwarp();
        if (lane == 0) marrive(&S.empty[s]);
      }
#if SK_PROF
      if (tid == 0) { const int b = part < 2 ? 0 : 2; pw[b] += tw1 - tw0; pw[b + 1] += clock64() - tw1; }
#endif
    }
    if (curk >= 0) { SK1_STAGE_Y(); flush(p, S, ycur, curk, tid); }
#if SK_PROF
    if (tid == 0) { long long* q = p.prof + blockIdx.x * 10; q[0] = pw[0]; q[1] = pw[1]; q[2] = pw[2]; q[3] = pw[3]; q[4] = clock64() - pt0; q[5] = nst; }
#endif
  }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (tid == 0) {  // grid barrier; the last arriver advances the generation (no CTA of this launch reads it again)
    int* bc = p.ctr + 1 + (gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == G - 1) atomicAdd(p.ctr, 1);
    while (ld_acquire(bc) < G) __nanosleep(64);
  }
  __syncthreads();
  STAMP(2);
  constexpr int COLS = 32;
  if (blockIdx.x * COLS >= HD) return;
  const int E = S.E;
  for (int k = 0; k < E; ++k) {  // rms of y * (1 + w) over all 4096 rows (every finishing CTA, from L2)
    const float* y = ycur + (size_t)k * HD;
    float ss = 0.f;
    for (int i = tid; i < HD; i += NTL) {
      const float v = __ldcg(y + i) * (1.f + __bfloat162float(p.normw[i]));
      ss += v * v;
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    if (lane == 0) S.wsum[0] = 0.f;
    __syncthreads();
    if (lane == 0) atomicAdd(&S.wsum[0], ss);
    __syncthreads();
    if (tid == 0) S.rs[k] = rsqrtf(S.wsum[0] / (float)HD + 1e-8f);
    __syncthreads();
  }
  if (tid < COLS) {
    const int r = blockIdx.x * COLS + tid;
    const float nw = 1.f + __bfloat162float(p.normw[r]);
    float o = 0.f;
    for (int k = 0; k < E; ++k) o += S.ew[k] * S.rs[k] * __ldcg(ycur + (size_t)k * HD + r) * nw;
    p.out[r] = __float2bfloat16(o);
  }
  STAMP(3);
}
}  // namespace sk1

void sk1_run(std::vector<torch::Tensor> w, torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out,
             torch::Tensor ctr, torch::Tensor ybuf, int64_t lo, int64_t nloc, int64_t grid, int64_t pdl) {
  sk1::Params p;
  p.wAu = (const unsigned char*)w[0].data_ptr(); p.wBc = (const unsigned char*)w[1].data_ptr();
  p.gsA = (const float*)w[2].data_ptr(); p.gsB = (const float*)w[3].data_ptr();
  p.lut = (const uint32_t*)w[4].data_ptr(); p.normw = (const __nv_bfloat16*)w[5].data_ptr();
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.ids = (const int*)ids.data_ptr(); p.tw = (const float*)tw.data_ptr();
  p.lo = (int)lo; p.nloc = (int)nloc;
  p.ctr = (int*)ctr.data_ptr(); p.ybuf = (float*)ybuf.data_ptr();
  p.prof = w.size() > 6 ? (long long*)w[6].data_ptr() : nullptr;
  p.out = (__nv_bfloat16*)out.data_ptr();
  const size_t sm = ((sizeof(sk1::Smem) + 127) / 128) * 128 + (size_t)sk1::NS * sk1::STAGE;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sk1::sk1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    attr = true;
  }
  auto st = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid); cfg.blockDim = dim3(sk1::NTL); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, sk1::sk1_kernel, p);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "sk1 launch: ", cudaGetErrorString(e));
}
