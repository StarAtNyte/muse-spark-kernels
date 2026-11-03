

# =====================================================================================================================
# SMOE: small-batch (M <= 8) routed-expert kernel with dynamic unit claiming (StarAtNyte, Apache-2.0).
#
# One persistent CTA per SM: 8 consumer warps + 1 TMA producer warp. Every expert's work is cut into equal 36,864-byte
# units (16 weight rows x full K, codes + e4m3 block scales, the same unit layout and FP4 decode as the skinny path
# above): 512 gate/up units (8 gate + 8 up rows, SwiGLU in-register) and 256 down units per active expert. The
# producer claims units from one global counter in the order [all gate/up units, expert by expert][all down units,
# expert by expert], so faster SMs take more units and no CTA ends far behind the others. A down unit waits until its
# expert's 512 gate/up units are done (per-expert counter); with the claim order that wait is almost never taken, and
# the ring keeps loading while it is. The eight consumer warps split K; their FP32 partials are summed in shared
# memory (all partial sums stay FP32). Gate/up results go to a BF16 intermediate (as SGLang keeps it); down results
# go to an FP32 buffer with the per-route sum of squares; a grid barrier then forms the RMSNorm-weighted sum.
# =====================================================================================================================
SMOE_ON = True
SMOE_MAX_M = 8
SMOE_NS = 6
SMOE_SPCT = 80

SMOE_CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace smoe {
#ifndef SMOE_X
#define SMOE_X 0
#endif
#ifndef SMOE_NCP
#define SMOE_NCP 1
#endif
#ifndef SMOE_SPCT
#define SMOE_SPCT 80
#endif
#ifndef SMOE_NS
#define SMOE_NS 6
#endif
constexpr int NCW = 8, NT = NCW * 32 + 32;
constexpr int UB = 36864, UC = 32768;
constexpr int NS = SMOE_NS;
constexpr int NAU = 512, NBU = 256, NU = 768;
constexpr int HID = 4096;

struct Params {
  const uint8_t* w;            // [nloc][768][UB]
  const float* gsA;            // [nloc]
  const float* gsB;            // [nloc]
  const __nv_bfloat16* pnw;    // [4096]
  const __nv_bfloat16* x;      // [M][4096]
  const int* ids;              // [M][8]
  const float* tw;             // [M][8]
  __nv_bfloat16* hbuf;         // [nloc][8][4096]
  float* ybuf;                 // [nloc][8][4096]
  float* ss;                   // [2][nloc][8]
  int* ctr;                    // [0] claim counter, [1] grid barrier, [2 + e] gate/up units done
  __nv_bfloat16* out;          // [M][4096]
  int M, lo, nloc;
};

struct __align__(128) Smem {
  uint8_t ring[NS][UB];
  unsigned long long full[NS], empty[NS], redb[NS];
  int unit[NS];
  int act[32];
  int nact, L;
  float rs[64], rw[64];
  int re[64];
  int flag[32];
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(unsigned long long* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(su32(b)), "r"(n));
}
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mb_arrive_tx(unsigned long long* b, uint32_t tx) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(su32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned long long* b, uint32_t par) {
  uint32_t a = su32(b);
  asm volatile(
      "{\n.reg .pred p;\nW%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W%=;\n}\n" ::"r"(a), "r"(par)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, unsigned long long* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(b)) : "memory");
}
__device__ __forceinline__ int ld_acq(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 8 E2M1 codes (DQ7 bit layout of _pack_codes) -> 4 bf16x2 A-fragment registers holding e2m1 * 2^-126
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
}
// e4m3 byte b of word -> bf16x2(scale * 2^64), both halves (exact for positive normal bytes; prepare() checks)
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)

__device__ __forceinline__ void load_frag(uint32_t* fr, const __nv_bfloat16* src, int M, int warp, int lane, bool cg) {
  const int g = lane >> 2, t = lane & 3;
  if (g < M) {
    const uint4* p = reinterpret_cast<const uint4*>(src + (size_t)g * HID + warp * 512 + t * 16);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint4 a = cg ? __ldcg(p + c * 8) : __ldg(p + c * 8);
      uint4 b = cg ? __ldcg(p + c * 8 + 1) : __ldg(p + c * 8 + 1);
      fr[c * 8 + 0] = a.x; fr[c * 8 + 1] = a.y; fr[c * 8 + 2] = a.z; fr[c * 8 + 3] = a.w;
      fr[c * 8 + 4] = b.x; fr[c * 8 + 5] = b.y; fr[c * 8 + 6] = b.z; fr[c * 8 + 7] = b.w;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 64; ++i) fr[i] = 0u;
  }
}

__global__ void __launch_bounds__(NT, 1) smoe_kernel(const Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x;
  const int M = p.M;
  int* claim = p.ctr;
  unsigned* gbar = reinterpret_cast<unsigned*>(p.ctr + 1);
  int* cntA = p.ctr + 2;

  if (tid < 32) S.flag[tid] = 0;
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mb_init(&S.full[i], 1); mb_init(&S.empty[i], NCW); mb_init(&S.redb[i], NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.L = (int)(*reinterpret_cast<volatile unsigned*>(gbar) / (unsigned)G);
  }
  __syncthreads();
  // routing: which local experts have tokens (every CTA computes the same list)
  if (tid < M * 8) {
    const int e = p.ids[tid] - p.lo;
    if (e >= 0 && e < p.nloc) S.flag[e] = 1;
  }
  __syncthreads();
  if (warp == 0) {
    const unsigned m = __ballot_sync(0xffffffffu, lane < p.nloc && S.flag[lane]);
    if ((m >> lane) & 1u) S.act[__popc(m & ((1u << lane) - 1u))] = lane;
    if (lane == 0) S.nact = __popc(m);
  }
  const int par = S.L & 1;
  if (blockIdx.x == 0) {  // zero the other launch parity's sums of squares (the previous launch has finished)
    for (int i = tid; i < p.nloc * 8; i += NT) p.ss[(par ^ 1) * p.nloc * 8 + i] = 0.f;
  }
  __syncthreads();
  const int nact = S.nact;
  const int NAt = nact * NAU, N = nact * NU;

  if (warp == NCW) {
    // ---------------- producer: claim units, stream them into the ring
    if (lane == 0) {
      // first SPCT% of the sequence: static round-robin (no atomics); the tail: dynamic claims, each issued one unit
      // ahead so its latency overlaps the slot wait
      const int S0 = (N * SMOE_SPCT / 100) / G, D0 = S0 * G;
      int nextj = S0 > 0 ? (int)blockIdx.x : D0 + atomicAdd(claim, 1);
      for (int k = 0;; ++k) {
        const int slot = k % NS;
        if (k >= NS) mb_wait(&S.empty[slot], (uint32_t)(((k / NS) - 1) & 1));
        const int j = nextj;
        if (j >= N) { S.unit[slot] = -1; mb_arrive(&S.full[slot]); break; }
        int e, u;
        if (j < NAt) { e = S.act[j >> 9]; u = j & 511; }
        else { const int jj = j - NAt; e = S.act[jj >> 8]; u = NAU + (jj & 255); }
        S.unit[slot] = (e << 10) | u;
        mb_arrive_tx(&S.full[slot], UB);
#pragma unroll
        for (int q = 0; q < SMOE_NCP; ++q)
          bulk_g2s(S.ring[slot] + q * (UB / SMOE_NCP), p.w + ((size_t)e * NU + u) * UB + q * (UB / SMOE_NCP), UB / SMOE_NCP, &S.full[slot]);
        nextj = (k + 1 < S0) ? (int)blockIdx.x + (k + 1) * G : D0 + atomicAdd(claim, 1);
      }
    }
    __syncwarp();
  } else {
    // ---------------- consumers
    uint32_t fr[64];
    int key = -2;
    const int g = lane >> 2, t = lane & 3;
    for (int k = 0;; ++k) {
      const int slot = k % NS;
      mb_wait(&S.full[slot], (uint32_t)((k / NS) & 1));
      const int code = S.unit[slot];
      if (code < 0) break;
      const int e = code >> 10, u = code & 1023;
      const bool isB = u >= NAU;
      const int nk = isB ? e : -1;
      if (nk != key) {
        if (isB) {
          if (lane == 0) while (ld_acq(cntA + e) < NAU) __nanosleep(64);
          __syncwarp();
          load_frag(fr, p.hbuf + (size_t)e * 8 * HID, M, warp, lane, true);
        } else {
          load_frag(fr, p.x, M, warp, lane, false);
        }
        key = nk;
      }
      const unsigned char* base = S.ring[slot];
      const uint4 sc = *reinterpret_cast<const uint4*>(base + UC + warp * 512 + lane * 16);
      float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < (SMOE_X == 1 ? 0 : 8); ++c) {
        const uint4 w = *reinterpret_cast<const uint4*>(base + (warp * 8 + c) * 512 + lane * 16);
        const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
        const uint32_t sg = SCD(swd, (c & 1) * 2), sg8 = SCD(swd, (c & 1) * 2 + 1);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma((st & 1) ? acc2 : acc, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8),
              fr[c * 8 + 2 * st], fr[c * 8 + 2 * st + 1]);
        }
      }
      // partials into this warp's own (consumed) code region of the slot; the rotating epilogue warp sums them
      __syncwarp();
      *reinterpret_cast<float4*>(S.ring[slot] + warp * 4096 + lane * 16) =
          make_float4(acc[0] + acc2[0], acc[1] + acc2[1], acc[2] + acc2[2], acc[3] + acc2[3]);
      __syncwarp();
      if (lane == 0) mb_arrive(&S.redb[slot]);
      const int ew = k % NCW;
      if (warp != ew) {
        if (lane == 0) mb_arrive(&S.empty[slot]);
      } else {
        mb_wait(&S.redb[slot], (uint32_t)((k / NS) & 1));
        float4 s = *reinterpret_cast<const float4*>(S.ring[slot] + lane * 16);
#pragma unroll
        for (int w2 = 1; w2 < NCW; ++w2) {
          const float4 v = *reinterpret_cast<const float4*>(S.ring[slot] + w2 * 4096 + lane * 16);
          s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive(&S.empty[slot]);
        const int t0 = 2 * t, t1 = 2 * t + 1;
        if (SMOE_X == 2) { if (!isB) { __syncwarp(); if (lane == 0) atomicAdd(cntA + e, 1); } }
        else if (!isB) {
          const float gk = __ldg(p.gsA + e) * 4.611686018427388e18f;  // * 2^62
          const float g0 = s.x * gk, g1 = s.y * gk, u0 = s.z * gk, u1 = s.w * gk;
          const float h0 = g0 / (1.f + __expf(-g0)) * u0, h1 = g1 / (1.f + __expf(-g1)) * u1;
          const int row = u * 8 + g;
          __nv_bfloat16* hb = p.hbuf + (size_t)e * 8 * HID + row;
          hb[(size_t)t0 * HID] = __float2bfloat16(h0);
          hb[(size_t)t1 * HID] = __float2bfloat16(h1);
          __threadfence();
          __syncwarp();
          if (lane == 0) atomicAdd(cntA + e, 1);
        } else {
          const float gk = __ldg(p.gsB + e) * 4.611686018427388e18f;
          const int r0 = (u - NAU) * 16 + g, r1 = r0 + 8;
          const float w0 = 1.f + __bfloat162float(p.pnw[r0]), w1 = 1.f + __bfloat162float(p.pnw[r1]);
          const float y00 = s.x * gk * w0, y01 = s.y * gk * w0, y10 = s.z * gk * w1, y11 = s.w * gk * w1;
          float* yb = p.ybuf + (size_t)e * 8 * HID;
          yb[(size_t)t0 * HID + r0] = y00; yb[(size_t)t1 * HID + r0] = y01;
          yb[(size_t)t0 * HID + r1] = y10; yb[(size_t)t1 * HID + r1] = y11;
          float q0 = y00 * y00 + y10 * y10, q1 = y01 * y01 + y11 * y11;
#pragma unroll
          for (int o = 4; o < 32; o <<= 1) {
            q0 += __shfl_xor_sync(0xffffffffu, q0, o);
            q1 += __shfl_xor_sync(0xffffffffu, q1, o);
          }
          if (g == 0) {
            float* sp = p.ss + (size_t)par * p.nloc * 8 + e * 8;
            if (t0 < M) atomicAdd(sp + t0, q0);
            if (t1 < M) atomicAdd(sp + t1, q1);
          }
        }
      }
    }
  }
  // ---------------- grid barrier, then the RMSNorm-weighted route sum
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const unsigned old = atomicAdd(gbar, 1u);
    const unsigned target = (old / (unsigned)G + 1u) * (unsigned)G;
    while ((unsigned)ld_acq(reinterpret_cast<const int*>(gbar)) < target) __nanosleep(32);
  }
  __syncthreads();
  if (blockIdx.x == 0) {  // every CTA is past its claims and its gate/up waits
    if (tid < p.nloc) cntA[tid] = 0;
    if (tid == 32) *claim = 0;
  }
  if (tid < M * 8) {
    const int e = p.ids[tid] - p.lo, tk = tid >> 3;
    if (e >= 0 && e < p.nloc) {
      S.re[tid] = e;
      S.rs[tid] = rsqrtf(__ldcg(p.ss + (size_t)par * p.nloc * 8 + e * 8 + tk) * (1.f / HID) + 1e-8f);
      S.rw[tid] = p.tw[tid];
    } else {
      S.re[tid] = -1;
    }
  }
  __syncthreads();
  const int NG = M * (HID / 4);
  for (int gi = blockIdx.x * NT + tid; gi < NG; gi += G * NT) {
    const int tk = gi / (HID / 4), d = (gi % (HID / 4)) * 4;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int e = S.re[tk * 8 + s];
      if (e >= 0) {
        const float f = S.rw[tk * 8 + s] * S.rs[tk * 8 + s];
        const float4 y = __ldcg(reinterpret_cast<const float4*>(p.ybuf + ((size_t)e * 8 + tk) * HID + d));
        a.x += f * y.x; a.y += f * y.y; a.z += f * y.z; a.w += f * y.w;
      }
    }
    __nv_bfloat162 lo2 = __floats2bfloat162_rn(a.x, a.y), hi2 = __floats2bfloat162_rn(a.z, a.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&lo2);
    o.y = *reinterpret_cast<uint32_t*>(&hi2);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HID + d) = o;
  }
}
}  // namespace smoe

void smoe_run(torch::Tensor w, torch::Tensor gsA, torch::Tensor gsB, torch::Tensor pnw, torch::Tensor x,
              torch::Tensor ids, torch::Tensor tw, torch::Tensor hbuf, torch::Tensor ybuf, torch::Tensor ss,
              torch::Tensor ctr, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  smoe::Params p;
  p.w = w.data_ptr<uint8_t>();
  p.gsA = gsA.data_ptr<float>();
  p.gsB = gsB.data_ptr<float>();
  p.pnw = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ids = ids.data_ptr<int>();
  p.tw = tw.data_ptr<float>();
  p.hbuf = reinterpret_cast<__nv_bfloat16*>(hbuf.data_ptr());
  p.ybuf = ybuf.data_ptr<float>();
  p.ss = ss.data_ptr<float>();
  p.ctr = ctr.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.M = (int)x.size(0);
  p.lo = (int)lo;
  p.nloc = (int)nloc;
  static bool attr = false;
  const int smem = (int)sizeof(smoe::Smem);
  if (!attr) {
    cudaFuncSetAttribute(smoe::smoe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  smoe::smoe_kernel<<<(int)grid, smoe::NT, smem, at::cuda::getCurrentCUDAStream()>>>(p);
}
'''

SMOE_CPP = r'''
#include <torch/extension.h>
void smoe_run(torch::Tensor w, torch::Tensor gsA, torch::Tensor gsB, torch::Tensor pnw, torch::Tensor x,
              torch::Tensor ids, torch::Tensor tw, torch::Tensor hbuf, torch::Tensor ybuf, torch::Tensor ss,
              torch::Tensor ctr, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
'''

_SMOE_EXT = {}


def _smoe_ext(ns=None, spct=None, x=0, ncp=1):
    ns = SMOE_NS if ns is None else ns
    spct = SMOE_SPCT if spct is None else spct
    key = (ns, spct, x, ncp)
    if key not in _SMOE_EXT:
        import hashlib
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        _SMOE_EXT[key] = load_inline(
            name=f"okc_smoe_{hashlib.sha1(SMOE_CUDA.encode()).hexdigest()[:10]}_ns{ns}_sp{spct}_x{x}_c{ncp}_sm{arch}",
            cpp_sources=SMOE_CPP, cuda_sources=SMOE_CUDA, functions=["smoe_run"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo", f"-DSMOE_NS={ns}", f"-DSMOE_SPCT={spct}", f"-DSMOE_X={x}", f"-DSMOE_NCP={ncp}"],
            verbose=False)
    return _SMOE_EXT[key]


def _smoe_prepare(weights, lo, cnt, dev):
    """Unit-major copy of the local experts (bit/byte permutation of the given NVFP4 codes and scales only):
    [cnt][768 units][32,768 code bytes | 4,096 scale bytes]."""
    I = 4096
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(4096, device=dev)
    buf = torch.empty((cnt, 768, 36864), dtype=torch.uint8, device=dev)
    for i, e in enumerate(range(lo, lo + cnt)):
        ca = _pack_codes(weights["w13"][e], rowsA).view(torch.uint8).view(512, 32768)
        sa = _pack_scales(weights["w13_sf"][e], rowsA).view(512, 4096)
        cb = _pack_codes(weights["w2"][e], rowsB).view(torch.uint8).view(256, 32768)
        sb = _pack_scales(weights["w2_sf"][e], rowsB).view(256, 4096)
        buf[i, :512, :32768] = ca
        buf[i, :512, 32768:] = sa
        buf[i, 512:, :32768] = cb
        buf[i, 512:, 32768:] = sb
    return {
        "w": buf,
        "gsA": weights["w13_gs"][lo:lo + cnt].float().contiguous(),
        "gsB": weights["w2_gs"][lo:lo + cnt].float().contiguous(),
        "pnw": weights["post_norm_w"].contiguous(),
        "hbuf": torch.zeros((cnt, 8, 4096), dtype=torch.bfloat16, device=dev),
        "ybuf": torch.zeros((cnt, 8, 4096), dtype=torch.float32, device=dev),
        "ss": torch.zeros((2, cnt, 8), dtype=torch.float32, device=dev),
        "ctr": torch.zeros(2 + 32 + 30, dtype=torch.int32, device=dev),
    }


def _smoe_call(sm, x, ids, tw, lo, cnt, grid, ns=None, spct=None, xx=0, ncp=1):
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    _smoe_ext(ns, spct, xx, ncp).smoe_run(sm["w"], sm["gsA"], sm["gsB"], sm["pnw"], x, ids, tw, sm["hbuf"], sm["ybuf"], sm["ss"],
                           sm["ctr"], out, lo, cnt, grid)
    return out



def _smoe_scales_ok(*sfs):
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True

