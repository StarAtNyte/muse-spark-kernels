

# ======================================================================================================================
# proj_big (team keell0renz, 2026-11-04): m = 128 / 256 BF16 projections on a 12-bit packed weight stream.
# The lossless 12-bit codec (low byte raw + 4-bit sign/exponent-window code, rare out-of-window weights corrected exactly
# in FP32) is vjeux/okc-2026 muse_bf16_proj.py (Apache-2.0); its decode8 is reused verbatim. New here: a warp-specialized
# TMA + register-A wgmma GEMM (weights = wgmma A, decoded straight into registers; x = wgmma B via a 128B-swizzled TMA
# tile), 128 x 128 FP32 accumulator per consumer warpgroup, split-K across the CTAs of one thread-block cluster with the
# partial tiles combined through distributed shared memory (no global workspace, no extra launch).
#   mode 0: CTA = 256 weight rows x 128 tokens (warpgroup = 128 rows each)
#   mode 1: CTA = 128 weight rows x 256 tokens (warpgroup = 128 tokens each, both decode the same rows)
#   mode 2: CTA = 128 weight rows x 128 tokens (warpgroups take alternate 64-k stages, summed in shared memory)
# BFT: split-K partial tiles held in BF16 in shared memory (organizer clarification 2026-11-03/04: BF16 partials and
# BF16-held split combines are allowed within tolerance); the combine itself accumulates in FP32.
# ======================================================================================================================
import hashlib as _pb_hashlib
import os as _pb_os

import torch as _pb_torch


def _pb_wgmma_defs():
    regs = ", ".join(f"%{i}" for i in range(64))
    cons = ", ".join(f'"+f"(d[{i}])' for i in range(64))
    return f"""
__device__ __forceinline__ void wgmma128(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %69, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%64, %65, %66, %67}}, %68, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}
"""


_PB_CUDA = r"""
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace pbk {

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint64_t pol_ef() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ uint64_t pol_el() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d; asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
// vjeux/okc-2026 (Apache-2.0): 8 weights from 8 low bytes + 8 4-bit codes -> 4 bf16x2 A-fragment registers
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
__device__ __forceinline__ float4 ld_dsm_f4(uint32_t a) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint4 ld_dsm_u4(uint32_t a) {
  uint4 v; asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }

@@WGMMA@@

// Weight stream Wst: [NRB][U][R/16 row groups][3 parts][32 lanes][16 B] (vjeux 12-bit fragment order per 64-k unit),
// one contiguous R*96-byte chunk per (R-row block, 64-k unit); R = 64 * NWG.
// CTA (rb, tb, sp): weight rows rb*R .. +R, tokens tb*128 .. +128, 64-k units [sp*U/S, (sp+1)*U/S).
// Warpgroup w (warps 4w..4w+3): rows +64w, one m64n128k16 per k-step, 64 FP32 accumulators per thread.
// Warp 4*NWG: producer (TMA x tile 128 tok x 64 k + bulk copy of the weight chunk, one mbarrier per stage).
// CLX = 2: clusters of two row blocks; each CTA fetches half of the x tile and multicasts it to both.
// CLY = 2: clusters of two token blocks; each CTA fetches half of the weight chunk and multicasts it to both.
// Split-K: partial tile -> shared memory (FP32, escapes corrected exactly there) -> split 0 stores BF16, the other splits
// wait for it (monotonic per-tile counters, all CTAs co-resident) and add theirs with red.global.add.v4.bf16x2.
__device__ __forceinline__ void red_bf16x8(void* p, uint4 v) {
  asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void mbar_arrive_remote(uint32_t bar_cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(bar_cluster_addr) : "memory");
}
__device__ __forceinline__ void tma_2d_mc(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint16_t mask, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5, %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_g2s_mc(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint16_t mask, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1], %2, [%3], %4, %5;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "h"(mask), "l"(pol) : "memory");
}

template <int NWG, int NS, int CLX, int CLY>
__global__ void __launch_bounds__(NWG * 128 + 64, 1)
pb_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Wst, __nv_bfloat16* __restrict__ out,
          const int4* __restrict__ etab, const int* __restrict__ eoff, const __nv_bfloat16* __restrict__ x,
          int* __restrict__ cnt, uint32_t base4, int M, int N, int K, int U, int wkeep, int dbg, int ecap) {
  constexpr int R = 64 * NWG;
  constexpr int NCW = 4 * NWG;                 // consumer warps
  constexpr int NT = NCW * 32 + 64;            // + producer warp + escape warp
  constexpr int CL = CLX * CLY;
  constexpr int WB = R * 96;
  constexpr int XB = 128 * 128;
  constexpr int SB = XB + WB;
  constexpr int RP = R + 4;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  __shared__ int s_target;
  const uint32_t sraw = su32(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  float* tile = reinterpret_cast<float*>(smem_raw + (sbase - sraw));
  // after the ring: escape entries [ecap] int4, then their x columns [ecap][128 tokens] bf16
  int4* esc_ent = reinterpret_cast<int4*>(smem_raw + (sbase - sraw) + NS * SB);
  uint16_t* esc_x = reinterpret_cast<uint16_t*>(esc_ent + ecap);
  const uint32_t full0 = su32(bars), empty0 = full0 + 8 * NS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, tb = blockIdx.y, sp = blockIdx.z, SPLIT = gridDim.z, NRB = gridDim.x;
  const int tileid = tb * NRB + rb;
  const int u0 = (int)((long long)sp * U / SPLIT), u1 = (int)((long long)(sp + 1) * U / SPLIT);
  const int nu = (dbg & 32) ? 0 : u1 - u0;
  const uint32_t cr = CL > 1 ? cluster_rank() : 0u;
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NCW * CL + 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    if (sp > 0) s_target = atomicAdd(cnt + 2 * tileid + 1, 1) / (SPLIT - 1) + 1;   // this call's epoch + 1
  }
  if constexpr (CL > 1) cluster_sync_all(); else __syncthreads();

  float acc[64];
  const int wg = warp >> 2, wq = warp & 3;
  if (warp == NCW) {
    if (lane == 0) {
      const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * WB;
      const uint64_t pe = wkeep ? pol_el() : pol_ef(), pl = pol_el();
      for (int i = 0; i < nu; ++i) {
        const int slot = i % NS;
        if (i >= NS) mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const uint32_t dst = sbase + slot * SB;
        const uint32_t fb = full0 + 8 * slot;
        mbar_expect_tx(fb, (dbg & 2 ? 0 : XB) + (dbg & 4 ? 0 : WB));
        if (!(dbg & 2)) {
          if constexpr (CLX == 2) tma_2d_mc(dst + cr * (XB / 2), &xmap, fb, (u0 + i) * 64, tb * 128 + cr * 64, (uint16_t)3, pl);
          else tma_2d(dst, &xmap, fb, (u0 + i) * 64, tb * 128, pl);
        }
        if (!(dbg & 4)) {
          if constexpr (CLY == 2) bulk_g2s_mc(dst + XB + cr * (WB / 2), wsrc + (size_t)i * WB + cr * (WB / 2), WB / 2, fb, (uint16_t)3, pe);
          else bulk_g2s(dst + XB, wsrc + (size_t)i * WB, WB, fb, pe);
        }
      }
    }
  } else if (warp == NCW + 1) {
    // escape warp: copies this CTA's escape entries (sorted by k) and, as each stage lands, their x columns
    const int e0 = eoff[sp * NRB + rb], ne = eoff[sp * NRB + rb + 1] - e0;
    for (int e = lane; e < ne; e += 32) esc_ent[e] = etab[e0 + e];
    __syncwarp();
    int ep = 0;
    for (int i = 0; i < nu; ++i) {
      const int slot = i % NS;
      mbar_wait(full0 + 8 * slot, (i / NS) & 1);
      const uint32_t st = sbase + slot * SB;
      const int ucol = (u0 + i) * 64;
      while (ep < ne && esc_ent[ep].y < ucol + 64) {
        const int kk = esc_ent[ep].y - ucol;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int t = lane + 32 * j;
          uint16_t v;
          asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(st + t * 128 + ((((kk >> 3) ^ (t & 7))) << 4) + (kk & 7) * 2));
          esc_x[ep * 128 + t] = v;
        }
        ++ep;
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  } else {
#pragma unroll
    for (int q = 0; q < 64; ++q) acc[q] = 0.f;
    uint32_t af[2][8];
    int prev_slot = -1;
    const uint32_t woff = (uint32_t)(XB + (wg * 4 + wq) * 1536 + lane * 16);
    for (int i = 0; i < nu; ++i) {
      const int slot = i % NS;
      mbar_wait(full0 + 8 * slot, (i / NS) & 1);
      const uint32_t st = sbase + slot * SB;
      if (dbg & 1) {
        if (lane == 0) {
          mbar_arrive(empty0 + 8 * slot);
          if constexpr (CL > 1) mbar_arrive_remote(mapa(empty0 + 8 * slot, cr ^ 1u));
        }
        continue;
      }
      uint4 L0, L1, C;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L0.x), "=r"(L0.y), "=r"(L0.z), "=r"(L0.w) : "r"(st + woff));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L1.x), "=r"(L1.y), "=r"(L1.z), "=r"(L1.w) : "r"(st + woff + 512));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(C.x), "=r"(C.y), "=r"(C.z), "=r"(C.w) : "r"(st + woff + 1024));
#pragma unroll
      for (int kp = 0; kp < 2; ++kp) {
        uint32_t* a = af[kp];
        const uint4 L = kp ? L1 : L0;
        decode8(L.x, L.y, kp ? C.z : C.x, base4, a);
        decode8(L.z, L.w, kp ? C.w : C.y, base4, a + 4);
        wg_fence();
        wgmma128(acc, a[0], a[1], a[2], a[3], desc_sw128(st + (2 * kp) * 32));
        wgmma128(acc, a[4], a[5], a[6], a[7], desc_sw128(st + (2 * kp + 1) * 32));
        wg_commit();
        wg_wait<1>();
        if (kp == 0 && prev_slot >= 0 && lane == 0) {
          mbar_arrive(empty0 + 8 * prev_slot);
          if constexpr (CL > 1) mbar_arrive_remote(mapa(empty0 + 8 * prev_slot, cr ^ 1u));
        }
      }
      prev_slot = slot;
    }
    wg_wait<0>();
  }
  __syncthreads();   // every stage consumed: the ring becomes the partial tile [128 tok][RP] FP32
  if (dbg & 16) { if (sp == 0 && SPLIT > 1 && tid == 0) atomicAdd(cnt + 2 * tileid, 1); return; }
  if (warp < NCW) {
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int row = wg * 64 + wq * 16 + g + (q >= 2 ? 8 : 0);
        const int tok = 8 * j + 2 * t4 + (q & 1);
        tile[tok * RP + row] = acc[4 * j + q];
      }
  }
  __syncthreads();
  // exact corrections for this CTA's escapes (weights outside the exponent window): tile[t][row] += x[t, k] * delta
  {
    const int ne = eoff[sp * NRB + rb + 1] - eoff[sp * NRB + rb];
    for (int idx = tid; idx < ne * 128; idx += NT) {
      const int e = idx >> 7, t = idx & 127;
      const int4 en = esc_ent[e];
      atomicAdd(tile + t * RP + en.x, __uint_as_float((uint32_t)esc_x[idx] << 16) * __int_as_float(en.z));
    }
  }
  __syncthreads();
  if (sp > 0 && !(dbg & 8)) {
    if (tid == 0) {
      const int target = s_target;
      while (ld_acquire(cnt + 2 * tileid) - target < 0) __nanosleep(64);
      __threadfence();
    }
    __syncthreads();
  }
  constexpr int R8 = R / 8;
  for (int idx = tid; idx < 128 * R8; idx += NT) {
    const int t = idx / R8, r8 = idx - (idx / R8) * R8;
    const int tg = tb * 128 + t;
    if (tg >= M) continue;
    const float* s = tile + t * RP + r8 * 8;
    const float4 a = *reinterpret_cast<const float4*>(s), b = *reinterpret_cast<const float4*>(s + 4);
    uint4 o;
    o.x = pack_bf2(a.x, a.y); o.y = pack_bf2(a.z, a.w); o.z = pack_bf2(b.x, b.y); o.w = pack_bf2(b.z, b.w);
    __nv_bfloat16* dst = out + (size_t)tg * N + (size_t)rb * R + r8 * 8;
    if (sp == 0) *reinterpret_cast<uint4*>(dst) = o;
    else red_bf16x8(dst, o);
  }
  if (sp == 0 && SPLIT > 1) {
    __syncthreads();
    if (tid == 0) { __threadfence(); atomicAdd(cnt + 2 * tileid, 1); }
  }
  if constexpr (CL > 1) cluster_sync_all();   // the peer's remote mbarrier arrivals target this CTA until here
}

static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

template <int NWG, int NS, int CLX, int CLY>
static int launch_t(int64_t w, int64_t x, int64_t out, int64_t etab, int64_t eoff, int64_t cnt, int64_t base4,
                    int M, int N, int K, int split, int wkeep, int dbg, int ecap, int64_t stream, int query) {
  constexpr int R = 64 * NWG;
  constexpr int NT = NWG * 128 + 64;
  constexpr int SB = 128 * 128 + R * 96;
  const size_t smem = (size_t)NS * SB + 1024 + (size_t)ecap * (16 + 256);
  auto kern = pb_kernel<NWG, NS, CLX, CLY>;
  if (smem > 232448 - 2048) return -2;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / R, (M + 127) / 128, split);
  cfg.blockDim = dim3(NT, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CLX; at[0].val.clusterDim.y = CLY; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (query >= 2) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    return query == 2 ? fa.numRegs : query == 3 ? fa.maxThreadsPerBlock : (int)fa.sharedSizeBytes;
  }
  if (query) {
    int n = 0;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&n, (void*)kern, &cfg);
    return e == cudaSuccess ? n * CLX * CLY : -(int)e;   // CTAs resident at once
  }
  CUtensorMap mp;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)(CLX == 2 ? 64 : 128)};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, reinterpret_cast<void*>(x), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) return -1000 - (int)r;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, mp, reinterpret_cast<const uint8_t*>(w), reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<const int4*>(etab), reinterpret_cast<const int*>(eoff), reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<int*>(cnt), (uint32_t)base4, M, N, K, K / 64, wkeep, dbg, ecap);
  return e == cudaSuccess ? 0 : (int)e;
}

}  // namespace pbk

int pb_launch(int64_t nwg, int64_t ns, int64_t clx, int64_t cly, int64_t w, int64_t x, int64_t out, int64_t etab,
              int64_t eoff, int64_t cnt, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t split, int64_t wkeep,
              int64_t dbg, int64_t ecap, int64_t stream, int64_t query) {
@@CASES@@
  return -99;
}
"""

# (mode, stages, bft) instantiations compiled
_PB_INST = [  # (warpgroups, stages, cluster x, cluster y) compiled
    (2, 5, 1, 1), (2, 6, 1, 1), (2, 7, 1, 1),
]
_PB_EXT = [None]


def _pb_ext():
    if _PB_EXT[0] is None:
        from torch.utils.cpp_extension import load_inline as _li
        cases = "\n".join(
            f"  if (nwg == {g} && ns == {s} && clx == {a} && cly == {b}) return pbk::launch_t<{g}, {s}, {a}, {b}>(w, x, out, "
            f"etab, eoff, cnt, base4, (int)M, (int)N, (int)K, (int)split, (int)wkeep, (int)dbg, (int)ecap, stream, (int)query);"
            for g, s, a, b in _PB_INST)
        src = _PB_CUDA.replace("@@WGMMA@@", _pb_wgmma_defs()).replace("@@CASES@@", cases)
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\n"
               "int pb_launch(" + ", ".join(["int64_t"] * 20) + ");")
        tag = _pb_hashlib.sha256(src.encode()).hexdigest()[:16]
        verbose = bool(_pb_os.environ.get("PB_VERBOSE"))
        flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"] + (["--ptxas-options=-v"] if verbose else [])
        _PB_EXT[0] = _li("pbk_" + tag, cpp, cuda_sources=src, functions=["pb_launch"], no_implicit_headers=True,
                         with_pytorch_error_handling=False, verbose=verbose, extra_cuda_cflags=flags)
    return _PB_EXT[0]


def _pb_pack(w):
    """vjeux/okc-2026 lossless 12-bit packing (Apache-2.0), returning the global escape list.
    -> dict(packed [U, N/16, 3, 32, 16] uint8, base4, rows, cols, delta) or None."""
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(_pb_torch.int16).to(_pb_torch.int32) & 0xFFFF
    hi = bits >> 8
    e71 = hi & 0x7F
    hist = _pb_torch.bincount(e71.flatten(), minlength=128)
    win = _pb_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(_pb_torch.int16).view(_pb_torch.bfloat16)
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None

    def frag(t):  # [N, K] -> [N/16, K/16, 32, 8] mma A-fragment order
        t = t.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6)
        return t.contiguous().view(n // 16, k // 16, 32, 8)

    lo_f = frag((bits & 0xFF).to(_pb_torch.uint8))
    cd_f = frag(code.to(_pb_torch.uint8))
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8)
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8)
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = _pb_torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [U, nrg, 3, 32, 16]
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows.long(), "cols": cols.long(),
            "delta": delta.float().contiguous()}


def _pb_stream(c, R):
    """[U, N/16, 3, 32, 16] -> [N/R, U, R/16, 3, 32, 16]: one contiguous R*96-byte chunk per (R-row block, 64-k unit)"""
    U, nrg = c["packed"].shape[0], c["packed"].shape[1]
    return c["packed"].view(U, nrg * 16 // R, R // 16, 3, 32, 16).permute(1, 0, 2, 3, 4, 5).contiguous()


def _pb_table(c, R, split, n, k):
    """per (split sp, R-row block rb) escape lists, key sp * NRB + rb: int4 {row in block, k, delta bits, 0}"""
    dev = c["delta"].device
    U, nrb = k // 64, n // R
    rows, cols, delta = c["rows"], c["cols"], c["delta"]
    bounds = _pb_torch.tensor([(s * U) // split for s in range(split + 1)], device=dev)
    sp = _pb_torch.searchsorted(bounds, cols // 64, right=True) - 1
    key = sp * nrb + rows // R
    val = _pb_torch.stack([rows % R, cols, delta.view(_pb_torch.int32).long(), _pb_torch.zeros_like(rows)], 1)
    order = _pb_torch.argsort(key * k + cols)   # by CTA, then by k (the escape warp walks them with the stages)
    key, val = key[order], val[order]
    cnt = _pb_torch.bincount(key, minlength=nrb * split)
    off = _pb_torch.zeros(nrb * split + 1, dtype=_pb_torch.int32, device=dev)
    off[1:] = _pb_torch.cumsum(cnt, 0).to(_pb_torch.int32)
    tab = val.to(_pb_torch.int32).contiguous()
    if tab.numel() == 0:
        tab = _pb_torch.zeros(1, 4, dtype=_pb_torch.int32, device=dev)
    return tab, off.contiguous(), int(cnt.max()) if cnt.numel() else 0


_PB_N = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
# (layer, m) -> (warpgroups, stages, cluster x, cluster y, split, keep weights in L2)
PB_ROUTES = {}


def pb_cell(st, weights, layer, nwg, ns, clx, cly, split, wkeep=0):
    """prepare one (layer, config): stream (shared per layer and R) + escape table + split counters; a cell or None"""
    n, k = _PB_N[layer]
    R = 64 * nwg
    if (nwg, ns, clx, cly) not in _PB_INST or n % R or (n // R) % clx or split < 1 or split > k // 64:
        return None
    if layer not in st["packs"]:
        st["packs"][layer] = _pb_pack(weights[layer])
    c = st["packs"][layer]
    if c is None:
        return None
    if (layer, R) not in st["streams"]:
        st["streams"][(layer, R)] = _pb_stream(c, R)
    tab, off, emax = _pb_table(c, R, split, n, k)
    cnt = _pb_torch.zeros(2 * (n // R) * 4, dtype=_pb_torch.int32, device=tab.device)
    return {"layer": layer, "cfg": (nwg, ns, clx, cly), "split": split, "wkeep": int(wkeep), "n": n, "k": k, "R": R,
            "w": st["streams"][(layer, R)], "tab": tab, "off": off, "cnt": cnt, "base4": c["base4"], "emax": emax,
            "ecap": (emax + 7) // 8 * 8}


def pb_run(x, cell):
    m = x.shape[0]
    out = _pb_torch.empty((m, cell["n"]), dtype=_pb_torch.bfloat16, device=x.device)
    rc = _pb_ext().pb_launch(*cell["cfg"], cell["w"].data_ptr(), x.data_ptr(), out.data_ptr(), cell["tab"].data_ptr(),
                             cell["off"].data_ptr(), cell["cnt"].data_ptr(), cell["base4"], m, cell["n"], cell["k"],
                             cell["split"], cell["wkeep"], cell.get("dbg", 0), cell["ecap"],
                             _pb_torch.cuda.current_stream().cuda_stream, 0)
    if rc != 0:
        raise RuntimeError("pb launch %d" % rc)
    return out


def pb_occ(cell, m=128, query=1):
    """CTAs of this configuration resident at once (the split-K handoff needs the whole grid resident)"""
    return _pb_ext().pb_launch(*cell["cfg"], 0, 0, 0, 0, 0, 0, 0, m, cell["n"], cell["k"], cell["split"], 0, 0, cell["ecap"], 0, query)


def pb_ctas(cell, m):
    return (cell["n"] // cell["R"]) * ((m + 127) // 128) * cell["split"]


def pb_selftest(cell, weights, m):
    """random normal + outlier inputs with probe rows against an FP32 reference of the original weights"""
    w = weights[cell["layer"]]
    g = _pb_torch.Generator(device=w.device).manual_seed(777 + m)
    for outl in (False, True):
        x = _pb_torch.randn((m, cell["k"]), generator=g, device=w.device)
        if outl:
            x[:, _pb_torch.randperm(cell["k"], generator=g, device=w.device)[: cell["k"] // 100]] *= 30
        for j, r in enumerate((0, m // 2, m - 1)):
            x[r] = 0
            x[r, (r * 977 + 31 * j + 5) % cell["k"]] = 64.0 if j != 1 else -16.0
        x = x.to(_pb_torch.bfloat16)
        got = pb_run(x, cell).float()
        _pb_torch.cuda.synchronize()
        ref = x.float() @ w.float().T
        if not bool(_pb_torch.isfinite(got).all()):
            return False, "nonfinite"
        rel = float((got - ref).norm() / ref.norm())
        rows = [0, m // 2, m - 1]
        prel = float(((got[rows] - ref[rows]).norm(dim=1) / ref[rows].norm(dim=1)).max())
        if rel > 6e-3 or prel > 1e-5:
            return False, "rel %.2e probe %.2e" % (rel, prel)
    return True, "ok rel %.2e" % rel


def pb_prepare(weights, routes=None):
    st = {"packs": {}, "streams": {}, "cells": {}, "log": []}
    routes = PB_ROUTES if routes is None else routes
    if not routes:
        return st
    dev = next(iter(weights.values())).device
    if _pb_torch.cuda.get_device_capability(dev) != (9, 0):
        return st
    for (layer, m), cfg in routes.items():
        try:
            cell = pb_cell(st, weights, layer, *cfg)
            if cell is None:
                st["log"].append(f"pb {layer}:{m} unavailable")
                continue
            ctas = pb_ctas(cell, m)
            if ctas > pb_occ(cell, m) or (cell["cfg"][3] == 2 and (m + 127) // 128 % 2):
                st["log"].append(f"pb {layer}:{m} {cfg}: {ctas} CTAs do not fit at once")
                continue
            ok, msg = pb_selftest(cell, weights, m)
            st["log"].append(f"pb {layer}:{m} {cfg} selftest {msg} esc_max {cell['emax']} ctas {ctas}/{pb_occ(cell, m)}")
            if ok:
                st["cells"][(layer, m)] = cell
        except Exception as exc:  # a failed route keeps the parent's kernel for that cell
            st["log"].append(f"pb {layer}:{m} failed {repr(exc)[:200]}")
    return st
