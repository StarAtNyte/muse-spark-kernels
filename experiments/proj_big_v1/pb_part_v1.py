

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

// Weight stream Wst: [NRB][U][R/16 row groups][3 parts][32 lanes][16 B] (vjeux 12-bit fragment order per 64-k unit).
// Escape table: per CTA a list of int4 {row in CTA, k column, FP32 delta bits, 0}.
template <int MODE, int NS, bool BFT>
__global__ void __launch_bounds__(384, 1)
pb_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Wst, __nv_bfloat16* __restrict__ out,
          const int4* __restrict__ etab, const int* __restrict__ eoff, const __nv_bfloat16* __restrict__ x,
          uint32_t base4, int M, int N, int K, int U) {
  constexpr int R = MODE == 0 ? 256 : 128;
  constexpr int TOK = MODE == 1 ? 256 : 128;
  constexpr int WB = R * 96;
  constexpr int XB = TOK * 128;
  constexpr int SB = XB + WB;
  constexpr bool BT = BFT && MODE != 2;
  constexpr int RP = BT ? R + 8 : R + 4;      // tile row pitch (elements)
  constexpr int NCW = MODE == 2 ? 4 : 8;      // warps releasing each stage
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  const uint32_t sraw = su32(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* sgen = smem_raw + (sbase - sraw);
  const uint32_t full0 = su32(bars), empty0 = full0 + 8 * NS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, SPLIT = gridDim.y;
  const int u0 = (int)((long long)sp * U / SPLIT), u1 = (int)((long long)(sp + 1) * U / SPLIT);
  const int nu = u1 - u0;
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
  }
  __syncthreads();

  float acc0[64], acc1[64];
  const int wg = warp >> 2, wq = warp & 3;
  const int rw = MODE == 0 ? wg * 128 : 0;
  const int tw = MODE == 1 ? wg * 128 : 0;
  if (warp >= 8) {
    if (warp == 8 && lane == 0) {
      const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * WB;
      const uint64_t pe = pol_ef(), pl = pol_el();
      for (int i = 0; i < nu; ++i) {
        const int slot = i % NS;
        if (i >= NS) mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const uint32_t dst = sbase + slot * SB;
        mbar_expect_tx(full0 + 8 * slot, SB);
        tma_2d(dst, &xmap, full0 + 8 * slot, (u0 + i) * 64, 0, pl);
        bulk_g2s(dst + XB, wsrc + (size_t)i * WB, WB, full0 + 8 * slot, pe);
      }
    }
  } else {
#pragma unroll
    for (int q = 0; q < 64; ++q) { acc0[q] = 0.f; acc1[q] = 0.f; }
    uint32_t af[2][8];
    const int first = MODE == 2 ? wg : 0;
    constexpr int step = MODE == 2 ? 2 : 1;
    int prev_slot = -1;
    const uint32_t woff = (uint32_t)(XB + ((rw >> 4) + wq) * 1536 + lane * 16);
    for (int i = first; i < nu; i += step) {
      const int slot = i % NS;
      mbar_wait(full0 + 8 * slot, (i / NS) & 1);
      const uint32_t st = sbase + slot * SB;
      const uint32_t xb = st + tw * 128;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t* a = af[j & 1];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t wa = st + woff + h * 4 * 1536;
          uint32_t lw0, lw1, cw;
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + j * 4));
          decode8(lw0, lw1, cw, base4, a + h * 4);
        }
        wg_fence();
        const uint64_t d = desc_sw128(xb + j * 32);
        wgmma128(acc0, a[0], a[1], a[2], a[3], d);
        wgmma128(acc1, a[4], a[5], a[6], a[7], d);
        wg_commit();
        wg_wait<1>();
        if (j == 0 && prev_slot >= 0 && lane == 0) mbar_arrive(empty0 + 8 * prev_slot);
      }
      prev_slot = slot;
    }
    wg_wait<0>();
  }
  __syncthreads();   // every stage consumed by every warpgroup: the ring becomes the partial tile

  // ---- partial tile [TOK][RP] in shared memory (FP32, or BF16 when BT)
  if (warp < 8) {
    const int g = lane >> 2, t4 = lane & 3;
    const bool writer = MODE != 2 || wg == 1;
    if (writer) {
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int row = rw + h * 64 + wq * 16 + g + (q >= 2 ? 8 : 0);
            const int tok = tw + 8 * j + 2 * t4 + (q & 1);
            const float v = h ? acc1[4 * j + q] : acc0[4 * j + q];
            if constexpr (BT) reinterpret_cast<__nv_bfloat16*>(sgen)[tok * RP + row] = __float2bfloat16_rn(v);
            else reinterpret_cast<float*>(sgen)[tok * RP + row] = v;
          }
    }
  }
  if constexpr (MODE == 2) {
    __syncthreads();
    if (warp < 4) {
      const int g = lane >> 2, t4 = lane & 3;
      float* tile = reinterpret_cast<float*>(sgen);
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int row = h * 64 + wq * 16 + g + (q >= 2 ? 8 : 0);
            const int tok = 8 * j + 2 * t4 + (q & 1);
            tile[tok * RP + row] += h ? acc1[4 * j + q] : acc0[4 * j + q];
          }
    }
  }
  __syncthreads();
  // exact corrections for this CTA's escapes (weights outside the exponent window): tile[t][row] += x[t, k] * delta
  {
    const int cta = rb * SPLIT + sp;
    const int e0 = eoff[cta], ne = eoff[cta + 1] - e0;
    for (int idx = tid; idx < ne * TOK; idx += 384) {
      const int e = e0 + idx / TOK, t = idx - (idx / TOK) * TOK;
      if (t >= M) continue;
      const int4 en = etab[e];
      const float v = __bfloat162float(x[(size_t)t * K + en.y]) * __int_as_float(en.z);
      if constexpr (BT) atomicAdd(reinterpret_cast<__nv_bfloat16*>(sgen) + t * RP + en.x, __float2bfloat16_rn(v));
      else atomicAdd(reinterpret_cast<float*>(sgen) + t * RP + en.x, v);
    }
  }
  cluster_sync_all();

  // ---- split-K combine across the cluster (DSMEM), FP32 sums, BF16 output; rank c owns a token slice
  {
    const int CK = SPLIT;
    const int t0 = sp * TOK / CK, t1 = (sp + 1) * TOK / CK;
    const uint32_t tb = sbase;
    if constexpr (BT) {
      constexpr int R8 = R / 8;
      for (int idx = tid; idx < (t1 - t0) * R8; idx += 384) {
        const int t = t0 + idx / R8, r8 = idx - (idx / R8) * R8;
        const uint32_t la = tb + (uint32_t)((t * RP + r8 * 8) * 2);
        float s[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) s[c] = 0.f;
        for (int q = 0; q < CK; ++q) {
          const uint4 v = ld_dsm_u4(mapa(la, (uint32_t)q));
          s[0] += bf_lo(v.x); s[1] += bf_hi(v.x); s[2] += bf_lo(v.y); s[3] += bf_hi(v.y);
          s[4] += bf_lo(v.z); s[5] += bf_hi(v.z); s[6] += bf_lo(v.w); s[7] += bf_hi(v.w);
        }
        if (t < M) {
          uint4 o;
          o.x = pack_bf2(s[0], s[1]); o.y = pack_bf2(s[2], s[3]); o.z = pack_bf2(s[4], s[5]); o.w = pack_bf2(s[6], s[7]);
          *reinterpret_cast<uint4*>(out + (size_t)t * N + (size_t)rb * R + r8 * 8) = o;
        }
      }
    } else {
      constexpr int R4 = R / 4;
      for (int idx = tid; idx < (t1 - t0) * R4; idx += 384) {
        const int t = t0 + idx / R4, r4 = idx - (idx / R4) * R4;
        const uint32_t la = tb + (uint32_t)((t * RP + r4 * 4) * 4);
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int q = 0; q < CK; ++q) {
          const float4 v = ld_dsm_f4(mapa(la, (uint32_t)q));
          s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        if (t < M) {
          uint2 o; o.x = pack_bf2(s.x, s.y); o.y = pack_bf2(s.z, s.w);
          *reinterpret_cast<uint2*>(out + (size_t)t * N + (size_t)rb * R + r4 * 4) = o;
        }
      }
    }
  }
  cluster_sync_all();   // remote tiles stay alive until every rank has read them
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

template <int MODE, int NS, bool BFT>
static int launch_t(int64_t w, int64_t x, int64_t out, int64_t etab, int64_t eoff, int64_t base4,
                    int M, int N, int K, int nrb, int split, int64_t stream, int query) {
  constexpr int R = MODE == 0 ? 256 : 128;
  constexpr int TOK = MODE == 1 ? 256 : 128;
  constexpr int SB = TOK * 128 + R * 96;
  const size_t smem = (size_t)NS * SB + 1024;
  auto kern = pb_kernel<MODE, NS, BFT>;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(nrb, split, 1);
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = split; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (query) {
    int n = 0;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&n, (void*)kern, &cfg);
    return e == cudaSuccess ? n : -(int)e;
  }
  CUtensorMap mp;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)TOK};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, reinterpret_cast<void*>(x), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) return -1000 - (int)r;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, mp, reinterpret_cast<const uint8_t*>(w), reinterpret_cast<__nv_bfloat16*>(out),
                                     reinterpret_cast<const int4*>(etab), reinterpret_cast<const int*>(eoff),
                                     reinterpret_cast<const __nv_bfloat16*>(x), (uint32_t)base4, M, N, K, K / 64);
  return e == cudaSuccess ? 0 : (int)e;
}

}  // namespace pbk

int pb_launch(int64_t mode, int64_t ns, int64_t bft, int64_t w, int64_t x, int64_t out, int64_t etab, int64_t eoff,
              int64_t base4, int64_t M, int64_t N, int64_t K, int64_t nrb, int64_t split, int64_t stream, int64_t query) {
@@CASES@@
  return -99;
}
"""

# (mode, stages, bft) instantiations compiled
_PB_INST = [(0, 4, 0), (0, 4, 1), (0, 5, 1), (1, 4, 0), (1, 4, 1), (2, 6, 0)]
_PB_EXT = [None]


def _pb_ext():
    if _PB_EXT[0] is None:
        from torch.utils.cpp_extension import load_inline as _li
        cases = "\n".join(
            f"  if (mode == {m} && ns == {s} && bft == {b}) return pbk::launch_t<{m}, {s}, {'true' if b else 'false'}>"
            f"(w, x, out, etab, eoff, base4, (int)M, (int)N, (int)K, (int)nrb, (int)split, stream, (int)query);"
            for m, s, b in _PB_INST)
        src = _PB_CUDA.replace("@@WGMMA@@", _pb_wgmma_defs()).replace("@@CASES@@", cases)
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\n"
               "int pb_launch(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, "
               "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);")
        tag = _pb_hashlib.sha256(src.encode()).hexdigest()[:16]
        _PB_EXT[0] = _li("pbk_" + tag, cpp, cuda_sources=src, functions=["pb_launch"], no_implicit_headers=True,
                         with_pytorch_error_handling=False, verbose=bool(_pb_os.environ.get("PB_VERBOSE")),
                         extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                            "--ptxas-options=-v"] if _pb_os.environ.get("PB_VERBOSE") else
                         ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
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
    U, nrg = c["packed"].shape[0], c["packed"].shape[1]
    nrb = nrg * 16 // R
    return c["packed"].view(U, nrb, R // 16, 3, 32, 16).permute(1, 0, 2, 3, 4, 5).contiguous()


def _pb_table(c, mode, split, n, k):
    """per CTA (rb * split + sp) escape lists: int4 {row in CTA, k, delta bits, 0}, offsets [ncta + 1]"""
    dev = c["delta"].device
    R = 256 if mode == 0 else 128
    U = k // 64
    nrb = n // R
    rows, cols, delta = c["rows"], c["cols"], c["delta"]
    unit = cols // 64
    bounds = _pb_torch.tensor([(s * U) // split for s in range(split + 1)], device=dev)
    sp = _pb_torch.searchsorted(bounds, unit, right=True) - 1
    rb = rows // R
    rl = rows % R
    key = rb * split + sp
    val = _pb_torch.stack([rl, cols, delta.view(_pb_torch.int32).long(), _pb_torch.zeros_like(rl)], 1)
    order = _pb_torch.argsort(key, stable=True)
    key, val = key[order], val[order]
    nslot = nrb * split
    cnt = _pb_torch.bincount(key, minlength=nslot)
    off = _pb_torch.zeros(nslot + 1, dtype=_pb_torch.int32, device=dev)
    off[1:] = _pb_torch.cumsum(cnt, 0).to(_pb_torch.int32)
    tab = val.to(_pb_torch.int32).contiguous()
    if tab.numel() == 0:
        tab = _pb_torch.zeros(1, 4, dtype=_pb_torch.int32, device=dev)
    return tab, off.contiguous(), int(cnt.max()) if cnt.numel() else 0


_PB_N = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
# (layer, m) -> (mode, stages, bft, split)
PB_ROUTES = {}


def pb_cell(st, weights, layer, mode, ns, bft, split):
    """prepare one (layer, config): stream (shared per R) + escape table; returns a cell dict or None"""
    n, k = _PB_N[layer]
    R = 256 if mode == 0 else 128
    if n % R or split < 1 or split > 8:
        return None
    if layer not in st["packs"]:
        st["packs"][layer] = _pb_pack(weights[layer])
    c = st["packs"][layer]
    if c is None:
        return None
    sk = (layer, R)
    if sk not in st["streams"]:
        st["streams"][sk] = _pb_stream(c, R)
    tab, off, emax = _pb_table(c, mode, split, n, k)
    return {"layer": layer, "mode": mode, "ns": ns, "bft": bft, "split": split, "n": n, "k": k, "nrb": n // R,
            "w": st["streams"][sk], "tab": tab, "off": off, "base4": c["base4"], "emax": emax}


def pb_run(x, cell):
    m = x.shape[0]
    out = _pb_torch.empty((m, cell["n"]), dtype=_pb_torch.bfloat16, device=x.device)
    rc = _pb_ext().pb_launch(cell["mode"], cell["ns"], cell["bft"], cell["w"].data_ptr(), x.data_ptr(), out.data_ptr(),
                             cell["tab"].data_ptr(), cell["off"].data_ptr(), cell["base4"], m, cell["n"], cell["k"],
                             cell["nrb"], cell["split"], _pb_torch.cuda.current_stream().cuda_stream, 0)
    if rc != 0:
        raise RuntimeError("pb launch %d" % rc)
    return out


def pb_occ(cell):
    return _pb_ext().pb_launch(cell["mode"], cell["ns"], cell["bft"], 0, 0, 0, 0, 0, 0, 128, cell["n"], cell["k"],
                               cell["nrb"], cell["split"], 0, 1)


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
        if rel > 4e-3 or prel > 1e-5:
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
            ok, msg = pb_selftest(cell, weights, m)
            st["log"].append(f"pb {layer}:{m} {cfg} selftest {msg} esc_max {cell['emax']} occ {pb_occ(cell)}")
            if ok:
                st["cells"][(layer, m)] = cell
        except Exception as exc:  # a failed route keeps the parent's kernel for that cell
            st["log"].append(f"pb {layer}:{m} failed {repr(exc)[:200]}")
    return st
