"""rk2: router GEMM (M 33..256) split-K with larger CTA tiles (NWG consumer warpgroups x 64 W rows, BN tokens).
Same hi|lo-on-one-x-tile scheme, in-kernel clear + grid barrier and red.add.v4 reduction as vjeux rk_kernel.
Standalone module for screening: builds both the original rk (baseline) and rk2 in one extension."""
import os, torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
HDR = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>
"""

def _mma(n):
    regs = ", ".join(f"%{i}" for i in range(n // 2))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(n // 2))
    return (f"__device__ __forceinline__ void r2_mma{n}(float (&d)[{n // 2}], uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;\\n"\n'
            f'               : {outs} : "l"(da), "l"(db));\n}}\n')

RK2_SRC = "".join(_mma(n) for n in (64, 128, 256)) + r"""
template <int BN> __device__ __forceinline__ void r2_mma(float (&d)[BN / 2], uint64_t da, uint64_t db) {
  if constexpr (BN == 64) r2_mma64(d, da, db);
  else if constexpr (BN == 128) r2_mma128(d, da, db);
  else r2_mma256(d, da, db);
}
template <int NWG, int BN, int QS, bool PK>
__global__ void __launch_bounds__(NWG * 128 + 64, 1) rk2_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                               float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt, int mode, const char* __restrict__ wp) {
  constexpr int R = 64 * NWG, NRT = 256 / R, NT = NWG * 128;
  constexpr int A_BYTES = R * 128, W_BYTES = 2 * A_BYTES, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char r2_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(r2_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int bid = blockIdx.x;
  const int rt = bid % NRT, tt = (bid / NRT) % ntt, sp = bid / (NRT * ntt);
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, NWG); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == NWG * 4 + 1) {   // helper: clear a slice of out, count, wait for all slices, release the adds
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0 || (mode & 4)) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();
        __nanosleep(64);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == NWG * 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        if constexpr (PK) {
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(st), "l"(wp + ((size_t)rt * 128 + kb) * W_BYTES), "r"(W_BYTES), "r"(fb) : "memory");
        } else {
          rk_tma2d(st, &mapW, fb, kb * 64, rt * R);
          rk_tma2d(st + A_BYTES, &mapW, fb, 8192 + kb * 64, rt * R);
        }
        rk_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  const int wg = tid >> 7, wt = tid & 127;
  float d[BN / 2];
#pragma unroll
  for (int e = 0; e < BN / 2; ++e) d[e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST + wg * 64 * 128, b = sbase + s * ST + W_BYTES;
    if (!(mode & 2))
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      r2_mma<BN>(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      r2_mma<BN>(d, rk_desc(a + A_BYTES + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && wt == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int e = 0; e < BN / 2; ++e) asm volatile("" : "+f"(d[e]) :: "memory");
  if constexpr (NWG > 1) asm volatile("bar.sync 1, %0;" :: "n"(NT) : "memory");   // every warpgroup is done reading the ring
  constexpr int TP = R + 4;
  float* T = reinterpret_cast<float*>(r2_smem + (sbase - sraw));
  const int rl = wg * 64 + ((wt >> 5) * 16) + ((wt & 31) >> 2);
#pragma unroll
  for (int b8 = 0; b8 < BN / 8; ++b8) {
    const int cl = b8 * 8 + (wt & 3) * 2;
    T[cl * TP + rl] = d[4 * b8]; T[cl * TP + rl + 8] = d[4 * b8 + 2];
    T[(cl + 1) * TP + rl] = d[4 * b8 + 1]; T[(cl + 1) * TP + rl + 8] = d[4 * b8 + 3];
  }
  asm volatile("bar.sync 1, %0;" :: "n"(NT) : "memory");
  rk_mbar_wait(zbar, 0);
  constexpr int Q = R / 4;
  if (mode & 1) return;
  for (int idx = tid; idx < BN * Q; idx += NT) {
    const int tk = idx / Q, qd = idx % Q;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * R + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}
static CUtensorMap r2_map(const void* ptr, uint64_t rows, uint64_t cols, uint32_t brows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, brows}; cuuint32_t estr[2] = {1, 1};
  CUresult r = rk_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "rk2: cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
template <int NWG, int BN, bool PK = false>
static void rk2_launch(at::Tensor& Wc, at::Tensor& x, at::Tensor& out, int S, int* zc, int mode, const void* wp = nullptr) {
  constexpr int R = 64 * NWG, ST = 2 * R * 128 + BN * 128;
  constexpr int QS0 = (225 * 1024) / ST, QS = QS0 > 6 ? 6 : QS0;
  static_assert(QS * ST >= BN * (R + 4) * 4, "transpose buffer");
  const int M = x.size(0), ntt = (M + BN - 1) / BN, grid = (256 / R) * ntt * S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); }
  TORCH_CHECK(grid <= nsm, "rk2: grid ", grid, " > SMs ", nsm);
  TORCH_CHECK(S >= 1 && S <= 128, "rk2: S");
  const CUtensorMap mw = r2_map(Wc.data_ptr(), 256, 16384, R), mx = r2_map(x.data_ptr(), M, 8192, BN);
  const int smem = 1024 + QS * ST + 8 * (2 * QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk2_kernel<NWG, BN, QS, PK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk2_kernel<NWG, BN, QS, PK><<<grid, NWG * 128 + 64, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, S, ntt, zc, mode, (const char*)wp);
  C10_CUDA_CHECK(cudaGetLastError());
}
void rk2(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t cfg, int64_t S, at::Tensor zc, int64_t slot, int64_t mode, at::Tensor wp) {
  TORCH_CHECK(Wc.size(0) == 256 && Wc.size(1) == 16384 && x.size(1) == 8192 && x.size(0) >= 1 && x.size(0) <= 256, "rk2: shape");
  int* z = zc.data_ptr<int>() + slot;
  switch (cfg) {
    case 0: rk2_launch<1, 64>(Wc, x, out, (int)S, z, (int)mode); break;
    case 1: rk2_launch<1, 128>(Wc, x, out, (int)S, z, (int)mode); break;
    case 2: rk2_launch<1, 256>(Wc, x, out, (int)S, z, (int)mode); break;
    case 3: rk2_launch<2, 64>(Wc, x, out, (int)S, z, (int)mode); break;
    case 4: rk2_launch<2, 128>(Wc, x, out, (int)S, z, (int)mode); break;
    case 5: rk2_launch<2, 256>(Wc, x, out, (int)S, z, (int)mode); break;
    case 6: rk2_launch<4, 64>(Wc, x, out, (int)S, z, (int)mode); break;
    case 7: rk2_launch<4, 128>(Wc, x, out, (int)S, z, (int)mode); break;
    case 8: rk2_launch<1, 64, true>(Wc, x, out, (int)S, z, (int)mode, wp.data_ptr()); break;
    case 9: rk2_launch<1, 128, true>(Wc, x, out, (int)S, z, (int)mode, wp.data_ptr()); break;
    case 10: rk2_launch<2, 64, true>(Wc, x, out, (int)S, z, (int)mode, wp.data_ptr()); break;
    case 11: rk2_launch<2, 128, true>(Wc, x, out, (int)S, z, (int)mode, wp.data_ptr()); break;
    default: TORCH_CHECK(false, "rk2: cfg");
  }
}
"""
CFGS = {0: (1, 64), 1: (1, 128), 2: (1, 256), 3: (2, 64), 4: (2, 128), 5: (2, 256), 6: (4, 64), 7: (4, 128),
        8: (1, 64), 9: (1, 128), 10: (2, 64), 11: (2, 128)}
PACKED = {8, 9, 10, 11}
CFGS.update({100: (1, 64), 101: (1, 128), 102: (1, 64), 103: (1, 128)})
CFGS.update({110: (1, 64), 111: (1, 128)})
RK3_BN = {100: 64, 101: 128, 102: 1064, 103: 1128, 110: 10064, 111: 10128}

def pack_w(hi, lo, R):
    """[rt][kb][hi|lo][R rows][64 k] bf16, each R x 128 B tile pre-swizzled (TMA 128B pattern: chunk c of row r at c ^ (r & 7))."""
    def tiles(t):
        t = t.view(256 // R, R, 128, 8, 8)               # rt, r, kb, chunk, elem
        r = torch.arange(R, device=t.device) % 8
        c = torch.arange(8, device=t.device)
        perm = c[None, :] ^ r[:, None]                   # destination chunk for source chunk c of row r
        inv = torch.argsort(perm, dim=1)                 # source chunk for destination chunk
        idx = inv[None, :, None, :, None].expand(t.shape)
        return torch.gather(t, 3, idx).permute(0, 2, 1, 3, 4)   # rt, kb, r, chunk, elem
    return torch.stack([tiles(hi), tiles(lo)], dim=2).contiguous()


RK3_SRC = r"""
// rk3: rk structure (1 consumer WG, 64 W rows x BN tokens, split-K, in-kernel clear + red.add) with the lo term stored
// as int8 relative to hi's exponent (lo = q * 2^(E_hi - 142)): per (rt, kb) a contiguous 12 KB packet [hi 8 KB swizzled |
// q 4 KB row-major] by one bulk copy; the consumer expands q to a BF16 lo tile (swizzled) in smem one stage ahead of the MMA.
template <int BN, int QS, bool CW>
__global__ void __launch_bounds__(CW ? 320 : 192, 1) rk3_kernel(const __grid_constant__ CUtensorMap mapX, const char* __restrict__ wp,
                                                    float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt, int mode) {
  constexpr int PKT = 12288, LO = 12288, XO = 20480, ST = XO + BN * 128;
  extern __shared__ __align__(1024) unsigned char r3_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(r3_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = r3_smem + (sbase - sraw);
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS, readyb = zbar + 8;
  const int rt = blockIdx.x & 3, tt = (blockIdx.x >> 2) % ntt, sp = (blockIdx.x >> 2) / ntt;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int WP = CW ? 8 : 4, WH = WP + 1;   // producer / helper warps
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, 1); rk_mbar_init(readyb + 8 * i, 1); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == WH) {
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();
        __nanosleep(128);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == WP) {
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, PKT + BN * 128);
        const int kb = kb0 + i;
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(st), "l"(wp + ((size_t)rt * 128 + kb) * PKT), "r"(PKT), "r"(fb) : "memory");
        rk_tma2d(st + XO, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  // consumer warpgroup
  const int ct = tid & 127;
  auto expand = [&](int s) {   // q (int8, row-major [64][64]) + hi exponents -> BF16 lo tile, 128B-swizzled like hi
    unsigned char* st = gbase + s * ST;
    if (!(mode & 8))
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int id = ct + 128 * u, r = id >> 3, c = id & 7;
      const int so = r * 128 + ((c ^ (r & 7)) << 4);
      const uint4 h = *reinterpret_cast<const uint4*>(st + so);
      const uint2 q = *reinterpret_cast<const uint2*>(st + 8192 + r * 64 + c * 8);
      const uint32_t hw[4] = {h.x, h.y, h.z, h.w};
      const uint32_t qw[2] = {q.x, q.y};
      uint32_t o[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        float f[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int E = (hw[p] >> (16 * e + 7)) & 0xFF;
          const int qi = (int)(signed char)((qw[p >> 1] >> (8 * (2 * (p & 1) + e))) & 0xFF);
          const float sc = __int_as_float(max(E - 15, 0) << 23);
          f[e] = (float)qi * sc;
        }
        __nv_bfloat162 b = __floats2bfloat162_rn(f[0], f[1]);
        o[p] = *reinterpret_cast<uint32_t*>(&b);
      }
      *reinterpret_cast<uint4*>(st + LO + so) = make_uint4(o[0], o[1], o[2], o[3]);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if constexpr (CW) asm volatile("bar.sync 2, 128;" ::: "memory");
    else asm volatile("bar.sync 1, 128;" ::: "memory");
  };
  if constexpr (CW) {
    if (warp >= 4) {   // converter warpgroup: expand each stage's lo as soon as it lands
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
        expand(s);
        if (ct == 0) rk_mbar_arrive(readyb + 8 * s);
      }
      return;
    }
  }
  float d[BN / 2];
#pragma unroll
  for (int e = 0; e < BN / 2; ++e) d[e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    if constexpr (CW) rk_mbar_wait(readyb + 8 * s, (i / QS) & 1);
    else expand(s);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + XO;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      r2_mma<BN>(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      r2_mma<BN>(d, rk_desc(a + LO + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int e = 0; e < BN / 2; ++e) asm volatile("" : "+f"(d[e]) :: "memory");
  constexpr int TP = 68;
  float* T = reinterpret_cast<float*>(gbase);
  const int rl = ((tid >> 5) * 16) + ((tid & 31) >> 2);
#pragma unroll
  for (int b8 = 0; b8 < BN / 8; ++b8) {
    const int cl = b8 * 8 + (tid & 3) * 2;
    T[cl * TP + rl] = d[4 * b8]; T[cl * TP + rl + 8] = d[4 * b8 + 2];
    T[(cl + 1) * TP + rl] = d[4 * b8 + 1]; T[(cl + 1) * TP + rl + 8] = d[4 * b8 + 3];
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  rk_mbar_wait(zbar, 0);
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * 64 + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}
template <int BN, bool CW>
static void rk3_launch(at::Tensor& wp, at::Tensor& x, at::Tensor& out, int S, int* zc, int mode) {
  constexpr int ST = 20480 + BN * 128;
  constexpr int QS0 = (225 * 1024) / ST, QS = QS0 > 6 ? 6 : QS0;
  static_assert(QS * ST >= BN * 68 * 4, "transpose buffer");
  const int M = x.size(0), ntt = (M + BN - 1) / BN, grid = 4 * ntt * S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); }
  TORCH_CHECK(grid <= nsm && S >= 1 && S <= 128, "rk3: grid");
  const CUtensorMap mx = r2_map(x.data_ptr(), M, 8192, BN);
  const int smem = 1024 + QS * ST + 8 * (3 * QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk3_kernel<BN, QS, CW>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk3_kernel<BN, QS, CW><<<grid, CW ? 320 : 192, smem, at::cuda::getCurrentCUDAStream()>>>(mx, (const char*)wp.data_ptr(), out.data_ptr<float>(), M, S, ntt, zc, mode);
  C10_CUDA_CHECK(cudaGetLastError());
}
void rk3(at::Tensor wp, at::Tensor x, at::Tensor out, int64_t bn, int64_t S, at::Tensor zc, int64_t slot) {
  const int mode = bn >= 10000 ? 8 : 0; bn %= 10000;
  TORCH_CHECK(wp.numel() == 4 * 128 * 12288 && x.size(1) == 8192 && x.size(0) >= 1 && x.size(0) <= 256, "rk3: shape");
  int* z = zc.data_ptr<int>() + slot;
  if (bn == 64) rk3_launch<64, false>(wp, x, out, (int)S, z, mode);
  else if (bn == 128) rk3_launch<128, false>(wp, x, out, (int)S, z, mode);
  else if (bn == 1064) rk3_launch<64, true>(wp, x, out, (int)S, z, mode);
  else if (bn == 1128) rk3_launch<128, true>(wp, x, out, (int)S, z, mode);
  else TORCH_CHECK(false, "rk3: bn");
}
"""

def pack_q8(w):
    """Router FP32 W -> per (rt, kb) 12 KB packets [hi BF16 64x64, TMA-128B-swizzled | q int8 64x64 row-major],
    lo = w - hi = q * 2^(E_hi - 142) with E_hi hi's biased exponent (|q| <= 128, clamped to 127)."""
    hi = w.to(torch.bfloat16)
    lo = w - hi.float()
    E = (hi.view(torch.int16).to(torch.int32) >> 7) & 0xFF
    sc = torch.ldexp(torch.ones_like(lo), (E - 142).clamp(min=-126))
    q = torch.round(lo / sc).clamp(-127, 127).to(torch.int8)
    q = torch.where(E >= 16, q, torch.zeros_like(q))
    t = hi.view(4, 64, 128, 8, 8)
    r = torch.arange(64, device=w.device) % 8
    perm = torch.arange(8, device=w.device)[None, :] ^ r[:, None]
    idx = perm[None, :, None, :, None].expand(t.shape)
    hs = torch.gather(t, 3, idx).permute(0, 2, 1, 3, 4).reshape(4, 128, 4096).view(torch.uint8)
    qs = q.view(4, 64, 128, 64).permute(0, 2, 1, 3).reshape(4, 128, 4096).view(torch.uint8)
    return torch.cat([hs, qs], dim=2).contiguous()

def q8_reconstruct(w):
    """Host model of what rk3 multiplies by: hi + q * 2^(E-142) in FP64 (for error accounting)."""
    hi = w.to(torch.bfloat16)
    lo = w - hi.float()
    E = (hi.view(torch.int16).to(torch.int32) >> 7) & 0xFF
    sc = torch.ldexp(torch.ones_like(lo), (E - 142).clamp(min=-126))
    q = torch.round(lo / sc).clamp(-127, 127)
    q = torch.where(E >= 16, q, torch.zeros_like(q))
    return hi.double() + q.double() * sc.double()

CPP = """#include <torch/extension.h>
void rk3(at::Tensor wp, at::Tensor x, at::Tensor out, int64_t bn, int64_t S, at::Tensor zc, int64_t slot);
void rk(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t S, at::Tensor zc);
void rk2(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t cfg, int64_t S, at::Tensor zc, int64_t slot, int64_t mode, at::Tensor wp);
"""

def build(orig_src):
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    return load_inline(name="rk2_screen_v5", cpp_sources=CPP, cuda_sources=HDR + orig_src + RK2_SRC + RK3_SRC,
                       functions=["rk", "rk2", "rk3"], extra_cuda_cflags=["-O3", "-Xptxas=-v", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                       extra_ldflags=["-lcuda"], no_implicit_headers=True, verbose=True)

ORIG_SRC = r"""
// ============================================================================ rk: router GEMM, m 33..256 (sm_90)
// out[M, 256] (FP32) = x[M, 8192] (BF16) @ (hi + lo)[256, 8192]^T on router_cat = [hi | lo] [256, 16384] (BF16).
// CTA (192 threads) = 1 consumer warpgroup (SS-wgmma m64n64k16 on the hi tile, then the lo tile, of each 64-k block)
// + 1 producer warp (TMA, RK_QS-stage ring of 24 KB: hi 8 KB | lo 8 KB | x 8 KB, 128B swizzle) + 1 helper warp.
// Tile = 64 W rows x 64 tokens; split-K over S CTAs per tile (grid 4 x ntt x S <= 128: one CTA per SM, all resident).
// Each CTA ADDS its partial tile into the FP32 output (smem-transposed, 16 B red.add.v4.f32): the reduction is
// ORDER-DEPENDENT (run-to-run differences ~1e-7 relative, inside the router tolerance; POD item 6). The output is
// cleared inside the kernel: every CTA's helper warp reads the grid counter of this grid size (generation =
// value / grid: earlier launches' counts are complete and this CTA has not counted yet), clears its slice of out,
// counts it (release), waits (acquire, 2 s limit -> trap) until every CTA has counted, and releases the consumers'
// adds through a CTA mbarrier. Monotonic counters (one per token-tile count = per grid size): no host reset.
#define RK_QS 6
__device__ __forceinline__ uint64_t rk_desc(uint32_t addr) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void rk_mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void rk_mbar_expect(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void rk_mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void rk_mbar_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void rk_tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void rk_wgmma(float d[4][8], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(da), "l"(db));
}

__global__ void __launch_bounds__(192, 1) rk_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                     float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt) {
  constexpr int BM = 64, BN = 64, NJ = 4, QS = RK_QS;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char rk_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rk_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int rt = blockIdx.x & 3, tt = (blockIdx.x >> 2) % ntt, sp = (blockIdx.x >> 2) / ntt;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, 1); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 5) {   // helper: clear this CTA's slice of out, count it, wait for every slice, release the adds
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;   // float4s in out
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();   // 2 s: a clear that never lands is a bug, never a slow node
        __nanosleep(128);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        rk_tma2d(st, &mapW, fb, kb * 64, rt * BM);
        rk_tma2d(st + BM * 128, &mapW, fb, 8192 + kb * 64, rt * BM);
        rk_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  // consumer warpgroup
  float d[NJ][8];
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) d[j][e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      rk_wgmma(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      rk_wgmma(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));   // stage i-1's wgmmas are done
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+f"(d[j][e]) :: "memory");
  // partial tile -> shared memory [token][row] (the ring is idle: every stage was waited for), then 16 B adds
  constexpr int TP = 68;
  float* T = reinterpret_cast<float*>(rk_smem + (sbase - sraw));
  const int rl = (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int cl = j * 16 + h * 8 + (tid & 3) * 2;
      T[cl * TP + rl] = d[j][4 * h]; T[cl * TP + rl + 8] = d[j][4 * h + 2];
      T[(cl + 1) * TP + rl] = d[j][4 * h + 1]; T[(cl + 1) * TP + rl + 8] = d[j][4 * h + 3];
    }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  rk_mbar_wait(zbar, 0);   // every CTA's slice of out is clear (acquired by the helper warp)
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * BM + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}
static PFN_cuTensorMapEncodeTiled_v12000 rk_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr; void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr);
    TORCH_CHECK(p != nullptr, "rk: cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap rk_map(const void* ptr, uint64_t rows, uint64_t cols) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, 64}; cuuint32_t estr[2] = {1, 1};
  CUresult r = rk_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "rk: cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
void rk(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t S, at::Tensor zc) {
  const int M = x.size(0);
  TORCH_CHECK(Wc.size(0) == 256 && Wc.size(1) == 16384 && x.size(1) == 8192 && M >= 1 && M <= 256, "rk: shape");
  const int ntt = (M + 63) / 64;
  const int grid = 4 * ntt * (int)S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); }
  TORCH_CHECK(grid <= nsm, "rk: grid ", grid, " exceeds the SM count ", nsm, " (every CTA must be resident: one per SM)");
  const CUtensorMap mw = rk_map(Wc.data_ptr(), 256, 16384), mx = rk_map(x.data_ptr(), M, 8192);
  const int smem = 1024 + RK_QS * 24576 + 8 * (2 * RK_QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk_kernel<<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
  C10_CUDA_CHECK(cudaGetLastError());
}
"""
