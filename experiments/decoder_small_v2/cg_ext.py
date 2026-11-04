"""decoder_small_v2: m = 1 lossless 12-bit compressed GEMV (cgemv) for the decoder's T = 1 projections.

Kernel adapted from experiments/projection_fp32_compactq128_bucket_v7 (cgemv_kernel, vjeux/okc-2026 lineage, Apache-2.0).
Changes: PDL support. The kernel issues every weight load (independent of the previous kernel) before
griddepcontrol.wait, then reads x; it triggers its dependents right after the wait. FP32 accumulation throughout;
the only rounding is the final BF16 output (no partials leave the CTA).
"""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

CG_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float cg_esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
__device__ __forceinline__ void cg_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void cg_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// ROWS output rows per CTA, NTH compute threads + 1 escape warp. Weights row-major planes: L [N][K] low bytes,
// C [N][K/2] 4-bit codes. WAIT: launched as a PDL secondary, so x is read only after griddepcontrol.wait.
template <int ROWS, int NTH, int K, bool WAIT>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: x[k] * delta per row from the per-CTA table (entry 0 = count)
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    if (WAIT) cg_wait();
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * cg_esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
  if (WAIT) cg_wait();
  if (tid == 0) cg_trigger();
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = *reinterpret_cast<const uint4*>(xp);       // plain loads: x is written by the previous kernel
    xv[j][1] = *reinterpret_cast<const uint4*>(xp + 8);
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

template <int ROWS, int NTH, int K, bool WAIT>
static void cg_launch(int N, const uint8_t* a0, const uint8_t* a1, const __nv_bfloat16* xp, const int2* tp, int pm,
                      __nv_bfloat16* op, uint32_t b4, cudaStream_t st) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / ROWS); cfg.blockDim = dim3(NTH + 32); cfg.dynamicSmemBytes = 0; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = WAIT ? 1 : 0;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, cgemv_kernel<ROWS, NTH, K, WAIT>, a0, a1, xp, tp, pm, op, b4);
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth, int64_t pdl) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");
  auto st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
#define CGL(R_, N_, K_) \
  if (K == K_ && rows == R_ && nth == N_) { \
    if (pdl) cg_launch<R_, N_, K_, true>(N, a0, a1, xp, tp, pm, op, b4, st); \
    else cg_launch<R_, N_, K_, false>(N, a0, a1, xp, tp, pm, op, b4, st); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  CGL(2, 256, 8192) CGL(2, 512, 8192) CGL(4, 256, 8192) CGL(1, 512, 8192)
  CGL(2, 128, 4096) CGL(2, 256, 4096) CGL(4, 128, 4096) CGL(1, 256, 4096)
  CGL(2, 64, 1024)
  TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);
}
'''
CG_CPP = "void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4, int64_t rows, int64_t nth, int64_t pdl);"

_EXT = None


def cg_ext():
    global _EXT
    if _EXT is None:
        tag = hashlib.md5(CG_CUDA.encode()).hexdigest()[:12]
        _EXT = load_inline("dsv2_cgemv_" + tag, CG_CPP, cuda_sources=CG_CUDA, functions=["cgemv"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _EXT


def cg_compress(w, rows_per_cta):
    """Row-major lossless 12-bit planes (low byte + 4-bit sign/exponent-offset code) + per-CTA escape table."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)
    dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
    wtrue = w[rows, cols].float()
    if not bool(((dec + (wtrue - dec)) == wtrue).all()):
        return None
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101, "rows": rows_per_cta}


def cg_run(cg, x, n, nth, pdl):
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    cg_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], cg["rows"], nth, int(pdl))
    return out
