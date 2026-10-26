"""Lossless 12-bit BF16 weights (8-bit low byte + 4-bit sign/exponent-window code) and a swap-AB mma.sync GEMV for 1..32 tokens."""

CUDA_SRC = r"""
#ifndef NOX
#define NOX 0
#endif
#ifndef NOW
#define NOW 0
#endif
#ifndef NODEC
#define NODEC 0
#endif
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)), "l"(gsrc));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// one k64 chunk of a 16-row tile = 3 planes x 32 lanes x uint4: plane0/1 hold the low bytes, plane2 the 4-bit codes.
// Packed index: ((tile * KC + chunk) * 3 + plane) * 32 + lane.
template <int NT, int NW, int NCH, int S>
__global__ void __launch_bounds__(NW * 32)
bf16c_gemm(const uint4* __restrict__ Wc, const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ esc_off,
           const int2* __restrict__ esc, int N, int K, int m, uint32_t base4) {
  extern __shared__ uint4 smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int KC = K >> 6, CPW = KC / NW;
  const int tile0 = blockIdx.x * NT;

  const uint4* wsrc[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) wsrc[n] = Wc + (((size_t)(tile0 + n) * KC + warp * CPW) * 3) * 32 + lane;
  const bf16* xp[NCH];
#pragma unroll
  for (int h = 0; h < NCH; h++) {
    const int tok = h * 8 + g;
    xp[h] = tok < m ? X + (size_t)tok * K + warp * CPW * 64 + 16 * t : nullptr;
  }
  pdl_wait();

  uint4* ring = smem + warp * (S * NT * 3 * 32);
  auto issue = [&](int c) {
    if (c < CPW) {
#pragma unroll
      for (int n = 0; n < NT; n++)
#pragma unroll
        for (int p = 0; p < 3; p++) { if (!NOW) cp16(&ring[(((c % S) * NT + n) * 3 + p) * 32 + lane], wsrc[n] + (size_t)c * 96 + p * 32); }
    }
    cp_commit();
  };
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);

  float acc[NT][NCH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][h][r] = 0.f;

  uint4 xcur[NCH][2], xnxt[NCH][2];
#pragma unroll
  for (int h = 0; h < NCH; h++) {
    if (xp[h] && !NOX) {
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
      if (xp[h] && i + 1 < CPW && !NOX) {
        const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (i + 1) * 64);
        xnxt[h][0] = __ldcg(xv);
        xnxt[h][1] = __ldcg(xv + 1);
      }
    }
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const uint4 l0 = ring[(((i % S) * NT + n) * 3 + 0) * 32 + lane];
      const uint4 l1 = ring[(((i % S) * NT + n) * 3 + 1) * 32 + lane];
      const uint4 cd = ring[(((i % S) * NT + n) * 3 + 2) * 32 + lane];
      const uint32_t lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
      const uint32_t cw[4] = {cd.x, cd.y, cd.z, cd.w};
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        const uint32_t nb = cw[ks];
        uint32_t a0, a1, a2, a3;
        if (NODEC) { a0 = lw[2 * ks]; a1 = lw[2 * ks] + nb; a2 = lw[2 * ks + 1]; a3 = lw[2 * ks + 1] ^ nb; }
        else {
        const uint32_t h0 = ((nb & 0x07070707u) + base4) | ((nb << 4) & 0x80808080u);
        const uint32_t h1 = (((nb >> 4) & 0x07070707u) + base4) | (nb & 0x80808080u);
        a0 = prmt(lw[2 * ks], h0, 0x5140u); a1 = prmt(lw[2 * ks], h0, 0x7362u);
        a2 = prmt(lw[2 * ks + 1], h1, 0x5140u); a3 = prmt(lw[2 * ks + 1], h1, 0x7362u);
        }
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const uint4 xs = xcur[h][ks >> 1];
          mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
        }
      }
    }
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      xcur[h][0] = xnxt[h][0];
      xcur[h][1] = xnxt[h][1];
    }
  }
  cp_wait<0>();
  __syncthreads();

  // cross-warp reduction, exact escape corrections, bf16 store
  float* red = reinterpret_cast<float*>(smem);
  float* fin = red + NW * NT * NCH * 4 * 32;
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int r = 0; r < 4; r++) red[(((warp * NT + n) * NCH + h) * 4 + r) * 32 + lane] = acc[n][h][r];
  __syncthreads();
  for (int item = threadIdx.x; item < NT * NCH * 4 * 32; item += NW * 32) {
    const int li = item & 31, r = (item >> 5) & 3, h = (item >> 7) % NCH, n = (item >> 7) / NCH;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w++) s += red[(((w * NT + n) * NCH + h) * 4 + r) * 32 + li];
    const int row = (li >> 2) + 8 * (r >> 1), col = h * 8 + 2 * (li & 3) + (r & 1);
    fin[(n * 16 + row) * 32 + col] = s;
  }
  __syncthreads();
  for (int n = 0; n < NT; n++) {
    const int e0 = esc_off[tile0 + n], e1 = esc_off[tile0 + n + 1];
    for (int j = e0 + (threadIdx.x >> 5); j < e1; j += NW) {
      const int2 ev = esc[j];
      const int k = ev.x & 0xFFFF, row = ev.x >> 16;
      const float d = __int_as_float(ev.y);
      if (lane < m) atomicAdd(&fin[(n * 16 + row) * 32 + lane], d * __bfloat162float(X[(size_t)lane * K + k]));
    }
  }
  __syncthreads();
  for (int item = threadIdx.x; item < NT * 16 * 32; item += NW * 32) {
    const int col = item & 31, row = (item >> 5) & 15, n = item >> 9;
    if (col < m) OUT[(size_t)col * N + (tile0 + n) * 16 + row] = __float2bfloat16(fin[(n * 16 + row) * 32 + col]);
  }
}

template <int NT, int NW, int NCH, int S>
static int launch(const void* Wc, const void* X, void* OUT, const int* esc_off, const void* esc, int N, int K, int m, uint32_t base4, cudaStream_t st) {
  const int ring = NW * S * NT * 3 * 512, red = (NW * NT * NCH * 4 * 32 + NT * 16 * 32) * 4;
  const int dyn = ring > red ? ring : red;
  cudaFuncSetAttribute(bf16c_gemm<NT, NW, NCH, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  bf16c_gemm<NT, NW, NCH, S><<<N / (16 * NT), NW * 32, dyn, st>>>(reinterpret_cast<const uint4*>(Wc), reinterpret_cast<const bf16*>(X),
      reinterpret_cast<bf16*>(OUT), esc_off, reinterpret_cast<const int2*>(esc), N, K, m, base4);
  return 0;
}

extern "C" int bf16c_run(const void* Wc, const void* X, void* OUT, const int* esc_off, const void* esc, int N, int K, int m, int base,
                         int nt, int nw, int stages, cudaStream_t st) {
  const uint32_t base4 = (uint32_t)base * 0x01010101u;
  const int nch = m <= 8 ? 1 : m <= 16 ? 2 : 4;
#define RUN(NT_, NW_, S_)                                                                                                         \
  if (nt == NT_ && nw == NW_ && stages == S_) {                                                                                   \
    if (nch == 1) return launch<NT_, NW_, 1, S_>(Wc, X, OUT, esc_off, esc, N, K, m, base4, st);                                  \
    if (nch == 2) return launch<NT_, NW_, 2, S_>(Wc, X, OUT, esc_off, esc, N, K, m, base4, st);                                  \
    return launch<NT_, NW_, 4, S_>(Wc, X, OUT, esc_off, esc, N, K, m, base4, st);                                                \
  }
  RUN(1, 4, 4) RUN(1, 4, 6) RUN(1, 8, 4) RUN(1, 8, 6) RUN(1, 16, 4) RUN(2, 4, 4) RUN(2, 8, 4) RUN(2, 8, 6) RUN(2, 4, 6) RUN(4, 4, 4) RUN(1, 16, 6) RUN(1, 16, 8) RUN(1, 8, 8) RUN(1, 32, 4) RUN(1, 32, 3) RUN(1, 32, 5) RUN(2, 16, 4) RUN(2, 16, 6)
#undef RUN
  return 1;
}
"""

TORCH_GLUE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int bf16c_run(const void*, const void*, void*, const int*, const void*, int, int, int, int, int, int, int, cudaStream_t);
void bf16c(torch::Tensor wc, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc, int64_t N, int64_t K, int64_t base,
           int64_t nt, int64_t nw, int64_t stages) {
  int r = bf16c_run(wc.data_ptr(), x.data_ptr(), out.data_ptr(), esc_off.data_ptr<int>(), esc.data_ptr(), (int)N, (int)K, (int)x.size(0),
                    (int)base, (int)nt, (int)nw, (int)stages, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported bf16c config");
}
"""
CPP_SRC = "void bf16c(torch::Tensor wc, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc, int64_t N, int64_t K, int64_t base, int64_t nt, int64_t nw, int64_t stages);"


def pack(w):
    """w: bf16 [N, K] cuda. Returns (wc int32 [N/16, K/64, 3, 32, 4], esc_off int32 [N/16+1], esc int32 [n_esc, 2], base int, stats)."""
    import torch

    N, K = w.shape
    assert N % 16 == 0 and K % 64 == 0
    RT, KCn = N // 16, K // 64
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign = bits >> 15
    e7 = (bits >> 8) & 0x7F
    lo = bits & 0xFF
    cnt = torch.bincount(e7.flatten(), minlength=128).cpu()
    best, base = -1, 0
    for b in range(0, 121):
        c = int(cnt[b:b + 8].sum())
        if c > best:
            best, base = c, b
    off = e7 - base
    inwin = (off >= 0) & (off < 8)
    offc = torch.where(inwin, off, torch.zeros_like(off))
    nib = (sign << 3) | offc
    # gather into [RT, KCn, 32 lanes, 4 ks, 8 values]
    lanes = torch.arange(32, device=w.device)
    g, t = lanes // 4, lanes % 4
    ks = torch.arange(4, device=w.device)
    vi = torch.arange(8, device=w.device)
    rowsel = g[:, None, None] + 8 * ((vi[None, None, :] // 2) % 2)  # v0,v1 -> g ; v2,v3 -> g+8 ; v4,v5 -> g ; v6,v7 -> g+8
    colsel = 16 * t[:, None, None] + 4 * ks[None, :, None] + (vi[None, None, :] % 2) + 2 * (vi[None, None, :] // 4)
    def gather(a):
        a4 = a.view(RT, 16, KCn, 64).permute(0, 2, 1, 3)  # [RT, KCn, 16, 64]
        return a4[:, :, rowsel, colsel]  # [RT, KCn, 32, 4, 8]
    lo_g, nib_g = gather(lo), gather(nib)
    lw = torch.zeros((RT, KCn, 32, 4, 2), dtype=torch.int64, device=w.device)
    for j in range(4):
        lw[..., 0] |= lo_g[..., j].to(torch.int64) << (8 * j)
        lw[..., 1] |= lo_g[..., 4 + j].to(torch.int64) << (8 * j)
    nw_ = torch.zeros((RT, KCn, 32, 4), dtype=torch.int64, device=w.device)
    for j in range(4):
        nw_ |= (nib_g[..., j].to(torch.int64) | (nib_g[..., 4 + j].to(torch.int64) << 4)) << (8 * j)
    def s32(x):
        return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)
    out = torch.empty((RT, KCn, 3, 32, 4), dtype=torch.int32, device=w.device)
    # plane0 = [ks0.w0, ks0.w1, ks1.w0, ks1.w1], plane1 = [ks2.w0, ks2.w1, ks3.w0, ks3.w1], plane2 = codes of ks0..3
    out[:, :, 0] = s32(lw[:, :, :, 0:2, :].reshape(RT, KCn, 32, 4))
    out[:, :, 1] = s32(lw[:, :, :, 2:4, :].reshape(RT, KCn, 32, 4))
    out[:, :, 2] = s32(nw_)
    # exact corrections for values whose exponent is outside the window
    bad = (~inwin).view(RT, 16, KCn, 64)
    idx = bad.nonzero()
    rt_i, r_i, kc_i, c_i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    tw = w.view(RT, 16, KCn, 64)[rt_i, r_i, kc_i, c_i].float()
    dec_bits = ((sign.view(RT, 16, KCn, 64)[rt_i, r_i, kc_i, c_i] << 7) | base) << 8 | lo.view(RT, 16, KCn, 64)[rt_i, r_i, kc_i, c_i]
    dec = (dec_bits.to(torch.int32) << 16).view(torch.float32)
    delta = tw - dec
    order = torch.argsort(rt_i, stable=True)
    rt_i, r_i, kc_i, c_i, delta = rt_i[order], r_i[order], kc_i[order], c_i[order], delta[order]
    esc = torch.stack(((kc_i * 64 + c_i + (r_i << 16)).to(torch.int32), delta.contiguous().view(torch.int32)), dim=1).contiguous()
    esc_off = torch.zeros(RT + 1, dtype=torch.int32, device=w.device)
    esc_off[1:] = torch.cumsum(torch.bincount(rt_i, minlength=RT), 0).to(torch.int32)
    return out, esc_off, esc, base, dict(escapes=int(idx.shape[0]), frac=float(idx.shape[0]) / (N * K))
