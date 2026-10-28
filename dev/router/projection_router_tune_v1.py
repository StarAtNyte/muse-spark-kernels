# okc: problem=muse_bf16_proj
"""Integrate measured dense singleton GEMV and transposed hi/lo router wins.

Small pooler batches use the transposed split-K path. Remaining dense batches
retain cuBLAS after measured Triton regressions. Weights retain their required
precision; the router alone uses its explicitly permitted BF16 hi/lo split.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


@triton.jit
def _matmul(X, W, LO, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            SPLIT: tl.constexpr, ROUTER: tl.constexpr, WN: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    s = tl.program_id(2)
    ki = s * (K // SPLIT) + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(K // SPLIT // BK):
        kk = ki + i * BK
        a = tl.load(X + mi[:, None] * K + kk[None, :], mi[:, None] < M, 0)
        b = tl.load(W + kk[:, None] * WN + ni[None, :], ni[None, :] < N, 0)
        acc = tl.dot(a, b, acc)
        if ROUTER:
            lo = tl.load(LO + kk[:, None] * WN + ni[None, :], ni[None, :] < N, 0)
            acc = tl.dot(a, lo, acc)
    tl.store(P + s * M * N + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _finish(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    sums = tl.full((B,), 0, tl.float32)
    for s in range(SPLIT):
        sums += tl.load(P + s * SIZE + idx, idx < SIZE, 0)
    tl.store(Y + idx, sums, idx < SIZE)


@triton.jit
def _gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    for start in range(K // BK):
        kk = start * BK + cols
        a = tl.load(X + kk).to(tl.float32)
        b = tl.load(W + rows[:, None] * K + kk[None, :],
                    rows[:, None] < N, 0).to(tl.float32)
        acc = tl.fma(b, a[None, :], acc)
    result = tl.sum(acc, 1)
    tl.store(Y + rows, result, rows < N)


def _prepare_v5(weights: Weights):
    state = dict(weights)
    hi = weights['router'].to(torch.bfloat16)
    state['router_hi'] = hi
    state['router_lo'] = (weights['router'] - hi.float()).to(torch.bfloat16)
    state['transposed'] = {name: state[name].T.contiguous()
                           for name in ('router_hi', 'router_lo', 'pooler')}
    return state


def _kernel_v5(data: Inputs, weights: Weights) -> torch.Tensor:
    """Return x @ W[:n].T, where W is weights[config["layer"]]: bfloat16, or float32 for the router."""
    x = data['x']
    layer = data['config']['layer']
    m, k = x.shape
    n = data['config']['n']
    router = layer == 'router'
    if m == 1 and not router:
        result = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _gemv[(triton.cdiv(n, 4),)](x, weights[layer], result, n, k, 4, 1024,
                                    num_warps=4)
        return result
    if not router and (layer != 'pooler' or m > 16):
        return x @ weights[layer][:n].T
    packed = weights['transposed']
    w = packed['router_hi'] if router else packed[layer]
    lo = packed['router_lo'] if router else w
    bm, bn, bk = (16 if m <= 32 else 32), 64, 128
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    split = min(16, triton.next_power_of_2(triton.cdiv(256, tiles)))
    result = torch.empty((m, n), device=x.device,
                         dtype=torch.float32 if router else torch.bfloat16)
    partial = (result if split == 1 else
               torch.empty((split, m, n), device=x.device, dtype=torch.float32))
    _matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        x, w, lo, partial, m, n, k, bm, bn, bk, split, router, w.shape[1],
        num_warps=4, num_stages=3)
    if split != 1:
        _finish[(triton.cdiv(m*n, 512),)](partial, result, m*n, split, 512)
    return result


_BC_CUDA = r'''
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
        for (int p = 0; p < 3; p++) cp16(&ring[(((c % S) * NT + n) * 3 + p) * 32 + lane], wsrc[n] + (size_t)c * 96 + p * 32);
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
    if (xp[h]) {
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
      if (xp[h] && i + 1 < CPW) {
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
        const uint32_t h0 = ((nb & 0x07070707u) + base4) | ((nb << 4) & 0x80808080u);
        const uint32_t h1 = (((nb >> 4) & 0x07070707u) + base4) | (nb & 0x80808080u);
        const uint32_t a0 = prmt(lw[2 * ks], h0, 0x5140u), a1 = prmt(lw[2 * ks], h0, 0x7362u);
        const uint32_t a2 = prmt(lw[2 * ks + 1], h1, 0x5140u), a3 = prmt(lw[2 * ks + 1], h1, 0x7362u);
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

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int bf16c_run(const void*, const void*, void*, const int*, const void*, int, int, int, int, int, int, int, cudaStream_t);
void bf16c(torch::Tensor wc, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc, int64_t N, int64_t K, int64_t base,
           int64_t nt, int64_t nw, int64_t stages) {
  int r = bf16c_run(wc.data_ptr(), x.data_ptr(), out.data_ptr(), esc_off.data_ptr<int>(), esc.data_ptr(), (int)N, (int)K, (int)x.size(0),
                    (int)base, (int)nt, (int)nw, (int)stages, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported bf16c config");
}
'''
_BC_CPP = r'''void bf16c(torch::Tensor wc, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc, int64_t N, int64_t K, int64_t base, int64_t nt, int64_t nw, int64_t stages);'''

def _bc_pack(w):
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



# ---- lossless 12-bit BF16 weights + swap-AB mma.sync GEMV for 1..32 tokens (own CUDA), chosen per (layer, m) by timing in prepare() ----
_BC_MOD = None


def _bc_module():
    global _BC_MOD
    if _BC_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _BC_MOD = load_inline("muse_bf16c", _BC_CPP, cuda_sources=_BC_CUDA, functions=["bf16c"], extra_cuda_cflags=["-O3"], verbose=False)
    return _BC_MOD


_BC_CFGS = [(1, 8, 4), (1, 8, 8), (1, 16, 4), (1, 16, 6), (1, 16, 8), (1, 32, 3), (1, 32, 4), (2, 8, 4), (2, 16, 4), (2, 16, 6), (1, 4, 4), (2, 4, 4)]
_BC_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')


def _bc_bucket(m):
    return 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 0


def _bc_run(state, layer, x, n, cfg):
    pk = state['bc'][layer]
    k = x.shape[1]
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    _bc_module().bf16c(pk['wc'], x, out, pk['esc_off'], pk['esc'], n, k, pk['base'], cfg[0], cfg[1], cfg[2])
    return out


def _bc_time(fn, flush, reps=12):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.sum()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def _prepare_dense(weights):
    state = _prepare_v5(weights)
    state['bc'] = {}
    state['bc_pick'] = {}
    try:
        mod = _bc_module()
        for layer in _BC_LAYERS:
            wc, esc_off, esc, base, _stats = _bc_pack(weights[layer])
            state['bc'][layer] = dict(wc=wc, esc_off=esc_off, esc=esc, base=base)
        flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
        for layer in _BC_LAYERS:
            n_full, k = weights[layer].shape
            for bucket in (1, 8, 16, 32):
                n = 1024 if (layer == 'unpooler' and bucket <= 8) else n_full
                if layer == 'unpooler' and bucket <= 8 and False:
                    continue
                x = torch.randn((bucket, k), device='cuda').to(torch.bfloat16)
                base_fn = lambda: _kernel_v5({'x': x, 'config': {'layer': layer, 'n': n}}, state)
                ref = x.float() @ weights[layer][:n].float().T
                t_base = _bc_time(base_fn, flush)
                best = (t_base * 0.97, None)
                for cfg in _BC_CFGS:
                    if (k // 64) % cfg[1] != 0 or n % (16 * cfg[0]) != 0:
                        continue
                    try:
                        o = _bc_run(state, layer, x, n, cfg)
                        torch.cuda.synchronize()
                        err = ((o.float() - ref).norm() / ref.norm()).item()
                        if err > 3e-3:
                            continue
                        t = _bc_time(lambda: _bc_run(state, layer, x, n, cfg), flush)
                    except Exception:
                        continue
                    if t < best[0]:
                        best = (t, cfg)
                state['bc_pick'][(layer, bucket)] = best[1]
                print('BCPICK', layer, bucket, best[1], 'base=%.1f best=%.1f' % (t_base, best[0]), flush=True)
    except Exception as ex:  # any failure: stock v5 paths only
        print('BC disabled:', repr(ex)[:200], flush=True)
        state['bc_pick'] = {}
    return state


def _kernel_dense(data, weights):
    x = data['x']
    layer = data['config']['layer']
    if layer != 'router' and weights['bc_pick']:
        m = x.shape[0]
        b = _bc_bucket(m)
        cfg = weights['bc_pick'].get((layer, b)) if b else None
        if cfg is not None and x.dtype == torch.bfloat16 and x.is_contiguous():
            return _bc_run(weights, layer, x, data['config']['n'], cfg)
    return _kernel_v5(data, weights)


# Tune only weight-backed synthetic inputs; runtime activations stay in kernel_fn.
# Each tuple is BM, BN, BK, split-K. Include the original dispatch as fallback.
_ROUTER_CFGS = [(16, 16, 128, s) for s in (4, 8, 16)] + [
    (16, 32, 128, s) for s in (4, 8, 16)] + [
    (16, 64, 128, s) for s in (4, 8, 16)] + [
    (32, 32, 128, 8), (32, 32, 128, 16), (32, 64, 128, 16)]


def _router_run(x, state, n, cfg):
    m, k = x.shape
    bm, bn, bk, split = cfg
    w = state['transposed']['router_hi']
    lo = state['transposed']['router_lo']
    result = torch.empty((m, n), device=x.device, dtype=torch.float32)
    partial = result if split == 1 else torch.empty(
        (split, m, n), device=x.device, dtype=torch.float32)
    _matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        x, w, lo, partial, m, n, k, bm, bn, bk, split, True, w.shape[1],
        num_warps=4, num_stages=3)
    if split != 1:
        _finish[(triton.cdiv(m*n, 512),)](partial, result, m*n, split, 512)
    return result


def prepare(weights):
    state = _prepare_dense(weights)
    state['router_pick'] = {}
    flush = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32,
                        device=weights['router'].device)
    k = weights['router'].shape[1]
    n = weights['router'].shape[0]
    for m in (1, 8, 16, 32, 128, 256):
        x = torch.randn((m, k), device=weights['router'].device,
                        dtype=torch.bfloat16)
        data = {'x': x, 'config': {'layer': 'router', 'n': n}}
        ref = x.float() @ weights['router'].T
        base = _bc_time(lambda: _kernel_v5(data, state), flush, reps=8)
        best = (base * 0.97, None)
        for cfg in _ROUTER_CFGS:
            try:
                out = _router_run(x, state, n, cfg)
                err = ((out - ref).norm() / ref.norm()).item()
                if not torch.isfinite(out).all().item() or err > 1e-4:
                    continue
                t = _bc_time(lambda: _router_run(x, state, n, cfg), flush, reps=8)
                if t < best[0]:
                    best = (t, cfg)
            except (RuntimeError, triton.CompilationError):
                continue
        state['router_pick'][m] = best[1]
        print('ROUTERPICK', m, best[1], 'base=%.2f best=%.2f' % (base, best[0]), flush=True)
    return state


def kernel_fn(data, state):
    x = data['x']
    if data['config']['layer'] == 'router':
        cfg = state['router_pick'].get(x.shape[0])
        if cfg is not None:
            return _router_run(x, state, data['config']['n'], cfg)
    return _kernel_dense(data, state)
