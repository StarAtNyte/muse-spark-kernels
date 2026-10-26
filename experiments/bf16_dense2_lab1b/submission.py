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


def _old_prepare(weights: Weights):
    state = dict(weights)
    hi = weights['router'].to(torch.bfloat16)
    state['router_hi'] = hi
    state['router_lo'] = (weights['router'] - hi.float()).to(torch.bfloat16)
    state['transposed'] = {name: state[name].T.contiguous()
                           for name in ('router_hi', 'router_lo', 'pooler')}
    return state


def _old_kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:
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


# ---- weight-streaming dense GEMM (CUDA) ----
_DENSE_CUDA = r'''
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

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
__device__ __forceinline__ void l2_prefetch_bulk(const void* p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
#endif
}
static int g_pdl = 0;
static int g_pf = 0;
static int g_dbg = 0;
extern "C" void dense_set_dbg(int v) { g_dbg = v; }
extern "C" void dense_set_pf(int v) { g_pf = v; }
extern "C" void dense_set_pdl(int v) { g_pdl = v; }
template <typename... KArgs, typename... Args>
static void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, bool pdl, Args... args) {
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

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc));
}
__device__ __forceinline__ void cp16z(void* sdst, const void* gsrc, uint32_t nbytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc), "r"(nbytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// W tiles: uint4 [rt][kc][ks][lane].  grid = (N/16/NTB, KSPLIT).  Each warp covers CPW consecutive k64 chunks.
// part: fp32 [KSPLIT][M][N] (or final bf16 [M][N] when KSPLIT == 1 and FINAL).
template <int NTB, int NCH, int NW, int S, int OM>
__global__ void __launch_bounds__(NW * 32)
dense_kernel(const uint4* __restrict__ Wq, const bf16* __restrict__ X, float* __restrict__ part, int M, int N, int K,
             int ksplit, int* __restrict__ cnt, void* __restrict__ outp, int pair, int nout, int out_fp32, int pf, int dbg) {
  extern __shared__ uint4 smem[];
  pdl_wait();
  pdl_trigger();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kcs = K / 64;                   // chunks along K
  const int cpw = kcs / (ksplit * NW);      // chunks per warp
  const int c0 = (blockIdx.y * NW + warp) * cpw;
  const int tile0 = blockIdx.x * NTB;

  // token chunks are processed in passes of NCH*8 tokens
  for (int tok0 = 0; tok0 < M; tok0 += NCH * 8) {
    const bf16* xp[NCH];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      const int tok = tok0 + h * 8 + g;
      xp[h] = tok < M ? X + (size_t)tok * K + 16 * t + c0 * 64 : nullptr;
    }
    float acc[NTB][NCH][4];
#pragma unroll
    for (int n = 0; n < NTB; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc[n][h][r] = 0.f;

    const uint4* wsrc[NTB];
#pragma unroll
    for (int n = 0; n < NTB; n++) wsrc[n] = Wq + ((size_t)(tile0 + n) * kcs + c0) * 128 + lane;
    constexpr int STG = NTB * 4 + NCH * 2;  // uint4 slots per lane per stage: weights, then activations
    uint4* ring = smem + warp * (S * STG * 32);
    if (pf && lane == 0 && tok0 == 0) {
#pragma unroll
      for (int n = 0; n < NTB; n++) l2_prefetch_bulk(wsrc[n] - lane, (uint32_t)cpw * 2048u);
    }

    auto issue = [&](int c) {
      if (c < cpw) {
#pragma unroll
        for (int n = 0; n < NTB; n++)
#pragma unroll
          for (int ks = 0; ks < 4; ks++) { if (!(dbg & 1)) cp16(&ring[((c % S) * STG + n * 4 + ks) * 32 + lane], wsrc[n] + c * 128 + ks * 32); }
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const bool ok = xp[h] != nullptr;
          const bf16* src = ok ? xp[h] + c * 64 : X;
          cp16z(&ring[((c % S) * STG + NTB * 4 + 2 * h) * 32 + lane], src, ok ? 16u : 0u);
          cp16z(&ring[((c % S) * STG + NTB * 4 + 2 * h + 1) * 32 + lane], src + 8, ok ? 16u : 0u);
        }
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);

#pragma unroll 1
    for (int i = 0; i < cpw; i++) {
      cp_wait<S - 2>();
      issue(i + S - 1);
#pragma unroll
      for (int n = 0; n < NTB; n++) {
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint4 a = ring[((i % S) * STG + n * 4 + ks) * 32 + lane];
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = ring[((i % S) * STG + NTB * 4 + 2 * h + (ks >> 1)) * 32 + lane];
            if (dbg & 2) acc[n][h][0] += __uint_as_float(a.x ^ xs.x); else mma16816(acc[n][h], a.x, a.y, a.z, a.w, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
          }
        }
      }
    }
    cp_wait<0>();
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int n = 0; n < NTB; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) red[(((warp * NTB + n) * NCH + h) * 4 + r) * 32 + lane] = acc[n][h][r];
    __syncthreads();
    for (int item = threadIdx.x; item < NTB * NCH * 4 * 32; item += NW * 32) {
      const int li = item & 31, r = (item >> 5) & 3, h = (item >> 7) % NCH, n = (item >> 7) / NCH;
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) s += red[(((w * NTB + n) * NCH + h) * 4 + r) * 32 + li];
      const int row = (tile0 + n) * 16 + (li >> 2) + 8 * (r >> 1);
      const int tok = tok0 + h * 8 + 2 * (li & 3) + (r & 1);
      if (tok < M) {
        if constexpr (OM == 1) reinterpret_cast<bf16*>(part)[(size_t)tok * N + row] = __float2bfloat16(s);
        else part[((size_t)blockIdx.y * M + tok) * N + row] = s;
      }
    }
    if constexpr (OM == 2) {
      // last-arriving CTA of this output tile reduces all K-split partials (and router hi/lo halves)
      __shared__ int s_last;
      __threadfence();
      __syncthreads();
      const int ntile_out = pair ? (N / 2) / (16 * NTB) : N / (16 * NTB);
      const int slot = (blockIdx.x % ntile_out) * ((M + NCH * 8 - 1) / (NCH * 8)) + tok0 / (NCH * 8);
      if (threadIdx.x == 0) {
        const int old = atomicAdd(&cnt[slot], 1);
        s_last = (old == ksplit * (pair ? 2 : 1) - 1);
      }
      __syncthreads();
      if (s_last) {
        __threadfence();
        const int obase = (blockIdx.x % ntile_out) * NTB * 16;
        const int tn = min(NCH * 8, M - tok0);
        for (int idx = threadIdx.x; idx < NTB * 16 * tn; idx += NW * 32) {
          const int r = idx % (NTB * 16), tk = idx / (NTB * 16);
          float acc = 0.f;
          for (int k = 0; k < ksplit; k++) {
            const float* p = part + ((size_t)k * M + tok0 + tk) * N + obase + r;
            acc += __ldcg(p);
            if (pair) acc += __ldcg(p + nout);
          }
          if (out_fp32) reinterpret_cast<float*>(outp)[(size_t)(tok0 + tk) * nout + obase + r] = acc;
          else reinterpret_cast<bf16*>(outp)[(size_t)(tok0 + tk) * nout + obase + r] = __float2bfloat16(acc);
        }
        if (threadIdx.x == 0) cnt[slot] = 0;
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

// out[m][n] = sum_ks part[ks][m][n] (+ part[..][m][n + N] when pair: router hi/lo), as bf16 or fp32.
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, int M, int NOUT, int NTOT,
                              int ksplit, int pair, int out_fp32) {
  pdl_wait();
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * NOUT) return;
  const int m = idx / NOUT, n = idx % NOUT;
  float s = 0.f;
  for (int k = 0; k < ksplit; k++) {
    s += part[((size_t)k * M + m) * NTOT + n];
    if (pair) s += part[((size_t)k * M + m) * NTOT + n + NOUT];
  }
  if (out_fp32) reinterpret_cast<float*>(out)[idx] = s;
  else reinterpret_cast<bf16*>(out)[idx] = __float2bfloat16(s);
}

template <int NTB, int NCH, int NW, int S, int OM>
static int launch(const void* W, const void* X, float* part, int M, int N, int K, int ksplit, int* cnt, void* outp,
                  int pair, int nout, int out_fp32, cudaStream_t st) {
  dim3 grid(N / 16 / NTB, ksplit);
  const int dyn = NW * S * (NTB * 4 + NCH * 2) * 512;
  cudaFuncSetAttribute(dense_kernel<NTB, NCH, NW, S, OM>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(dense_kernel<NTB, NCH, NW, S, OM>, grid, dim3(NW * 32), (size_t)dyn, st, g_pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const bf16*>(X), part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, g_pf, g_dbg);
  return 0;
}

extern "C" int dense_run(int ntb, int nch, int nw, int s, int om, const void* W, const void* X, float* part, int M, int N,
                         int K, int ksplit, int* cnt, void* outp, int pair, int nout, int out_fp32, cudaStream_t st) {
#define D(NTB_, NCH_, NW_, S_) if (ntb == NTB_ && nch == NCH_ && nw == NW_ && s == S_) { if (om == 1) return launch<NTB_, NCH_, NW_, S_, 1>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); if (om == 2) return launch<NTB_, NCH_, NW_, S_, 2>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); return launch<NTB_, NCH_, NW_, S_, 0>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); }
  D(1,1,2,4) D(1,1,4,4) D(1,1,8,4) D(1,2,2,4) D(1,2,4,4) D(1,2,8,4) D(1,4,2,4) D(1,4,4,4) D(1,4,8,4) D(2,1,2,4) D(2,1,4,4) D(2,1,8,4) D(2,2,2,4) D(2,2,4,4) D(2,2,8,4) D(2,4,2,4) D(2,4,4,4) D(2,4,8,4) D(1,8,4,4) D(1,8,8,4) D(1,1,16,4) D(1,2,16,4) D(1,4,16,4) D(2,1,16,4)
#undef D
  return 1;
}

__global__ void probe_kernel(const uint4* __restrict__ p, size_t n, float* __restrict__ out) {
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
#pragma unroll 8
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint4 v = __ldcs(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) out[0] = 1.f;
}
extern "C" void probe_run(const void* p, size_t bytes, float* out, int blocks, int threads, cudaStream_t st) {
  probe_kernel<<<blocks, threads, 0, st>>>(reinterpret_cast<const uint4*>(p), bytes / 16, out);
}


// ---- lossless exponent-compressed weights -------------------------------------------------------------------
// Per (rt, kc, lane): 3 uint4 slots = sign|mant bytes of ks0..3 (2 slots) + 4 regs of 4-bit exponent codes (code = emax - exp).
// Elements whose exponent is below emax-15 are "escapes": they decode to code 15 and are corrected from a sparse list.
__device__ __forceinline__ uint32_t dec_reg(uint32_t m, uint32_t e, uint32_t base2, int r) {
  const uint32_t t = __byte_perm(m, 0, (r & 1) ? 0x3322 : 0x1100);
  uint32_t c;
  if (r == 0) c = (e << 7) & 0x07800780u;
  else if (r == 1) c = (e << 3) & 0x07800780u;
  else if (r == 2) c = (e >> 1) & 0x07800780u;
  else c = (e >> 5) & 0x07800780u;
  const uint32_t ex = base2 - c;
  uint32_t o;
  asm("lop3.b32 %0, %1, %2, %3, 0xea;\n" : "=r"(o) : "r"(t), "r"(0x807F807Fu), "r"(ex));  // (t & mask) | ex
  return o;
}

template <int NTB, int NCH, int NW, int S, int OM>
__global__ void __launch_bounds__(NW * 32)
dense_ckernel(const uint4* __restrict__ Wq, const bf16* __restrict__ X, float* __restrict__ part, int M, int N, int K,
              int ksplit, int* __restrict__ cnt, void* __restrict__ outp, int pair, int nout, int out_fp32, int pf,
              int emax, const int* __restrict__ eoff, const int4* __restrict__ esc) {
  extern __shared__ uint4 smem[];
  pdl_wait();
  pdl_trigger();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kcs = K / 64;
  const int cpw = kcs / (ksplit * NW);
  const int c0 = (blockIdx.y * NW + warp) * cpw;
  const int tile0 = blockIdx.x * NTB;
  const int kbeg = blockIdx.y * NW * cpw * 64, kend = kbeg + NW * cpw * 64;
  const uint32_t base2 = (uint32_t)(emax << 7) * 0x00010001u;

  for (int tok0 = 0; tok0 < M; tok0 += NCH * 8) {
    const bf16* xp[NCH];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      const int tok = tok0 + h * 8 + g;
      xp[h] = tok < M ? X + (size_t)tok * K + 16 * t + c0 * 64 : nullptr;
    }
    float acc[NTB][NCH][4];
#pragma unroll
    for (int n = 0; n < NTB; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc[n][h][r] = 0.f;

    const uint4* wsrc[NTB];
#pragma unroll
    for (int n = 0; n < NTB; n++) wsrc[n] = Wq + ((size_t)(tile0 + n) * kcs + c0) * 96 + lane;
    constexpr int STG = NTB * 3 + NCH * 2;
    uint4* ring = smem + warp * (S * STG * 32);
    if (pf && lane == 0 && tok0 == 0) {
#pragma unroll
      for (int n = 0; n < NTB; n++) l2_prefetch_bulk(wsrc[n] - lane, (uint32_t)cpw * 1536u);
    }

    auto issue = [&](int c) {
      if (c < cpw) {
#pragma unroll
        for (int n = 0; n < NTB; n++)
#pragma unroll
          for (int sl = 0; sl < 3; sl++) cp16(&ring[((c % S) * STG + n * 3 + sl) * 32 + lane], wsrc[n] + c * 96 + sl * 32);
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const bool ok = xp[h] != nullptr;
          const bf16* src = ok ? xp[h] + c * 64 : X;
          cp16z(&ring[((c % S) * STG + NTB * 3 + 2 * h) * 32 + lane], src, ok ? 16u : 0u);
          cp16z(&ring[((c % S) * STG + NTB * 3 + 2 * h + 1) * 32 + lane], src + 8, ok ? 16u : 0u);
        }
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);

#pragma unroll 1
    for (int i = 0; i < cpw; i++) {
      cp_wait<S - 2>();
      issue(i + S - 1);
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        const uint4 s0 = ring[((i % S) * STG + n * 3 + 0) * 32 + lane];
        const uint4 s1 = ring[((i % S) * STG + n * 3 + 1) * 32 + lane];
        const uint4 s2 = ring[((i % S) * STG + n * 3 + 2) * 32 + lane];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint32_t m0 = ks == 0 ? s0.x : ks == 1 ? s0.z : ks == 2 ? s1.x : s1.z;
          const uint32_t m1 = ks == 0 ? s0.y : ks == 1 ? s0.w : ks == 2 ? s1.y : s1.w;
          const uint32_t e = ks == 0 ? s2.x : ks == 1 ? s2.y : ks == 2 ? s2.z : s2.w;
          const uint32_t a0 = dec_reg(m0, e, base2, 0), a1 = dec_reg(m0, e, base2, 1);
          const uint32_t a2 = dec_reg(m1, e, base2, 2), a3 = dec_reg(m1, e, base2, 3);
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = ring[((i % S) * STG + NTB * 3 + 2 * h + (ks >> 1)) * 32 + lane];
            mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
          }
        }
      }
    }
    cp_wait<0>();
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int n = 0; n < NTB; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) red[(((warp * NTB + n) * NCH + h) * 4 + r) * 32 + lane] = acc[n][h][r];
    __syncthreads();
    for (int item = threadIdx.x; item < NTB * NCH * 4 * 32; item += NW * 32) {
      const int li = item & 31, r = (item >> 5) & 3, h = (item >> 7) % NCH, n = (item >> 7) / NCH;
      float sacc = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) sacc += red[(((w * NTB + n) * NCH + h) * 4 + r) * 32 + li];
      const int row = (tile0 + n) * 16 + (li >> 2) + 8 * (r >> 1);
      const int tok = tok0 + h * 8 + 2 * (li & 3) + (r & 1);
      if (tok < M) {
        for (int q = eoff[tile0 + n]; q < eoff[tile0 + n + 1]; q++) {
          const int4 en = esc[q];
          if (en.x == row && en.y >= kbeg && en.y < kend) sacc += __int_as_float(en.z) * __bfloat162float(X[(size_t)tok * K + en.y]);
        }
        if constexpr (OM == 1) reinterpret_cast<bf16*>(part)[(size_t)tok * N + row] = __float2bfloat16(sacc);
        else part[((size_t)blockIdx.y * M + tok) * N + row] = sacc;
      }
    }
    if constexpr (OM == 2) {
      __shared__ int s_last;
      __threadfence();
      __syncthreads();
      const int ntile_out = N / (16 * NTB);
      const int slot = (blockIdx.x % ntile_out) * ((M + NCH * 8 - 1) / (NCH * 8)) + tok0 / (NCH * 8);
      if (threadIdx.x == 0) {
        const int old = atomicAdd(&cnt[slot], 1);
        s_last = (old == ksplit - 1);
      }
      __syncthreads();
      if (s_last) {
        __threadfence();
        const int obase = (blockIdx.x % ntile_out) * NTB * 16;
        const int tn = min(NCH * 8, M - tok0);
        for (int idx = threadIdx.x; idx < NTB * 16 * tn; idx += NW * 32) {
          const int r = idx % (NTB * 16), tk = idx / (NTB * 16);
          float a = 0.f;
          for (int k = 0; k < ksplit; k++) a += __ldcg(part + ((size_t)k * M + tok0 + tk) * N + obase + r);
          reinterpret_cast<bf16*>(outp)[(size_t)(tok0 + tk) * nout + obase + r] = __float2bfloat16(a);
        }
        if (threadIdx.x == 0) cnt[slot] = 0;
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

template <int NTB, int NCH, int NW, int S, int OM>
static int launch_c(const void* W, const void* X, float* part, int M, int N, int K, int ksplit, int* cnt, void* outp,
                    int nout, int emax, const int* eoff, const int4* esc, cudaStream_t st) {
  dim3 grid(N / 16 / NTB, ksplit);
  const int dyn = NW * S * (NTB * 3 + NCH * 2) * 512;
  const int dyn2 = dyn > NW * NTB * NCH * 4 * 128 ? dyn : NW * NTB * NCH * 4 * 128;
  cudaFuncSetAttribute(dense_ckernel<NTB, NCH, NW, S, OM>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn2);
  launch_k(dense_ckernel<NTB, NCH, NW, S, OM>, grid, dim3(NW * 32), (size_t)dyn2, st, g_pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const bf16*>(X), part, M, N, K, ksplit, cnt, outp, 0, nout, 0, g_pf, emax, eoff, esc);
  return 0;
}

extern "C" int dense_crun(int ntb, int nch, int nw, int s, int om, const void* W, const void* X, float* part, int M, int N,
                          int K, int ksplit, int* cnt, void* outp, int nout, int emax, const int* eoff, const void* esc, cudaStream_t st) {
#define DC(NTB_, NCH_, NW_, S_) if (ntb == NTB_ && nch == NCH_ && nw == NW_ && s == S_) { if (om == 1) return launch_c<NTB_, NCH_, NW_, S_, 1>(W, X, part, M, N, K, ksplit, cnt, outp, nout, emax, eoff, (const int4*)esc, st); if (om == 2) return launch_c<NTB_, NCH_, NW_, S_, 2>(W, X, part, M, N, K, ksplit, cnt, outp, nout, emax, eoff, (const int4*)esc, st); return launch_c<NTB_, NCH_, NW_, S_, 0>(W, X, part, M, N, K, ksplit, cnt, outp, nout, emax, eoff, (const int4*)esc, st); }
  DC(1,1,2,4) DC(1,1,4,4) DC(1,1,8,4) DC(1,2,2,4) DC(1,2,4,4) DC(1,2,8,4) DC(1,4,2,4) DC(1,4,4,4) DC(1,4,8,4) DC(2,1,2,4) DC(2,1,4,4) DC(2,1,8,4) DC(2,2,2,4) DC(2,2,4,4) DC(2,2,8,4) DC(2,4,2,4) DC(2,4,4,4) DC(2,4,8,4) DC(1,8,4,4) DC(1,8,8,4) DC(1,1,16,4) DC(1,2,16,4) DC(1,4,16,4) DC(2,1,16,4)
#undef DC
  return 1;
}


// ---- dense2: x slice staged once in shared memory, K split across a thread-block cluster (DSMEM reduction) ----
// grid (groups, KS), cluster (1, KS, 1).  Warp w owns NTB 16-row tiles of its CTA's row group and streams its weights for the
// CTA's K slice through a lane-private cp.async ring; the KS CTAs of a cluster each hold a K slice of the same rows and add their
// fp32 partial tiles through distributed shared memory.  Output bf16 [M, N] written directly.
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <int NW, int NTB, int NCH, int S, int KS>
__global__ void __launch_bounds__(NW * 32)
dense2_kernel(const uint4* __restrict__ Wq, const bf16* __restrict__ X, bf16* __restrict__ OUT, int M, int N, int K) {
  extern __shared__ __align__(16) uint8_t smem2[];
  cg::cluster_group cluster = cg::this_cluster();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rank = blockIdx.y;
  const int kcs = K / 64;
  const int kc_per = kcs / KS;
  const int Kc = K / KS;                       // elements in this CTA's K slice
  const int xp_pitch = Kc * 2 + 16;            // bytes per token row in smem
  constexpr int R = NW * NTB * 16;             // rows of the CTA's group
  uint8_t* xs = smem2;
  const size_t x_bytes = (size_t)NCH * 8 * xp_pitch;
  uint4* ring = reinterpret_cast<uint4*>(smem2 + x_bytes) + (size_t)warp * (S * NTB * 4 * 32);
  const int tile0 = blockIdx.x * NW * NTB + warp * NTB;

  const uint4* wsrc[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) wsrc[n] = Wq + ((size_t)(tile0 + n) * kcs + rank * kc_per) * 128 + lane;
  auto issue = [&](int c) {
    if (c < kc_per) {
#pragma unroll
      for (int n = 0; n < NTB; n++)
#pragma unroll
        for (int ks = 0; ks < 4; ks++) cp16(&ring[((c % S) * NTB * 4 + n * 4 + ks) * 32 + lane], wsrc[n] + c * 128 + ks * 32);
    }
    cp_commit();
  };
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);

  // weights are static: their prefetch is issued before waiting for the producer of x
  pdl_wait();
  pdl_trigger();
  // stage this CTA's x slice: [NCH*8 tokens][Kc] bf16 (rows >= M are zero)
  {
    const int vec_per_row = Kc / 8;  // 16-byte vectors per token row
    for (int idx = threadIdx.x; idx < NCH * 8 * vec_per_row; idx += NW * 32) {
      const int tok = idx / vec_per_row, v = idx % vec_per_row;
      const bool ok = tok < M;
      const bf16* src = ok ? X + (size_t)tok * K + (size_t)rank * Kc + v * 8 : X;
      cp16z(xs + (size_t)tok * xp_pitch + v * 16, src, ok ? 16u : 0u);
    }
    cp_commit();
  }
  cp_wait<0>();
  __syncthreads();

  float acc[NTB][NCH][4];
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][h][r] = 0.f;

  // all S-1 prefetch groups plus the x group are complete after the wait above; refill the pipeline normally from here
#pragma unroll 1
  for (int i = 0; i < kc_per; i++) {
    uint4 xr[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      const uint8_t* xrow = xs + (size_t)(h * 8 + g) * xp_pitch + (size_t)(i * 64 + 16 * t) * 2;
      xr[h][0] = *reinterpret_cast<const uint4*>(xrow);
      xr[h][1] = *reinterpret_cast<const uint4*>(xrow + 16);
    }
    issue(i + S - 1);
#pragma unroll
    for (int n = 0; n < NTB; n++) {
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        const uint4 a = ring[((i % S) * NTB * 4 + n * 4 + ks) * 32 + lane];
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const uint4 xv = xr[h][ks >> 1];
          mma16816(acc[n][h], a.x, a.y, a.z, a.w, (ks & 1) ? xv.z : xv.x, (ks & 1) ? xv.w : xv.y);
        }
      }
    }
    cp_wait<S - 2>();
  }
  cp_wait<0>();
  __syncthreads();  // everyone is done with x; reuse its space for the fp32 partial tile

  float* red = reinterpret_cast<float*>(smem2);   // [NCH*8][R + 4]
  const int RP = R + 4;
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int row = (warp * NTB + n) * 16 + g + 8 * (r >> 1);
        const int tok = h * 8 + 2 * t + (r & 1);
        red[tok * RP + row] = acc[n][h][r];
      }
  cluster.sync();
  // this CTA reduces R / KS rows (all tokens) over the KS partial tiles
  constexpr int RS = R / KS;
  const int row0 = rank * RS;
  const int n0 = blockIdx.x * R;
  for (int idx = threadIdx.x; idx < M * (RS / 2); idx += NW * 32) {
    const int tok = idx / (RS / 2), rr = (idx % (RS / 2)) * 2;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int src = 0; src < KS; src++) {
      const float* peer = cluster.map_shared_rank(red, src);
      s0 += peer[tok * RP + row0 + rr];
      s1 += peer[tok * RP + row0 + rr + 1];
    }
    *reinterpret_cast<__nv_bfloat162*>(OUT + (size_t)tok * N + n0 + row0 + rr) = __floats2bfloat162_rn(s0, s1);
  }
  cluster.sync();
}

template <int NW, int NTB, int NCH, int S, int KS>
static int launch2c(const void* W, const void* X, void* OUT, int M, int N, int K, cudaStream_t st) {
  const int Kc = K / KS;
  constexpr int R = NW * NTB * 16;
  if ((K / 64) % KS || N % R || R % (KS * 2) || (Kc % 8)) return 1;
  const size_t x_bytes = (size_t)NCH * 8 * (Kc * 2 + 16);
  const size_t ring_bytes = (size_t)NW * S * NTB * 4 * 32 * 16;
  size_t dyn = x_bytes + ring_bytes;
  const size_t red_bytes = (size_t)NCH * 8 * (R + 4) * 4;
  if (red_bytes > dyn) dyn = red_bytes;
  if (dyn > 227 * 1024) return 1;
  cudaFuncSetAttribute(dense2_kernel<NW, NTB, NCH, S, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / R, KS);
  cfg.blockDim = dim3(NW * 32);
  cfg.dynamicSmemBytes = dyn;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = KS;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl ? 2 : 1;
  cudaLaunchKernelEx(&cfg, dense2_kernel<NW, NTB, NCH, S, KS>, reinterpret_cast<const uint4*>(W), reinterpret_cast<const bf16*>(X),
                     reinterpret_cast<bf16*>(OUT), M, N, K);
  return 0;
}

extern "C" int dense2_run(int nw, int ntb, int nch, int s, int ks, const void* W, const void* X, void* OUT, int M, int N, int K,
                          cudaStream_t st) {
#define D2(NW_, NTB_, NCH_, S_, KS_) if (nw == NW_ && ntb == NTB_ && nch == NCH_ && s == S_ && ks == KS_) return launch2c<NW_, NTB_, NCH_, S_, KS_>(W, X, OUT, M, N, K, st);
#define D2K(NW_, NTB_, NCH_, S_) D2(NW_, NTB_, NCH_, S_, 2) D2(NW_, NTB_, NCH_, S_, 4) D2(NW_, NTB_, NCH_, S_, 8)
#define D2S(NW_, NTB_, NCH_) D2K(NW_, NTB_, NCH_, 4) D2K(NW_, NTB_, NCH_, 6)
#define D2N(NW_, NTB_) D2S(NW_, NTB_, 1) D2S(NW_, NTB_, 2) D2S(NW_, NTB_, 4)
  D2N(8, 1) D2N(4, 1) D2N(4, 2)
#undef D2N
#undef D2S
#undef D2K
#undef D2
  return 2;
}

extern "C" void dense_reduce(const float* part, void* out, int M, int nout, int ntot, int ksplit, int pair, int fp32,
                             cudaStream_t st) {
  const int n = M * nout;
  launch_k(reduce_kernel, dim3((n + 255) / 256), dim3(256), 0, st, g_pdl != 0, part, out, M, nout, ntot, ksplit, pair, fp32);
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int dense_run(int, int, int, int, int, const void*, const void*, float*, int, int, int, int, int*, void*, int, int, int, cudaStream_t);
extern "C" void dense_reduce(const float*, void*, int, int, int, int, int, int, cudaStream_t);
extern "C" void dense_set_pdl(int);
extern "C" void dense_set_pf(int);
extern "C" void dense_set_dbg(int);
extern "C" int dense2_run(int, int, int, int, int, const void*, const void*, void*, int, int, int, cudaStream_t);
void dense_set_dbg_t(int64_t v) { dense_set_dbg((int)v); }
extern "C" int dense_crun(int, int, int, int, int, const void*, const void*, float*, int, int, int, int, int*, void*, int, int, const int*, const void*, cudaStream_t);
extern "C" void probe_run(const void*, size_t, float*, int, int, cudaStream_t);
void probe_t(torch::Tensor p, torch::Tensor out, int64_t blocks, int64_t threads) {
  probe_run(p.data_ptr(), (size_t)p.numel() * p.element_size(), out.data_ptr<float>(), (int)blocks, (int)threads, at::cuda::getCurrentCUDAStream());
}
void dense_set_pdl_t(int64_t v) { dense_set_pdl((int)v); }
void dense_set_pf_t(int64_t v) { dense_set_pf((int)v); }

void dense_cgemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                 torch::Tensor out, int64_t ksplit, int64_t nout, torch::Tensor cnt, int64_t fix, int64_t emax, torch::Tensor eoff,
                 torch::Tensor esc) {
  const int M = X.size(0), K = X.size(1);
  const int N = (int)nout;
  const bool direct = ksplit == 1;
  const int om = direct ? 1 : (fix ? 2 : 0);
  int r = dense_crun((int)ntb, (int)nch, (int)nw, (int)s, om, W.data_ptr(), X.data_ptr(),
                     direct ? reinterpret_cast<float*>(out.data_ptr()) : part.data_ptr<float>(), M, N, K, (int)ksplit,
                     cnt.numel() ? cnt.data_ptr<int>() : nullptr, out.data_ptr(), (int)nout, (int)emax, eoff.data_ptr<int>(),
                     esc.data_ptr(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported dense config");
  if (om == 0) dense_reduce(part.data_ptr<float>(), out.data_ptr(), M, (int)nout, N, (int)ksplit, 0, 0, at::cuda::getCurrentCUDAStream());
}

void dense2_gemm(int64_t nw, int64_t ntb, int64_t nch, int64_t s, int64_t ks, torch::Tensor W, torch::Tensor X, torch::Tensor out) {
  int r = dense2_run((int)nw, (int)ntb, (int)nch, (int)s, (int)ks, W.data_ptr(), X.data_ptr(), out.data_ptr(), (int)X.size(0), (int)out.size(1), (int)X.size(1),
                     at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported dense2 config");
}

void dense_gemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                torch::Tensor out, int64_t ksplit, int64_t pair, int64_t nout, torch::Tensor cnt, int64_t fix) {
  const int M = X.size(0), K = X.size(1);
  const int N = (int)(pair ? 2 * nout : nout);
  const bool fp32 = out.scalar_type() == torch::kFloat32;
  const bool direct = ksplit == 1 && !pair && !fp32;
  const int om = direct ? 1 : (fix && ksplit > 1 ? 2 : 0);
  int r = dense_run((int)ntb, (int)nch, (int)nw, (int)s, om, W.data_ptr(), X.data_ptr(),
                    direct ? reinterpret_cast<float*>(out.data_ptr()) : part.data_ptr<float>(), M, N, K, (int)ksplit,
                    cnt.numel() ? cnt.data_ptr<int>() : nullptr, out.data_ptr(), (int)pair, (int)nout, fp32 ? 1 : 0,
                    at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported dense config");
  if (om == 0)
    dense_reduce(part.data_ptr<float>(), out.data_ptr(), M, (int)nout, N, (int)ksplit, (int)pair, fp32,
                 at::cuda::getCurrentCUDAStream());
}
'''
_DENSE_CPP = r'''
void probe_t(torch::Tensor p, torch::Tensor out, int64_t blocks, int64_t threads);
void dense_set_pdl_t(int64_t v);
void dense_set_pf_t(int64_t v);
void dense_set_dbg_t(int64_t v);
void dense2_gemm(int64_t nw, int64_t ntb, int64_t nch, int64_t s, int64_t ks, torch::Tensor W, torch::Tensor X, torch::Tensor out);
void dense_gemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                torch::Tensor out, int64_t ksplit, int64_t pair, int64_t nout, torch::Tensor cnt, int64_t fix);
void dense_cgemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                 torch::Tensor out, int64_t ksplit, int64_t nout, torch::Tensor cnt, int64_t fix, int64_t emax, torch::Tensor eoff,
                 torch::Tensor esc);
'''

def _dense_repack(w):
    """w: bf16 [N, K] -> bf16 tiles [N/16, K/64, 4(ks), 32(lane), 8] (one A fragment per lane per k16 step)."""
    N, K = w.shape
    RT, KCn = N // 16, K // 64
    # [rt, rh, g, kc, t, ks, p, h]  with k = kc*64 + t*16 + ks*4 + p*2 + h, row = rt*16 + rh*8 + g
    v = w.view(RT, 2, 8, KCn, 4, 4, 2, 2)
    # fragment regs: (rh, p) -> reg index = rh + 2*p, each reg = (h=0,h=1)
    out = v.permute(0, 3, 5, 2, 4, 6, 1, 7)  # [rt, kc, ks, g, t, p, rh, h]
    return out.contiguous().view(RT, KCn, 4, 32, 8)




def _dense_crepack(w):
    """Lossless exponent compression of bf16 [N, K]: (tiles uint8 [N/16, K/64, 3, 32, 16], emax, esc offsets, escapes int32 [n, 4]) or None."""
    import torch
    N, K = w.shape
    RT, KCn = N // 16, K // 64
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    emax = int(e.max())
    if emax < 20 or emax > 254:
        return None
    code = emax - e
    escm = code > 15
    nesc = int(escm.sum())
    if nesc > N * K // 2000:
        return None
    code = code.clamp(max=15)
    sm = ((sign << 7) | m).to(torch.uint8)
    smt = _dense_repack(sm)            # [RT, KCn, 4(ks), 32, 8]
    cdt = _dense_repack(code.to(torch.int32))
    smb = smt.permute(0, 1, 3, 2, 4).contiguous().view(RT, KCn, 32, 2, 16).permute(0, 1, 3, 2, 4)  # [RT, KCn, 2, 32, 16]
    sh = torch.tensor([4 * (j >> 1) + 16 * (j & 1) for j in range(8)], device=w.device, dtype=torch.int32)
    ev = (cdt << sh).sum(-1).to(torch.int32)          # [RT, KCn, 4, 32]
    eb = ev.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(RT, KCn, 1, 32, 16)
    tiles = torch.cat([smb, eb], 2).contiguous()
    rows, cols = escm.nonzero(as_tuple=True)
    wt = w[rows, cols].float()
    wd_bits = ((sign[rows, cols] << 15) | ((emax - 15) << 7) | m[rows, cols]).to(torch.int16)
    wd = wd_bits.view(torch.bfloat16).float()
    delta = (wt.double() - wd.double()).float()
    esc = torch.stack([rows.to(torch.int32), cols.to(torch.int32), delta.view(torch.int32), torch.zeros_like(rows, dtype=torch.int32)], 1).contiguous()
    if nesc == 0:
        esc = torch.zeros((1, 4), dtype=torch.int32, device=w.device)
    cntp = torch.bincount(rows // 16, minlength=RT)
    eoff = torch.zeros(RT + 1, dtype=torch.int32, device=w.device)
    eoff[1:] = torch.cumsum(cntp, 0).to(torch.int32)
    return tiles, emax, eoff, esc


_DMOD = None
_DENSE_EXTRA = []
_DENSE_PDL = True
_DENSE_PF = False
# (layer, m upper bound) -> (ntb, nch, nw, s, ksplit)
_DENSE_CFG = {'qkv_gate': [(10**9, (1, 4, 4, 4, 4))], 'o': [(10**9, (1, 4, 4, 4, 1))], 'pooler': [(10**9, (1, 4, 4, 4, 2))], 'unpooler': [(8, (1, 1, 4, 4, 4)), (10**9, (1, 4, 4, 4, 1))], 'router': [(10**9, (1, 4, 4, 4, 8))]}


def _dense_module():
    global _DMOD
    if _DMOD is None:
        from torch.utils.cpp_extension import load_inline
        _DMOD = load_inline("muse_dense_bf16", _DENSE_CPP, cuda_sources=_DENSE_CUDA, functions=["dense_gemm", "dense_cgemm", "dense_set_pdl_t", "dense_set_pf_t", "dense_set_dbg_t", "dense2_gemm", "probe_t"],
                            extra_cuda_cflags=["-O3", "--use_fast_math"] + _DENSE_EXTRA, verbose=False)
        _DMOD.dense_set_pdl_t(int(_DENSE_PDL and torch.cuda.get_device_capability()[0] >= 9))
        _DMOD.dense_set_pf_t(int(_DENSE_PF and torch.cuda.get_device_capability()[0] >= 9))
    return _DMOD


def prepare(weights):
    state = _old_prepare(weights)
    state['dense_mod'] = _dense_module()
    hi = weights['router'].to(torch.bfloat16)
    lo = (weights['router'] - hi.float()).to(torch.bfloat16)
    state['d_router'] = _dense_repack(torch.cat([hi, lo], 0).contiguous())
    # self-resetting reduction counters, one set per layer (the router runs on its own stream in the decoder)
    state['d_cnt'] = {name: torch.zeros(4096, dtype=torch.int32, device=hi.device)
                      for name in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')}
    for name in ('qkv_gate', 'o', 'pooler', 'unpooler'):
        state['d_' + name] = _dense_repack(weights[name].contiguous())
        state['c_' + name] = _dense_crepack(weights[name].contiguous())
    torch.cuda.synchronize()
    if _DENSE_LAB:
        _dense_lab(state)
    return state


def _dense_crun(x, state, layer, n, cfg):
    ntb, nch, nw, s, ksplit = cfg[:5]
    fix = cfg[5] if len(cfg) > 5 else 0
    m, k = x.shape
    tiles, emax, eoff, esc = state['c_' + layer]
    nt = n // 16
    part = torch.empty((ksplit, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    state['dense_mod'].dense_cgemm(ntb, nch, nw, s, tiles[:nt], x, part, out, ksplit, n, state['d_cnt'][layer], fix, emax, eoff[:nt + 1], esc)
    return out


def _dense_cfg(layer, m):
    table = _DENSE_CFG.get(layer)
    if table is None:
        return None
    for bound, cfg in table:
        if m <= bound:
            return cfg
    return table[-1][1]


def _dense_run(x, state, layer, n, cfg):
    ntb, nch, nw, s, ksplit = cfg[:5]
    fix = cfg[5] if len(cfg) > 5 else 0
    m, k = x.shape
    router = layer == 'router'
    wt = state['d_' + layer]
    ntot = 512 if router else n
    wt = wt[: ntot // 16]
    part = torch.empty((ksplit, m, ntot), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    state['dense_mod'].dense_gemm(ntb, nch, nw, s, wt, x, part, out, ksplit, 1 if router else 0, n, state['d_cnt'][layer], fix)
    return out


def kernel_fn(data, state):
    x = data['x']
    layer = data['config']['layer']
    n = data['config']['n']
    cfg = _dense_cfg(layer, x.shape[0])
    if cfg is None:
        return _old_kernel_fn(data, state)
    return _dense_run(x.contiguous(), state, layer, n, cfg)

_DENSE_LAB = False
def _dlab_time(fn, flush, reps=12):
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def _dense2_run(x, state, layer, n, cfg):
    nw, ntb, s, ks = cfg
    m = x.shape[0]
    nch = 1 if m <= 8 else 2 if m <= 16 else 4
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    state['dense_mod'].dense2_gemm(nw, ntb, nch, s, ks, state['d_' + layer][: n // 16], x, out)
    return out


def _dense_lab(state):
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    shapes = {'qkv_gate': (2304, 8192), 'o': (8192, 1024), 'pooler': (4096, 8192), 'unpooler': (8192, 4096)}
    for layer, (n_full, k) in shapes.items():
        for m in (8, 16, 32):
            n = 1024 if (layer == 'unpooler' and m <= 8) else n_full
            x = torch.randn((m, k), device='cuda').to(torch.bfloat16)
            base = state[layer][:n]
            ref = x.float() @ base.float().T
            cub = _dlab_time(lambda: x @ base.T, flush)
            res = []
            bad = 0
            for nw, ntb in ((8, 1), (4, 1), (4, 2)):
                for s in (4, 6):
                    for ks in (2, 4, 8):
                        cfg = (nw, ntb, s, ks)
                        try:
                            o = _dense2_run(x, state, layer, n, cfg).float()
                            torch.cuda.synchronize()
                        except Exception as ex:
                            continue
                        err = ((o - ref).norm() / ref.norm()).item()
                        if err > 5e-3:
                            bad += 1
                            continue
                        res.append((_dlab_time(lambda: _dense2_run(x, state, layer, n, cfg), flush, 8), cfg))
            res.sort()
            print('LABE', layer, 'm=%d' % m, 'cublas=%.1f' % cub, 'bad=%d' % bad, ' '.join('%.1f:%s' % (t, '-'.join(map(str, c))) for t, c in res[:5]), flush=True)

_DENSE_LAB = True
