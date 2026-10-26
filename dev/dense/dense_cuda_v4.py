"""Weight-streaming BF16 dense GEMM for small token counts (x [M,K] @ W[N,K].T), fp32 accumulation."""

CUDA_SRC = r"""
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
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// W tiles: uint4 [rt][kc][ks][lane].  grid = (N/16/NTB, KSPLIT).  Each warp covers CPW consecutive k64 chunks.
// part: fp32 [KSPLIT][M][N] (or final bf16 [M][N] when KSPLIT == 1 and FINAL).
template <int NTB, int NCH, int NW, int S, int OM>
__global__ void __launch_bounds__(NW * 32)
dense_kernel(const uint4* __restrict__ Wq, const bf16* __restrict__ X, float* __restrict__ part, int M, int N, int K,
             int ksplit, int* __restrict__ cnt, void* __restrict__ outp, int pair, int nout, int out_fp32, int pf) {
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
    uint4* ring = smem + warp * (S * NTB * 128);
    if (pf && lane == 0 && tok0 == 0) {
#pragma unroll
      for (int n = 0; n < NTB; n++) l2_prefetch_bulk(wsrc[n] - lane, (uint32_t)cpw * 2048u);
    }

    auto issue = [&](int c) {
      if (c < cpw) {
#pragma unroll
        for (int n = 0; n < NTB; n++)
#pragma unroll
          for (int ks = 0; ks < 4; ks++) cp16(&ring[((c % S) * NTB + n) * 128 + ks * 32 + lane], wsrc[n] + c * 128 + ks * 32);
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);

    uint4 xcur[NCH][2], xnxt[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      if (xp[h]) {
        const uint4* xv = reinterpret_cast<const uint4*>(xp[h]);
        xcur[h][0] = __ldg(xv);
        xcur[h][1] = __ldg(xv + 1);
      } else {
        xcur[h][0] = make_uint4(0, 0, 0, 0);
        xcur[h][1] = make_uint4(0, 0, 0, 0);
      }
      xnxt[h][0] = xcur[h][0];
      xnxt[h][1] = xcur[h][1];
    }

#pragma unroll 1
    for (int i = 0; i < cpw; i++) {
      cp_wait<S - 2>();
      issue(i + S - 1);
#pragma unroll
      for (int h = 0; h < NCH; h++) {
        if (xp[h] && i + 1 < cpw) {
          const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (i + 1) * 64);
          xnxt[h][0] = __ldg(xv);
          xnxt[h][1] = __ldg(xv + 1);
        }
      }
#pragma unroll
      for (int n = 0; n < NTB; n++) {
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint4 a = ring[((i % S) * NTB + n) * 128 + ks * 32 + lane];
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = xcur[h][ks >> 1];
            mma16816(acc[n][h], a.x, a.y, a.z, a.w, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
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
  const int dyn = NW * S * NTB * 2048;
  cudaFuncSetAttribute(dense_kernel<NTB, NCH, NW, S, OM>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(dense_kernel<NTB, NCH, NW, S, OM>, grid, dim3(NW * 32), (size_t)dyn, st, g_pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const bf16*>(X), part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, g_pf);
  return 0;
}

extern "C" int dense_run(int ntb, int nch, int nw, int s, int om, const void* W, const void* X, float* part, int M, int N,
                         int K, int ksplit, int* cnt, void* outp, int pair, int nout, int out_fp32, cudaStream_t st) {
#define D(NTB_, NCH_, NW_, S_) if (ntb == NTB_ && nch == NCH_ && nw == NW_ && s == S_) { if (om == 1) return launch<NTB_, NCH_, NW_, S_, 1>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); if (om == 2) return launch<NTB_, NCH_, NW_, S_, 2>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); return launch<NTB_, NCH_, NW_, S_, 0>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); }
  D(1,1,2,4) D(1,1,4,4) D(1,1,8,4) D(1,2,2,4) D(1,2,4,4) D(1,2,8,4) D(1,4,2,4) D(1,4,4,4) D(1,4,8,4) D(2,1,2,4) D(2,1,4,4) D(2,1,8,4) D(2,2,2,4) D(2,2,4,4) D(2,2,8,4) D(2,4,2,4) D(2,4,4,4) D(2,4,8,4) D(1,8,4,4) D(1,8,8,4)
#undef D
  return 1;
}

extern "C" void dense_reduce(const float* part, void* out, int M, int nout, int ntot, int ksplit, int pair, int fp32,
                             cudaStream_t st) {
  const int n = M * nout;
  launch_k(reduce_kernel, dim3((n + 255) / 256), dim3(256), 0, st, g_pdl != 0, part, out, M, nout, ntot, ksplit, pair, fp32);
}
"""

TORCH_GLUE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int dense_run(int, int, int, int, int, const void*, const void*, float*, int, int, int, int, int*, void*, int, int, int, cudaStream_t);
extern "C" void dense_reduce(const float*, void*, int, int, int, int, int, int, cudaStream_t);
extern "C" void dense_set_pdl(int);
extern "C" void dense_set_pf(int);
void dense_set_pdl_t(int64_t v) { dense_set_pdl((int)v); }
void dense_set_pf_t(int64_t v) { dense_set_pf((int)v); }

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
"""

CPP_SRC = r"""
void dense_set_pdl_t(int64_t v);
void dense_set_pf_t(int64_t v);
void dense_gemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                torch::Tensor out, int64_t ksplit, int64_t pair, int64_t nout, torch::Tensor cnt, int64_t fix);
"""


def dense_repack(w):
    """w: bf16 [N, K] -> bf16 tiles [N/16, K/64, 4(ks), 32(lane), 8] (one A fragment per lane per k16 step)."""
    N, K = w.shape
    RT, KCn = N // 16, K // 64
    # [rt, rh, g, kc, t, ks, p, h]  with k = kc*64 + t*16 + ks*4 + p*2 + h, row = rt*16 + rh*8 + g
    v = w.view(RT, 2, 8, KCn, 4, 4, 2, 2)
    # fragment regs: (rh, p) -> reg index = rh + 2*p, each reg = (h=0,h=1)
    out = v.permute(0, 3, 5, 2, 4, 6, 1, 7)  # [rt, kc, ks, g, t, p, rh, h]
    return out.contiguous().view(RT, KCn, 4, 32, 8)
