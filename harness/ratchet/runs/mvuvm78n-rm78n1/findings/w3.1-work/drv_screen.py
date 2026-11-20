"""Split-K TMA + wgmma BF16 GEMM for skinny-M projections (out[M,N] = x[M,K] @ W[N,K]^T).
CTA (rb, sp): W rows rb*BM..+BM (BM = 64*NWG, one m64 slab per consumer warpgroup), all M tokens (wgmma N = BN),
k blocks [sp*nkb/S, (sp+1)*nkb/S).  FP32 partials go to a workspace; once all S CTAs of a row block have arrived
(monotonic counter, grid <= #SMs so all co-resident) each CTA reduces 1/S of the tile and writes BF16."""
import torch
from torch.utils.cpp_extension import load_inline

SK_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>

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
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint64_t l2_pol_ef() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ uint64_t l2_pol_el() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ uint64_t desc128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
WGMMA_DEFS

template <int NWG, int BN, int STAGES, int REGS, int PACK, int CLU>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
sk_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          const __nv_bfloat16* __restrict__ Wp, __nv_bfloat16* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
          int M, int N, int K, int splits, unsigned long long* __restrict__ dbg) {
  constexpr int BM = NWG * 64;
  unsigned long long t_start = 0, t_first = 0, t_main = 0, t_sync = 0;
  if (dbg && threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
  constexpr int A_BYTES = BM * 128, B_BYTES = BN * 128, ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2, NT = NWG * 128;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(smem_raw) + 1023u) & ~1023u;
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = K >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int nkb = (int)(((long long)(sp + 1) * nkb_all) / splits) - kb0;
  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp >= NWG * 4) {
    if (REGS) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (warp == NWG * 4 && lane == 0) {
      const uint64_t pw = l2_pol_ef(), px = l2_pol_el();
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
        if (PACK) bulk_g2s(sbase + slot * ST_BYTES, Wp + ((size_t)rb * nkb_all + kb0 + i) * (BM * 64), A_BYTES, full0 + 8 * slot, pw);
        else tma2d(sbase + slot * ST_BYTES, &mapW, full0 + 8 * slot, kg, rb * BM, pw);
        tma2d(sbase + slot * ST_BYTES + A_BYTES, &mapX, full0 + 8 * slot, kg, 0, px);
      }
    }
    if (CLU) {
      asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
      asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
    }
    return;
  }
  if (REGS) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(REGS));
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    if (dbg && tid == 0 && i == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_first));
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, desc128(a_st + j * 32), desc128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
  if (dbg && tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_main));
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + wg * 64 + (warp & 3) * 16;
  if (CLU) {
    static_assert(!CLU || (NACC * NT * 4 <= STAGES * ST_BYTES), "reduction buffer does not fit");
    named_bar(1, NT);   // every consumer is done reading the pipeline buffers
    float4* red = reinterpret_cast<float4*>(smem_raw + (sbase - (uint32_t)__cvta_generic_to_shared(smem_raw)));
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j) red[j * NT + tid] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
    uint32_t rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int j0 = (int)(((long long)rank * (NACC / 4)) / splits);
    const int j1 = (int)(((long long)(rank + 1) * (NACC / 4)) / splits);
    for (int j = j0; j < j1; ++j) {
      const uint32_t la = sbase + (uint32_t)((j * NT + tid) * 16);
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int q = 0; q < splits; ++q) {
        uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"(q));
        float4 v;
        asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(ra) : "memory");
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      }
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(s.x);
        out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(s.z);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(s.y);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(s.w);
      }
    }
    if (dbg && tid == 0) {
      unsigned long long t_end; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
      unsigned long long* d = dbg + 6 * (blockIdx.y * gridDim.x + blockIdx.x);
      d[0] = t_start; d[1] = t_first; d[2] = t_main; d[3] = t_main; d[4] = t_end; d[5] = rank;
    }
    asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
    return;
  }
  float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
  for (int j = 0; j < NACC / 4; ++j)
    __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    while (*((volatile int*)(counters + rb)) - target < 0) __nanosleep(20);
  }
  named_bar(1, NT);
  __threadfence();
  if (dbg && tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_sync));
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(s.w);
    }
  }
  if (dbg) {
    named_bar(1, NT);
    if (tid == 0) {
      unsigned long long t_end; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
      unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      unsigned long long* d = dbg + 6 * (blockIdx.y * gridDim.x + blockIdx.x);
      d[0] = t_start; d[1] = t_first; d[2] = t_main; d[3] = t_sync; d[4] = t_end; d[5] = smid;
    }
  }
}

static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q; void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap map2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows}; cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

template <int NWG, int BN, int STAGES, int REGS, int PACK, int CLU>
static void launch_t(const at::Tensor& W, const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out, at::Tensor& ws, at::Tensor& cnt, int splits, const at::Tensor& dbg) {
  constexpr int BM = NWG * 64;
  const int M = x.size(0), N = W.size(0), K = W.size(1);
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES;
  auto kern = sk_kernel<NWG, BN, STAGES, REGS, PACK, CLU>;
  static bool done = false;
  if (!done) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); done = true; }
  CUtensorMap mW = map2d(W.data_ptr(), N, K, BM);
  CUtensorMap mX = map2d(x.data_ptr(), M, K, BN);
  dim3 grid(N / BM, splits);
  unsigned long long* dp = dbg.numel() ? reinterpret_cast<unsigned long long*>(dbg.data_ptr()) : nullptr;
  if (CLU) {
    static bool nps = false;
    if (!nps) { cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); nps = true; }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid; lc.blockDim = dim3(NWG * 128 + 128); lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = splits; at[0].val.clusterDim.z = 1;
    lc.attrs = at; lc.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&lc, kern, mW, mX, reinterpret_cast<const __nv_bfloat16*>(Wp.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, splits, dp);
    TORCH_CHECK(e == cudaSuccess, "cluster launch failed: ", cudaGetErrorString(e));
  } else {
    kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mW, mX, reinterpret_cast<const __nv_bfloat16*>(Wp.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, splits, dp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int sk_launch(at::Tensor W, at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor cnt, at::Tensor dbg,
              int64_t nwg, int64_t bn, int64_t stages, int64_t regs, int64_t splits, int64_t pack, int64_t clu) {
DISPATCH
  return 1;
}
'''

SK_CPP = "int sk_launch(at::Tensor W, at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor cnt, at::Tensor dbg, int64_t nwg, int64_t bn, int64_t stages, int64_t regs, int64_t splits, int64_t pack, int64_t clu);"


def _wgmma_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        out.append(f"""template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 {{{regs}}}, %{nacc}, %{nacc+1}, 1, 1, 1, 0, 0;"
               : {cons} : "l"(da), "l"(db));
}}""")
    return "\n".join(out)


def sk_build(configs, name="sk_splitk"):
    """configs: iterable of (nwg, bn, stages, regs, pack, clu)."""
    configs = sorted(set(configs))
    disp = "\n".join(f"  if (nwg=={a} && bn=={b} && stages=={c} && regs=={d} && pack=={e} && clu=={f}) {{ launch_t<{a},{b},{c},{d},{e},{f}>(W,Wp,x,out,ws,cnt,(int)splits,dbg); return 0; }}"
                     for a, b, c, d, e, f in configs)
    src = SK_SRC.replace("WGMMA_DEFS", _wgmma_defs(sorted({c[1] for c in configs}))).replace("DISPATCH", disp)
    import hashlib
    h = hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(f"{name}_{h}", SK_CPP, cuda_sources=src, functions=["sk_launch"],
                       extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "--ptxas-options=-v"],
                       verbose=bool(int(__import__("os").environ.get("SKV","0"))))


def sk_pack(w, bm):
    """[N, K] -> [N/bm, K/64, bm, 64] tiles, each the exact 128B-swizzled smem image TMA would produce."""
    n, k = w.shape
    t = w.view(n // bm, bm, k // 64, 8, 8).permute(0, 2, 1, 3, 4)          # [nb, kb, r, chunk, 8]
    r = torch.arange(bm, device=w.device).view(bm, 1)
    c = torch.arange(8, device=w.device).view(1, 8)
    src = (c ^ (r % 8))                                                      # dst chunk c holds src chunk c^(r%8)
    t = t[:, :, torch.arange(bm, device=w.device).view(bm, 1), src, :]       # [nb, kb, bm, 8, 8]
    return t.contiguous().view(n, k)


class SkRunner:
    def __init__(self, ext, device, max_ws_floats=16 << 20):
        self.ext = ext
        self.ws = torch.empty(max_ws_floats // 4 * 4, dtype=torch.float32, device=device)
        self.cnt = {}
        self.packs = {}
        self.dbg = torch.empty(0, dtype=torch.int64, device=device)

    def run(self, x, W, cfg):
        nwg, bn, stages, regs, splits, pack = cfg[:6]
        clu = cfg[6] if len(cfg) > 6 else 0
        m, n = x.shape[0], W.shape[0]
        key = (n, nwg, splits)
        c = self.cnt.get(key)
        if c is None:
            c = self.cnt[key] = torch.zeros(n // (64 * nwg), dtype=torch.int32, device=x.device)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        wp = W
        if pack:
            pk = (W.data_ptr(), nwg)
            wp = self.packs.get(pk)
            if wp is None:
                wp = self.packs[pk] = sk_pack(W, 64 * nwg)
        rc = self.ext.sk_launch(W, wp, x, out, self.ws, c, self.dbg, nwg, bn, stages, regs, splits, pack, clu)
        if rc:
            raise RuntimeError(f"sk config {cfg} not compiled")
        return out

# ---------------- driver ----------------
import importlib.util, json, os, statistics, sys, time
import torch.nn.functional as F
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R

CONFIGS = {
    ("qkv_gate", 128): [(2,128,6,0,4,1,1), (2,128,6,0,6,1,1), (2,128,6,0,7,1,1), (3,128,5,0,8,1,1), (1,128,8,0,3,1,1), (1,128,8,0,4,1,1)],
    ("qkv_gate", 256): [(2,256,4,0,4,1,1), (2,256,4,0,6,1,1), (2,256,4,0,7,1,1), (1,256,5,0,3,1,1), (1,256,5,0,4,1,1)],
    ("pooler", 128):   [(2,128,6,0,4,1,1), (2,128,6,0,2,1,1), (1,128,8,0,2,1,1), (2,128,6,0,8,1,1)],
    ("pooler", 256):   [(2,256,4,0,4,1,1), (2,256,4,0,2,1,1), (1,256,5,0,2,1,1), (2,256,4,0,8,1,1)],
}
ONLY = os.environ.get("ONLY")
if ONLY:
    CONFIGS = {k: v for k, v in CONFIGS.items() if k[0] in ONLY.split(",")}
t0 = time.time()
ext = sk_build({c[:4] + c[5:7] for v in CONFIGS.values() for c in v})
print(f"built in {time.time()-t0:.1f}s", flush=True)
weights = R.generate_weights(R.WEIGHTS_SEED)
dev = weights["qkv_gate"].device
runner = SkRunner(ext, dev)

flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)
ea = torch.cuda.Event(enable_timing=True); eb = torch.cuda.Event(enable_timing=True)
def capture(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay(); torch.cuda.synchronize()
    return g, out
def elapsed(g):
    flush.add_(1); ea.record(); g.replay(); eb.record(); eb.synchronize()
    return ea.elapsed_time(eb) * 1000.0
def paired(graphs, reps=40):
    t = [[] for _ in graphs]
    for r in range(reps):
        order = range(len(graphs)) if r % 2 == 0 else reversed(range(len(graphs)))
        for i in order:
            t[i].append(elapsed(graphs[i]))
    return [statistics.median(v) for v in t]

def screen(base_fn, label):
    rows = []
    for (layer, m), cfgs in CONFIGS.items():
        n = weights[layer].shape[0]
        sol = R.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989.5e12}) / 1000.0
        good = []
        for cfg in cfgs:
            try:
                for seed, dist in ((123, "normal"), (7, "outliers")):
                    d = R.generate_input(m, layer, seed=seed, dist=dist)
                    o = runner.run(d["x"], weights[layer], cfg)
                    torch.cuda.synchronize()
                    ok, msg = R.check_implementation(d, o, weights)
                    if not ok:
                        raise RuntimeError(msg)
                good.append(cfg)
            except Exception as e:
                print(f"CFG_FAIL {layer} m={m} {cfg}: {e}", flush=True)
        per = {}
        for seed in (42, 108, 999):
            d = R.generate_input(m, layer, seed=seed)
            gs = [capture(lambda: base_fn(d))[0]]
            for cfg in good:
                gs.append(capture(lambda cfg=cfg: runner.run(d["x"], weights[layer], cfg))[0])
            meds = paired(gs)
            for i, t in enumerate(meds):
                per.setdefault(i, []).append(t)
            del gs
        bt = statistics.geometric_mean(per[0])
        row = {"layer": layer, "m": m, "sol": round(sol, 2), label: round(bt, 2),
               "cands": {str(c): round(statistics.geometric_mean(per[i + 1]), 2) for i, c in enumerate(good)}}
        print("SCREEN_ROW", json.dumps(row), flush=True)
        rows.append(row)
    return rows

def cublas(d):
    return F.linear(d["x"], weights[d["config"]["layer"]])
screen(cublas, "cublas")

if os.environ.get("BASE", "1") == "1":
    spec = importlib.util.spec_from_file_location("cand", "/w/run/sub.py")
    cand = importlib.util.module_from_spec(spec); spec.loader.exec_module(cand)
    t0 = time.time(); state = cand.prepare(weights); print(f"base prepare {time.time()-t0:.1f}s", flush=True)
    screen(lambda d: cand.kernel_fn(d, state), "base")
