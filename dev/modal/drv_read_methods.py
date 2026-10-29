"""H200 achievable read bandwidth by method: LDG unroll depth, L2 256B prefetch hint, TMA bulk copies to smem.
Cold L2, CUDA graph, median of 30. Sizes match MoE decode loads (56.6 MB = 2 experts .. 425 MB = 15 experts)."""
import json, statistics
import torch
from torch.utils.cpp_extension import load_inline

CU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
template <int U, int HINT>
__global__ void ldg_k(const uint4* __restrict__ p, long long n, unsigned* out) {
  unsigned acc = 0;
  long long nth = (long long)gridDim.x * blockDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * nth < n; i += U * nth) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (HINT) asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + i + u * nth));
      else asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + i + u * nth));
    }
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n; i += nth) { uint4 a = p[i]; acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  if (acc == 0x12345678u) out[0] = acc;
}
// contiguous-chunk LDG: each CTA owns one contiguous slab (DRAM page locality)
template <int U>
__global__ void ldgc_k(const uint4* __restrict__ p, long long n, unsigned* out) {
  unsigned acc = 0;
  long long per = (n + gridDim.x - 1) / gridDim.x;
  long long b = blockIdx.x * per, e = min(n, b + per);
  for (long long i = b + threadIdx.x; i < e; i += U * blockDim.x) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      long long j = i + u * blockDim.x;
      if (j < e) asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + j));
      else v[u] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
// TMA bulk: one elected thread issues cp.async.bulk of CH bytes per stage into smem, NS stages; contiguous slab per CTA
template <int NS, int CH>
__global__ void bulk_k(const char* __restrict__ p, long long nbytes, unsigned* out) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) unsigned long long bar[NS];
  long long nch = nbytes / CH;
  long long per = (nch + gridDim.x - 1) / gridDim.x;
  long long c0 = blockIdx.x * per, c1 = min(nch, c0 + per);
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) {
      unsigned a = (unsigned)__cvta_generic_to_shared(&bar[s]);
      asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(a));
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  auto issue = [&](long long c, int s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(&bar[s]);
    unsigned d = (unsigned)__cvta_generic_to_shared(sm + s * CH);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(CH));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(d), "l"(p + c * CH), "r"(CH), "r"(a) : "memory");
  };
  if (threadIdx.x == 0) for (int s = 0; s < NS && c0 + s < c1; ++s) issue(c0 + s, s);
  unsigned acc = 0;
  for (long long c = c0, k = 0; c < c1; ++c, ++k) {
    int s = (int)(k % NS); unsigned ph = (unsigned)((k / NS) & 1);
    unsigned a = (unsigned)__cvta_generic_to_shared(&bar[s]);
    asm volatile("{ .reg .pred P; W: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @!P bra W; }" :: "r"(a), "r"(ph));
    acc ^= ((const unsigned*)(sm + s * CH))[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0 && c + NS < c1) issue(c + NS, s);
  }
  if (acc == 0x12345678u) out[0] = acc;
}
void run(int method, torch::Tensor buf, torch::Tensor out, int grid, int threads) {
  auto st = at::cuda::getCurrentCUDAStream();
  long long n = buf.numel() * buf.element_size();
  const uint4* p = (const uint4*)buf.data_ptr(); unsigned* o = (unsigned*)out.data_ptr();
  switch (method) {
    case 0: ldg_k<4, 0><<<grid, threads, 0, st>>>(p, n / 16, o); break;
    case 1: ldg_k<8, 0><<<grid, threads, 0, st>>>(p, n / 16, o); break;
    case 2: ldg_k<8, 1><<<grid, threads, 0, st>>>(p, n / 16, o); break;
    case 3: ldg_k<16, 1><<<grid, threads, 0, st>>>(p, n / 16, o); break;
    case 4: ldgc_k<8><<<grid, threads, 0, st>>>(p, n / 16, o); break;
    case 5: { auto f = bulk_k<4, 32768>; cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 32768);
              f<<<grid, threads, 4 * 32768, st>>>((const char*)p, n, o); break; }
    case 6: { auto f = bulk_k<6, 32768>; cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 6 * 32768);
              f<<<grid, threads, 6 * 32768, st>>>((const char*)p, n, o); break; }
    case 7: { auto f = bulk_k<8, 16384>; cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 16384);
              f<<<grid, threads, 8 * 16384, st>>>((const char*)p, n, o); break; }
  }
  cudaError_t e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
'''
ext = load_inline('read_methods', cpp_sources='void run(int method, torch::Tensor buf, torch::Tensor out, int grid, int threads);',
                  cuda_sources=CU, functions=['run'], extra_cuda_cflags=['-O3', '-arch=sm_90a'])
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
out = torch.zeros(4, dtype=torch.int32, device='cuda')
def med(fn, n=30):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(n):
        flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
empty = med(lambda: out.add_(1))
print('EMPTY_US', round(empty, 2), flush=True)
NAMES = ['ldg_u4', 'ldg_u8', 'ldg_u8_l2hint', 'ldg_u16_l2hint', 'ldg_contig_u8', 'tma_4x32K', 'tma_6x32K', 'tma_8x16K']
for mb in (56.6, 113.2, 340.0, 425.0):
    buf = torch.randint(0, 255, (int(mb * 1e6) // 32768 * 32768,), dtype=torch.uint8, device='cuda')
    rows = []
    for m, name in enumerate(NAMES):
        cfgs = ((132, 1024), (264, 512), (528, 256), (1056, 256)) if m < 5 else ((132, 256), (264, 256))
        for grid, thr in cfgs:
            if m >= 5 and grid == 264 and m != 7: continue  # 2 CTAs/SM only fit the 128 KB variant
            try:
                us = med(lambda: ext.run(m, buf, out, grid, thr))
            except Exception as e:
                print('FAIL', name, grid, thr, str(e)[:100], flush=True); continue
            rows.append((us, name, grid, thr))
    rows.sort()
    for us, name, grid, thr in rows[:8]:
        print('ROW', json.dumps(dict(MB=mb, method=name, grid=grid, threads=thr, us=round(us, 1),
              TBs=round(buf.numel() / us / 1e6, 2), TBs_minus_empty=round(buf.numel() / (us - empty) / 1e6, 2))), flush=True)
print('DONE', flush=True)
