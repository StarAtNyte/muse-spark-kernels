"""Random 128 B row-gather bandwidth on H200 (the judge's page=1 KV layout: K and V rows of 2 heads x 64 FP8 bytes at a
random permutation). Methods: LDG.128 with 8 threads per row and R rows in flight per thread group; cp.async 16 B into
a per-warp smem ring; TMA bulk 128 B per row. Cold L2, CUDA graph, median of 20."""
import json, statistics
import torch
from torch.utils.cpp_extension import load_inline

CU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
// LDG: a warp gathers 4 rows per step (8 lanes x 16 B per row); R independent steps in flight per lane
template <int R>
__global__ void __launch_bounds__(256) ldg_k(const uint4* __restrict__ K, const uint4* __restrict__ V, const int* __restrict__ idx,
                                             long long n, unsigned* out) {
  const int lane = threadIdx.x & 31, sub = lane >> 3, part = lane & 7;
  const long long wid = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const long long nw = ((long long)gridDim.x * blockDim.x) >> 5;
  unsigned acc = 0;
  for (long long r0 = wid * 4 * R; r0 < n; r0 += nw * 4 * R) {
    int rr[R];
#pragma unroll
    for (int j = 0; j < R; ++j) { long long r = r0 + j * 4 + sub; rr[j] = r < n ? __ldg(idx + r) : -1; }
    uint4 a[R], b[R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
      if (rr[j] >= 0) {
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a[j].x), "=r"(a[j].y), "=r"(a[j].z), "=r"(a[j].w) : "l"(K + (long long)rr[j] * 8 + part));
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[j].x), "=r"(b[j].y), "=r"(b[j].z), "=r"(b[j].w) : "l"(V + (long long)rr[j] * 8 + part));
      } else { a[j] = make_uint4(0, 0, 0, 0); b[j] = a[j]; }
    }
#pragma unroll
    for (int j = 0; j < R; ++j) acc ^= a[j].x ^ a[j].w ^ b[j].y ^ b[j].z;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
// TMA bulk: one elected lane per warp issues 128 B copies of K and V rows into a per-warp ring of D row pairs
template <int D>
__global__ void __launch_bounds__(128) tma_k(const char* __restrict__ K, const char* __restrict__ V, const int* __restrict__ idx,
                                             long long n, unsigned* out) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) unsigned long long bar[4][2];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  char* ring = sm + warp * 2 * D * 256;  // two halves of D row pairs
  const long long wid = (long long)blockIdx.x * 4 + warp, nw = (long long)gridDim.x * 4;
  if (lane == 0) for (int h = 0; h < 2; ++h) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&bar[warp][h])));
  __syncwarp();
  unsigned acc = 0;
  long long r = wid * D;
  int h = 0; unsigned ph[2] = {0, 0};
  auto issue = [&](long long rb, int hh) {
    if (lane == 0) {
      int cnt = (int)min((long long)D, n - rb);
      unsigned a = (unsigned)__cvta_generic_to_shared(&bar[warp][hh]);
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(cnt * 256));
      for (int j = 0; j < cnt; ++j) {
        const int row = __ldg(idx + rb + j);
        unsigned d = (unsigned)__cvta_generic_to_shared(ring + (hh * D + j) * 256);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];" :: "r"(d), "l"(K + (long long)row * 128), "r"(a) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];" :: "r"(d + 128), "l"(V + (long long)row * 128), "r"(a) : "memory");
      }
    }
  };
  if (r < n) issue(r, 0);
  for (; r < n; r += nw * D) {
    const long long rn = r + nw * D;
    if (rn < n) issue(rn, h ^ 1);
    unsigned a = (unsigned)__cvta_generic_to_shared(&bar[warp][h]);
    asm volatile("{ .reg .pred P; W: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @!P bra W; }" :: "r"(a), "r"(ph[h]));
    ph[h] ^= 1;
    acc ^= ((const unsigned*)(ring + h * D * 256))[lane];
    __syncwarp();
    h ^= 1;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
void run(int method, torch::Tensor K, torch::Tensor V, torch::Tensor idx, torch::Tensor out, int grid) {
  auto st = at::cuda::getCurrentCUDAStream();
  long long n = idx.numel();
  const uint4* k = (const uint4*)K.data_ptr(); const uint4* v = (const uint4*)V.data_ptr();
  const int* ix = (const int*)idx.data_ptr(); unsigned* o = (unsigned*)out.data_ptr();
  switch (method) {
    case 0: ldg_k<2><<<grid, 256, 0, st>>>(k, v, ix, n, o); break;
    case 1: ldg_k<4><<<grid, 256, 0, st>>>(k, v, ix, n, o); break;
    case 2: ldg_k<8><<<grid, 256, 0, st>>>(k, v, ix, n, o); break;
    case 3: ldg_k<16><<<grid, 256, 0, st>>>(k, v, ix, n, o); break;
    case 4: { auto f = tma_k<16>; int sm = 4 * 2 * 16 * 256; cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
              f<<<grid, 128, sm, st>>>((const char*)k, (const char*)v, ix, n, o); break; }
    case 5: { auto f = tma_k<32>; int sm = 4 * 2 * 32 * 256; cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
              f<<<grid, 128, sm, st>>>((const char*)k, (const char*)v, ix, n, o); break; }
    case 6: { auto f = tma_k<64>; int sm = 4 * 2 * 64 * 256; cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
              f<<<grid, 128, sm, st>>>((const char*)k, (const char*)v, ix, n, o); break; }
  }
  cudaError_t e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
'''
ext = load_inline('gather_bw', cpp_sources='void run(int method, torch::Tensor K, torch::Tensor V, torch::Tensor idx, torch::Tensor out, int grid);',
                  cuda_sources=CU, functions=['run'], extra_cuda_cflags=['-O3', '-arch=sm_90a'])
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
out = torch.zeros(4, dtype=torch.int32, device='cuda')
def med(fn, n=20):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(n):
        flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
empty = med(lambda: out.add_(1))
print('EMPTY_US', round(empty, 2), flush=True)
NAMES = ['ldg_r2', 'ldg_r4', 'ldg_r8', 'ldg_r16', 'tma_d16', 'tma_d32', 'tma_d64']
for rows in tuple(int(r) for r in __import__('os').environ.get('ROWS', '80000,2048,32768').split(',')):  # decode 32 / 16 at ~80K tokens per request
    K = torch.randint(0, 255, (rows, 128), dtype=torch.uint8, device='cuda')
    V = torch.randint(0, 255, (rows, 128), dtype=torch.uint8, device='cuda')
    for order in ('random',):
        idx = (torch.randperm(rows, device='cuda') if order == 'random' else torch.arange(rows, device='cuda')).to(torch.int32)
        nbytes = rows * 256
        best = []
        for m, name in enumerate(NAMES):
            grids = (66, 132, 252, 264, 528, 1056) if m < 4 else (132, 252, 264)
            for grid in grids:
                try: us = med(lambda: ext.run(m, K, V, idx, out, grid))
                except Exception as e: print('FAIL', name, grid, str(e)[:80], flush=True); continue
                best.append((us, name, grid))
        best.sort()
        for us, name, grid in best[:5]:
            print('ROW', json.dumps(dict(rows=rows, order=order, MB=round(nbytes / 1e6), method=name, grid=grid, us=round(us, 1),
                  TBs=round(nbytes / us / 1e6, 2), TBs_minus_empty=round(nbytes / (us - empty) / 1e6, 2))), flush=True)
print('DONE', flush=True)
