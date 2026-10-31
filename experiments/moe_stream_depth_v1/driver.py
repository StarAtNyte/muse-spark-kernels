"""MoE achievable-floor diagnostic: per scored bs, the submission's kernel_fn time vs a pure streaming read of
exactly the active experts' weight + scale bytes (one kernel, cold L2) and an empty-kernel graph."""
import importlib.util, json, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

CU = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
struct Seg { const uint4* p; long long n; };
struct Segs { Seg s[64]; int k; };
__global__ void __launch_bounds__(512) stream_k(Segs S, unsigned* out) {
  unsigned acc = 0;
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long long)gridDim.x * blockDim.x;
  for (int j = 0; j < S.k; ++j) {
    const uint4* p = S.s[j].p; long long n = S.s[j].n;
    long long i = tid;
    for (; i + 3 * nth < n; i += 4 * nth) {
      uint4 a, b, c, d;
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(p + i));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(p + i + nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w) : "l"(p + i + 2 * nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w) : "l"(p + i + 3 * nth));
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += nth) { uint4 a = p[i]; acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x12345678u) out[0] = acc;
}
__global__ void __launch_bounds__(512) stream_k8(Segs S, unsigned* out) {
  unsigned acc = 0;
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long long)gridDim.x * blockDim.x;
  for (int j = 0; j < S.k; ++j) {
    const uint4* p = S.s[j].p; long long n = S.s[j].n;
    long long i = tid;
    for (; i + 7 * nth < n; i += 8 * nth) {
      uint4 a, b, c, d, e, f, g, h;
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(p + i));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(p + i + nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w) : "l"(p + i + 2 * nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w) : "l"(p + i + 3 * nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e.x), "=r"(e.y), "=r"(e.z), "=r"(e.w) : "l"(p + i + 4 * nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(f.x), "=r"(f.y), "=r"(f.z), "=r"(f.w) : "l"(p + i + 5 * nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(g.x), "=r"(g.y), "=r"(g.z), "=r"(g.w) : "l"(p + i + 6 * nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(h.x), "=r"(h.y), "=r"(h.z), "=r"(h.w) : "l"(p + i + 7 * nth));
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w ^ e.x ^ e.y ^ e.z ^ e.w ^ f.x ^ f.y ^ f.z ^ f.w ^ g.x ^ g.y ^ g.z ^ g.w ^ h.x ^ h.y ^ h.z ^ h.w;
    }
    for (; i < n; i += nth) { uint4 a = p[i]; acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x12345678u) out[0] = acc;
}
__global__ void empty_k(unsigned* out) { if (threadIdx.x == 1023) out[0] = 1; }
void run_stream(std::vector<torch::Tensor> ts, torch::Tensor out, int grid, int threads, int depth) {
  Segs S; S.k = (int)ts.size();
  for (int j = 0; j < S.k; ++j) { S.s[j].p = (const uint4*)ts[j].data_ptr(); S.s[j].n = ts[j].numel() * ts[j].element_size() / 16; }
  if (depth == 4) stream_k<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(S, (unsigned*)out.data_ptr());
  else stream_k8<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(S, (unsigned*)out.data_ptr());
  cudaError_t e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
void run_empty(torch::Tensor out) { empty_k<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>((unsigned*)out.data_ptr()); }
'''
ext = load_inline('stream_floor', cpp_sources='void run_stream(std::vector<torch::Tensor> ts, torch::Tensor out, int grid, int threads, int depth); void run_empty(torch::Tensor out);',
                  cuda_sources=CU, functions=['run_stream', 'run_empty'], extra_cuda_cflags=['-O3', '-arch=sm_90a'])
w = R.generate_weights(1234)
st = S.prepare(w)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
outb = torch.zeros(4, dtype=torch.int32, device='cuda')
def cap(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def med(g, n=20):
    ts = []
    for _ in range(n):
        flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
ge = cap(lambda: ext.run_empty(outb)); empty_us = med(ge)
probe = [w['w13'][0], w['w2'][0]]
ext.run_stream(probe, outb, 264, 512, 4); torch.cuda.synchronize()
print('PROBE_OK', flush=True)
print('EMPTY_US', round(empty_us, 2), flush=True)
for bs in (16, 32):
    for seed in (419,):
        d = R.generate_input(bs, 'ep8', seed)
        ids = d['topk_ids'].long(); loc = ids[(ids >= 0) & (ids < 32)]
        act = loc.unique().tolist()
        segs = []
        for e in act:
            segs += [w['w13'][e], w['w13_sf'][e], w['w2'][e], w['w2_sf'][e]]
        nbytes = sum(t.numel() * t.element_size() for t in segs)
        gk = cap(lambda: S.kernel_fn(d, st)); k_us = med(gk)
        for grid, thr in ((264, 512), (528, 256)):
            graphs = {depth: cap(lambda depth=depth: ext.run_stream(segs, outb, grid, thr, depth)) for depth in (4, 8)}
            for depth in (4, 8):
                us = med(graphs[depth])
                print('ROW', json.dumps(dict(bs=bs, seed=seed, distinct=len(act), MB=round(nbytes/1e6, 1), kernel_us=round(k_us, 2), stream_us=round(us, 2), grid=grid, threads=thr, depth=depth)), flush=True)
print('DONE', flush=True)
