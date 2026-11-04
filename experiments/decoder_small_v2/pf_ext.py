"""decoder_small_v2: L2 prefetch of a later kernel's weight stream, issued from a side-stream kernel that runs
concurrently with the MoE (whose input-dependent expert stream leaves HBM bandwidth spare at T <= 8).

The kernel only moves weights (data independent) into L2 with cp.async.bulk.prefetch.L2; it reads no activations,
computes nothing and caches nothing across calls. Optional: start delay (globaltimer spin) and an L2 evict_last hint.
"""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

PF_CUDA = r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ uint64_t pf_now() { uint64_t t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

// each thread prefetches CHUNK-byte pieces strided over the grid
__global__ void pf_l2_kernel(const char* __restrict__ base, long long bytes, int chunk, long long delay_ns, int hint) {
  if (delay_ns > 0) {
    const uint64_t t0 = pf_now();
    while (pf_now() - t0 < (uint64_t)delay_ns) __nanosleep(256);
  }
  uint64_t pol = 0;
  if (hint) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  const long long n = (bytes + chunk - 1) / chunk;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const long long off = i * chunk;
    const uint32_t sz = (uint32_t)min((long long)chunk, bytes - off);
    const char* p = base + off;
    if (hint)
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(p), "r"(sz), "l"(pol) : "memory");
    else
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(sz) : "memory");
  }
}

void pf_l2(at::Tensor t, int64_t bytes, int64_t chunk, int64_t ctas, int64_t threads, int64_t delay_ns, int64_t hint) {
  auto st = at::cuda::getCurrentCUDAStream();
  pf_l2_kernel<<<(int)ctas, (int)threads, 0, st>>>(reinterpret_cast<const char*>(t.data_ptr()), (long long)bytes,
                                                   (int)chunk, (long long)delay_ns, (int)hint);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
PF_CPP = "void pf_l2(at::Tensor t, int64_t bytes, int64_t chunk, int64_t ctas, int64_t threads, int64_t delay_ns, int64_t hint);"
_EXT = None


def pf_ext():
    global _EXT
    if _EXT is None:
        tag = hashlib.md5(PF_CUDA.encode()).hexdigest()[:12]
        _EXT = load_inline("dsv2_pf_" + tag, PF_CPP, cuda_sources=PF_CUDA, functions=["pf_l2"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _EXT


def prefetch_l2(t, chunk=16384, ctas=8, threads=128, delay_ns=0, hint=1):
    """Prefetch the whole of tensor t into L2 (asynchronous, data independent)."""
    b = t.numel() * t.element_size()
    b = b // 16 * 16
    pf_ext().pf_l2(t, b, chunk, ctas, threads, delay_ns, hint)
