_BW_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
// pattern 0: Marlin-like. grid (nct, items); CTA owns `ntl` consecutive 64-row n-tiles (512 B pieces per k16 step), 8 warps split K (256 k-steps)
// pattern 1: contiguous: CTA owns a contiguous slab of the same size, each warp streams a contiguous 1/8 of it
__global__ void bw_k(const uint4* __restrict__ w, unsigned* out, int pattern, int ntiles, int ntl, int items, int kt_per_warp, size_t estride) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int item = blockIdx.y;
  const uint4* base = w + (size_t)item * estride;      // estride in uint4
  uint32_t acc = 0;
  if (pattern == 0) {
    const int nt0 = blockIdx.x * ntl;
#pragma unroll 8
    for (int i = 0; i < kt_per_warp; i++) {
      const int kt = warp * kt_per_warp + i;
      for (int j = 0; j < ntl; j++) {
        const uint4 v = __ldcs(base + ((size_t)kt * ntiles + nt0 + j) * 32 + lane);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    }
  } else {
    const size_t slab = (size_t)256 * ntl * 32;         // uint4 per CTA
    const uint4* b = base + (size_t)blockIdx.x * slab + (size_t)warp * (slab / 8);
#pragma unroll 8
    for (size_t i = lane; i < slab / 8; i += 32) {
      const uint4 v = __ldcs(b + i);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x12345u) out[0] = acc;
}
void bw(torch::Tensor w, torch::Tensor out, int64_t pattern, int64_t ntiles, int64_t ntl, int64_t items) {
  const size_t estride = (size_t)256 * ntiles * 32;
  dim3 grid(ntiles / ntl, items);
  bw_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)w.data_ptr(), (unsigned*)out.data_ptr(), (int)pattern, (int)ntiles, (int)ntl, (int)items, 32, estride);
}
'''
_BW_CPP = 'void bw(torch::Tensor w, torch::Tensor out, int64_t pattern, int64_t ntiles, int64_t ntl, int64_t items);'


def _mq_lab(state, weights):
    from torch.utils.cpp_extension import load_inline
    mod = load_inline("muse_mqbw", _BW_CPP, cuda_sources=_BW_SRC, functions=["bw"], extra_cuda_cflags=["-O3"], verbose=False)
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    out = torch.zeros(4, dtype=torch.int32, device='cuda')
    bank = state["banks"][(0, 32)]
    for name, w, ntiles in (('w13', bank['w13'], 128), ('w2', bank['w2'], 64)):
        for items in (3, 12, 24):
            row = []
            for pattern in (0, 1):
                for ntl in (1, 2, 4):
                    t = _lab_time(lambda: mod.bw(w, out, pattern, ntiles, ntl, items), flush)
                    gb = items * 256 * ntiles * 512 / 1e9
                    row.append('p%d/ntl%d=%.1fus(%.2fTB/s)' % (pattern, ntl, t, gb / t * 1e3))
            print('LABBW', name, 'items=%d' % items, ' '.join(row), flush=True)
