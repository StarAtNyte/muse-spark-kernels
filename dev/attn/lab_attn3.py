_GB_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
__global__ void gather_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc, const int* __restrict__ rows, int n, int lpr, int khmode, unsigned* out) {
  const int kh = khmode == 2 ? blockIdx.y : khmode;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthr = gridDim.x * blockDim.x;
  const int slots = nthr / lpr;
  const int c = tid % lpr, slot = tid / lpr;
  unsigned acc = 0;
#pragma unroll 4
  for (int i = slot; i < n; i += slots) {
    const long r = rows[i];
    const long off = lpr == 4 ? (r * 2 + kh) * 4 + c : r * 8 + c;
    const uint4 a = __ldg(kc + off), b = __ldg(vc + off);
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
  }
  if (acc == 0x12345u) out[0] = acc;
}
void gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor rows, int64_t lpr, int64_t khmode, int64_t blocks, torch::Tensor out) {
  dim3 grid(blocks, khmode == 2 ? 2 : 1);
  gather_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), rows.data_ptr<int>(), (int)rows.numel(), (int)lpr, (int)khmode, (unsigned*)out.data_ptr());
}
'''
_GB_CPP = 'void gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor rows, int64_t lpr, int64_t khmode, int64_t blocks, torch::Tensor out);'


def _fd_lab():
    from torch.utils.cpp_extension import load_inline
    mod = load_inline("muse_gbench", _GB_CPP, cuda_sources=_GB_SRC, functions=["gather"], extra_cuda_cflags=["-O3"], verbose=False)
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    out = torch.zeros(4, dtype=torch.int32, device='cuda')
    data = _lab_data(32, 1, 'full', 'agentic', seed=321)
    pt = data['page_table']
    rows = pt[pt >= 0].contiguous()
    perm_sorted = torch.sort(rows)[0].contiguous()
    n = rows.numel()
    kc, vc = data['k_cache'], data['v_cache']

    def tm(fn, reps=8):
        ts = []
        fn()
        for _ in range(reps):
            flush.add_(1)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    for blocks in (132 * 4, 132 * 8, 132 * 16):
        res = []
        for name, lpr, khm, r in (('64B_kh0', 4, 0, rows), ('64B_kh1', 4, 1, rows), ('64B_both_concurrent', 4, 2, rows), ('128B_row', 8, 0, rows),
                                  ('64B_kh0_sorted', 4, 0, perm_sorted), ('128B_sorted', 8, 0, perm_sorted)):
            t = tm(lambda: mod.gather(kc, vc, r, lpr, khm, blocks, out))
            by = n * (128 if lpr == 8 else 64) * 2 * (2 if khm == 2 else 1)
            res.append('%s=%.1fus(%.2fTB/s)' % (name, t, by / t / 1e6))
        print('LABG n=%d blocks=%d' % (n, blocks), ' '.join(res), flush=True)
