import os, sys
import attn_cuda as A

base = open("../../experiments/attention_fp8_hybrid_v7/submission.py").read()
base = base.replace("def kernel_fn(data):", "def _v7_kernel_fn(data):", 1)
if os.environ.get("FD_PDL", "0") == "1":
    o1 = "    token = tl.program_id(0)\n    heads = tl.program_id(1) * HEADS + tl.arange(0, HEADS)"
    o2 = "    t, qh = tl.program_id(0), tl.program_id(1)\n    b, token = t // LENGTH, t % LENGTH"
    assert base.count(o1) == 1 and base.count(o2) == 1
    base = base.replace(o1, "    tl.extra.cuda.gdc_launch_dependents()\n" + o1)
    base = base.replace(o2, "    tl.extra.cuda.gdc_wait()\n" + o2)
    base = base.replace("_merge_norm_gate[(tokens, hq)](\n            partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0),\n            splits, bm, num_warps=4)", "_merge_norm_gate[(tokens, hq)](\n            partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0),\n            splits, bm, num_warps=4)")
assert "_v7_kernel_fn" in base
glue_cuda = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int slide_run(int, const void*, const void*, const void*, void*, void*, const int*, const int*, const float*,
                         const float*, void*, int, int, int, int, int, int, float, float, int, int, cudaStream_t);
void slide_run_t(int64_t stages, torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
                 torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t HQ,
                 int64_t HKV, int64_t page, int64_t window, double theta, double sm_scale, int64_t scaled) {
  int r = slide_run((int)stages, qkv.data_ptr(), pos.data_ptr(), loc.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(),
                    seqlens.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(), out.data_ptr(), (int)seqlens.numel(),
                    (int)HQ, (int)HKV, (int)table.size(1), (int)page, (int)window, (float)theta, (float)sm_scale, (int)scaled,
                    (int)qkv.stride(0), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported slide config");
}
extern "C" void fd2_set_dbg(int);
void fd2_set_dbg_t(int64_t v) { fd2_set_dbg((int)v); }
extern "C" void fd_set_pdl(int);
void fd_set_pdl_t(int64_t v) { fd_set_pdl((int)v); }
extern "C" void pf_run(const void*, size_t, cudaStream_t);
void pf_t(torch::Tensor w) { pf_run(w.data_ptr(), (size_t)w.numel() * w.element_size(), at::cuda::getCurrentCUDAStream()); }
extern "C" int fd_run(int, int, int, const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*,
                      float*, int, int, int, int, int, int, int, int, int, float, int, cudaStream_t);

void fd_run_t(int64_t mt, int64_t stages, int64_t qk8, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table,
              torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L,
              int64_t HQ, int64_t HKV, int64_t page, int64_t window, int64_t splits, int64_t chunk, double sm_scale,
              int64_t scaled) {
  int r = fd_run((int)mt, (int)stages, (int)qk8, q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(),
                 seqlens.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(), part.data_ptr<float>(),
                 lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ, (int)HKV, (int)table.size(1), (int)page,
                 (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported fd config");
}
'''
cpp = '''
void pf_t(torch::Tensor w);
void fd2_set_dbg_t(int64_t v);
void fd_set_pdl_t(int64_t v);
void slide_run_t(int64_t stages, torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
                 torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t HQ,
                 int64_t HKV, int64_t page, int64_t window, double theta, double sm_scale, int64_t scaled);
void fd_run_t(int64_t mt, int64_t stages, int64_t qk8, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table,
              torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L,
              int64_t HQ, int64_t HKV, int64_t page, int64_t window, int64_t splits, int64_t chunk, double sm_scale,
              int64_t scaled);
'''
default = "{('full', False): [(1, 64, 4, 32)], ('sliding', False): [(1, 4, 4, 4)]}"
policy = os.environ.get("FD_POLICY", default)
glue = open("glue_attn.py").read().replace("__POLICY__", policy).replace("__PDL__", "True" if os.environ.get("FD_PDL", "0") == "1" else "False").replace("__PDLB__", os.environ.get("FD_PDLB", "1"))
out = base + "\n\n# ---- CUDA split-KV flash decode ----\n_FD_CUDA = r'''" + A.CUDA_SRC + glue_cuda + "'''\n_FD_CPP = r'''" + cpp + "'''\n" + glue
if os.environ.get("FD_LAB"):
    out += open("lab_attn.py").read() + "\n" + (open(os.environ["LAB2"]).read() if os.environ.get("LAB2") else "") + "\n_FD_LAB = True\n"
open(sys.argv[1], "w").write(out)
print("wrote")
