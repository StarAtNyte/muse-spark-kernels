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

if os.environ.get("WG5", "0") == "1":
    sys.path.insert(0, "../wg5")
    import wg5_cuda as W5
    wg_glue_cuda = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int wg5_run(const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*, float*,
                       int, int, int, int, int, int, int, int, float, int, int, cudaStream_t);
void wg5_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,
               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg) {
  int r = wg5_run(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(), seqlens.data_ptr<int>(), ks.data_ptr<float>(),
                  vs.data_ptr<float>(), part.data_ptr<float>(), lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ,
                  (int)table.size(1), (int)page, (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, (int)dbg,
                  at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "wg5");
}
'''
    wg_cpp = '''void wg5_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,
               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg);'''
    wg_py = r'''
_WG_MOD = None
_WG_ON = True
_WG_MINB = __WGMINB__
_WG_SPLITS = __WGSPLITS__


def _wg_module():
    global _WG_MOD
    if _WG_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _WG_MOD = load_inline("muse_wg5_attn", _WG_CPP, cuda_sources=_WG_CUDA, functions=["wg5_run_t"],
                              extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _WG_MOD


def _wg_forward(data, q, qkv, splits):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    span = data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), 64) * 64
    partial = torch.empty((batch, hkv, splits, 64, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, 64), device=qkv.device, dtype=torch.float32)
    _wg_module().wg5_run_t(q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'], data['k_scale'],
                           data['v_scale'], partial, lse, length, hq, cfg['page_size'], 0, splits, chunk,
                           cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0), 0)
    result = torch.empty((batch, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, 64, num_warps=4)
    return result
'''.replace("__WGMINB__", os.environ.get("WG_MINB", "8")).replace("__WGSPLITS__", os.environ.get("WG_SPLITS", "16"))
    out = out.replace("# ---- CUDA split-KV flash decode ----", "# ---- CUDA split-KV flash decode ----\n_WG_CUDA = r'''" + W5.CUDA_SRC + wg_glue_cuda + "'''\n_WG_CPP = r'''" + wg_cpp + "'''\n" + wg_py, 1)
    hook = "    if choice is None or ctx_bound > (1 << 20) + 64:"
    assert out.count(hook) == 1
    out = out.replace(hook, "    if (_WG_ON and cfg['q_len'] == 8 and not cfg['window'] and cfg['batch'] >= _WG_MINB and cfg['num_kv_heads'] == 2\n            and cfg['num_q_heads'] == 16 and ctx_bound <= (1 << 20) + 64):\n        qkv, q = _fd_prepare(data)\n        return _wg_forward(data, q, qkv, _WG_SPLITS)\n" + hook)
open(sys.argv[1], "w").write(out)
print("wrote")
