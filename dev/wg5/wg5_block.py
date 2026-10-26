"""Shared wg5 (wgmma verify attention) glue strings for build_attn.py / build_decoder_all.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wg5_cuda as W5

GLUE_CUDA = r'''
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

CPP = '''void wg5_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,
               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg);'''

PY = r'''
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
'''


def block(minb="8", splits="16"):
    py = PY.replace("__WGMINB__", str(minb)).replace("__WGSPLITS__", str(splits))
    return ("_WG_CUDA = r'''" + W5.CUDA_SRC + GLUE_CUDA + "'''\n_WG_CPP = r'''" + CPP + "'''\n" + py)
