"""Decoder-layer candidate: v4 + weight-major MoE + dense bf16 GEMMs + CUDA attention (fd2 / fused sliding).

Environment knobs (python literals): WM_CFG_SPEC, DENSE_CFG, FD_POLICY, SLIDE_STAGES, USE_DENSE, USE_ATTN."""
import os
import sys

sys.path.insert(0, "moe_wm")
sys.path.insert(0, "dense")
sys.path.insert(0, "attn")
import attn_cuda as A  # noqa: E402
import dense_cuda as D  # noqa: E402
import wm_cuda  # noqa: E402

src = open("../experiments/decode_attention_hybrid_v4/submission.py").read()

# ---------------- MoE block ----------------
repack_src = open("moe_wm/wm_cuda.py").read().split("def repack(w, sf):", 1)[1]
repack_src = "def _wm_repack(w, sf):" + repack_src.replace("    import torch\n\n", "")
wm_cfg = os.environ.get("WM_CFG_SPEC", "[(8, (2, 4, 0)), (16, (2, 4, 0)), (10**9, (4, 4, 1))]")
moe_glue = open("moe_wm/build_decoder.py").read().split("glue = r'''", 1)[1].split("'''.replace(\"__CFG__\", cfg)", 1)[0].replace("__CFG__", wm_cfg)
moe_block = ("\n# ---- weight-major NVFP4 experts (CUDA) ----\n_WM_CUDA = r'''" + wm_cuda.CUDA_SRC + wm_cuda.TORCH_GLUE +
             "'''\n_WM_CPP = r'''" + wm_cuda.CPP_SRC + "'''\n\n" + repack_src + "\n" + moe_glue + "\n")

# ---------------- dense block ----------------
dense_repack = "def _dense_repack(w):" + open("dense/dense_cuda.py").read().split("def dense_repack(w):", 1)[1]
dense_cfg = os.environ.get("DENSE_CFG", "{}")
dense_block = r'''
# ---- weight-streaming dense bf16 GEMM (CUDA) ----
_DENSE_CUDA = r\'\'\'''' + D.CUDA_SRC + D.TORCH_GLUE + r'''\'\'\'
_DENSE_CPP = r\'\'\'''' + D.CPP_SRC + r'''\'\'\'
''' + dense_repack + r'''
_DMOD = None
_DENSE_PDL = True
_DENSE_PF = False
# (layer) -> [(max tokens, (ntb, nch, nw, s, ksplit, fix))]
_DENSE_CFG = __DCFG__


def _dense_module():
    global _DMOD
    if _DMOD is None:
        from torch.utils.cpp_extension import load_inline
        _DMOD = load_inline("muse_dense_bf16", _DENSE_CPP, cuda_sources=_DENSE_CUDA,
                            functions=["dense_gemm", "dense_set_pdl_t", "dense_set_pf_t"],
                            extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
        _DMOD.dense_set_pdl_t(int(_DENSE_PDL and torch.cuda.get_device_capability()[0] >= 9))
        _DMOD.dense_set_pf_t(int(_DENSE_PF and torch.cuda.get_device_capability()[0] >= 9))
    return _DMOD


def _dense_prepare(weights, device):
    state = {"mod": _dense_module(), "tiles": {}, "cnt": {}}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state["tiles"][name] = _dense_repack(weights[name].contiguous())
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["tiles"]["router"] = _dense_repack(torch.cat([hi, lo], 0).contiguous())
    for name in state["tiles"]:
        state["cnt"][name] = torch.zeros(4096, dtype=torch.int32, device=device)
    return state


def _dense_cfg(layer, m):
    for bound, cfg in _DENSE_CFG.get(layer, ()):
        if m <= bound:
            return cfg
    return None


def _dgemm(x, layer, n, state):
    """x @ W[:n].T through the CUDA weight-streaming kernel when configured, else cuBLAS."""
    m = x.shape[0]
    cfg = _dense_cfg(layer, m)
    if cfg is None:
        w = state["dense"][layer]
        return _gemm(x, w[:n] if layer == "unpooler" else w)
    ntb, nch, nw, s, ksplit, fix = cfg
    d = state["dn"]
    router = layer == "router"
    ntot = 512 if router else n
    wt = d["tiles"][layer][: ntot // 16]
    part = torch.empty((ksplit, m, ntot), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    d["mod"].dense_gemm(ntb, nch, nw, s, wt, x.contiguous(), part, out, ksplit, 1 if router else 0, n, d["cnt"][layer], fix)
    return out

'''.replace("__DCFG__", dense_cfg)
dense_block = dense_block.replace("\\'\\'\\'", "'''")

# ---------------- attention block ----------------
glue_attn = open("attn/glue_attn.py").read()
fd_policy = os.environ.get("FD_POLICY", "{}")
glue_attn = glue_attn.split("def kernel_fn(data):", 1)[0].replace("__POLICY__", fd_policy)
slide_stages = int(os.environ.get("SLIDE_STAGES", "0"))   # 0 = disabled
glue_cuda = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
"""
b = open("attn/build_attn.py").read()
glue_cuda = b.split("glue_cuda = r'''", 1)[1].split("'''\ncpp = '''", 1)[0]
cpp = b.split("cpp = '''", 1)[1].split("'''\ndefault", 1)[0]
attn_block = ("\n# ---- CUDA attention (fd2 / fused sliding) ----\n_FD_CUDA = r'''" + A.CUDA_SRC + glue_cuda + "'''\n_FD_CPP = r'''" + cpp + "'''\n" +
              glue_attn + "\n_SLIDE_STAGES = %d\n" % slide_stages)

marker = "def _experts(x, topk_weights, topk_ids, state):"
assert marker in src
src = src.replace(marker, moe_block + attn_block + dense_block + "\n" + marker, 1)

# prepare(): add module states
src = src.replace('        "side_stream": torch.cuda.Stream(device),\n',
                  '        "side_stream": torch.cuda.Stream(device),\n        "wm": _wm_prepare(weights),\n        "dn": _dense_prepare(weights, device),\n', 1)
old = '    bank = state["bank"]\n    num_tokens, size_k = x.shape\n    size_n = bank["size_n"]'
assert old in src
src = src.replace(old, '    if x.shape[0] <= WM_MAX_TOKENS:\n        return _experts_wm(x, topk_weights, topk_ids, state)\n' + old, 1)

# ---- kernel_fn surgery ----
if os.environ.get("PF_ON", "0") == "1":
    old_q = '    qkv = _dgemm(data["attn_input"], "qkv_gate", 2304, state)' if os.environ.get("USE_DENSE", "1") == "1" else '    qkv = _gemm(data["attn_input"], dense["qkv_gate"])'
    new_q = old_q + '''
    _pf_main = torch.cuda.current_stream()
    state["pf_stream"].wait_stream(_pf_main)
    with torch.cuda.stream(state["pf_stream"]):
        _fd_module().pf_t(dense["o"])
        _fd_module().pf_t(dense["pooler"][:PF_ROWS])
'''
    assert old_q in src
    src = src.replace(old_q, new_q, 1)
    src = src.replace('    return next_attn_input, stream, topk_ids', '    _pf_main.wait_stream(state["pf_stream"])\n    return next_attn_input, stream, topk_ids', 1)
    src = src.replace('        "side_stream": torch.cuda.Stream(device),\n', '        "side_stream": torch.cuda.Stream(device),\n        "pf_stream": torch.cuda.Stream(device),\n', 1)
    src = src.replace('_SLIDE_STAGES = ', 'PF_ROWS = %d\n_SLIDE_STAGES = ' % int(os.environ.get("PF_ROWS", "1536")), 1)

use_dense = os.environ.get("USE_DENSE", "1") == "1"
use_attn = os.environ.get("USE_ATTN", "1") == "1"
if use_dense:
    src = src.replace('qkv = _gemm(data["attn_input"], dense["qkv_gate"])', 'qkv = _dgemm(data["attn_input"], "qkv_gate", 2304, state)', 1)
    src = src.replace('attn_output = _gemm(o, dense["o"])  # all-reduce: identity', 'attn_output = _dgemm(o, "o", HIDDEN, state)  # all-reduce: identity', 1)
    src = src.replace('expert_input = _gemm(ffn_input, dense["pooler"])', 'expert_input = _dgemm(ffn_input, "pooler", D_EXPERT, state)', 1)
    src = src.replace('unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])', 'unpooled = _dgemm(expert_output, "unpooler", UNPOOL_SLICE, state)', 1)
    src = src.replace('ffn_output = _gemm(expert_output, dense["unpooler"])', 'ffn_output = _dgemm(expert_output, "unpooler", HIDDEN, state)', 1)
    # router for more than 8 tokens: hi/lo through the dense kernel when configured
    src = src.replace('        if split_router and tokens <= 256:\n            logits = fused_split_router(ffn_input, state["router_hi_t"], state["router_lo_t"])',
                      '        if split_router and _dense_cfg("router", tokens) is not None:\n            logits = _dgemm(ffn_input, "router", 256, state)\n        elif split_router and tokens <= 256:\n            logits = fused_split_router(ffn_input, state["router_hi_t"], state["router_lo_t"])', 1)
if use_attn:
    old_att = "    _prepare_qkv[(tokens, triton.cdiv(hq + hkv, 4))](\n        qkv, data[\"positions\"], data[\"cache_loc\"], data[\"k_scale\"], data[\"v_scale\"],\n        q, k_cache, v_cache, hq, hkv, d, qkv.stride(0), cfg[\"rope_theta\"],\n        scaled, 4, num_warps=4, enable_fp_fusion=False)\n    if (cfg['window'] and batch <= 4) or (not cfg['window'] and length == 1):"
    assert old_att in src, "attention anchor"
    new_att = ("    fused_slide = _SLIDE_STAGES and cfg['window'] and length == 1 and cfg['page_size'] & (cfg['page_size'] - 1) == 0\n"
               "    if not fused_slide:\n"
               "        _prepare_qkv[(tokens, triton.cdiv(hq + hkv, 4))](\n"
               "            qkv, data[\"positions\"], data[\"cache_loc\"], data[\"k_scale\"], data[\"v_scale\"],\n"
               "            q, k_cache, v_cache, hq, hkv, d, qkv.stride(0), cfg[\"rope_theta\"],\n"
               "            scaled, 4, num_warps=4, enable_fp_fusion=False)\n"
               "    fd = None if fused_slide else _fd_choice(cfg, data['page_table'].shape[1] * cfg['page_size'])\n"
               "    if fused_slide:\n"
               "        o = _slide_forward(data, _SLIDE_STAGES, qkv).view(tokens, hq*d)\n"
               "    elif fd is not None and cfg['page_size'] & (cfg['page_size'] - 1) == 0:\n"
               "        o = _fd_forward(data, q, qkv, fd[0], fd[1], variant=fd[2]).view(tokens, hq*d)\n"
               "    elif (cfg['window'] and batch <= 4) or (not cfg['window'] and length == 1):")
    src = src.replace(old_att, new_att, 1)
open(sys.argv[1] if len(sys.argv) > 1 else "decoder_all.py", "w").write(src)
print("wrote")
