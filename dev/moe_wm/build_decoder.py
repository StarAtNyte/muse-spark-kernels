"""Decoder-layer candidate = decode_attention_hybrid_v4 + weight-major MoE experts."""
import os
import sys

import wm_cuda

src = open("../../experiments/decode_attention_hybrid_v4/submission.py").read()
repack_src = open("wm_cuda.py").read().split("def repack(w, sf):", 1)[1]
repack_src = "def _wm_repack(w, sf):" + repack_src.replace("    import torch\n\n", "")
cfg = os.environ.get("WM_CFG_SPEC", "[(8, (2, 4, 0)), (16, (2, 4, 0)), (10**9, (4, 4, 1))]")
glue = r'''
_WM_MOD = None
_WM_DBG = 0
_WM_EXTRA = []
_WM_PDL = True
_WM_FIN = False
_WM_CFG = __CFG__


def _wm_cfg(m):
    for bound, c in _WM_CFG:
        if m <= bound:
            return c
    return _WM_CFG[-1][1]


def _wm_module():
    global _WM_MOD
    if _WM_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _WM_MOD = load_inline("muse_wm_moe", _WM_CPP, cuda_sources=_WM_CUDA, functions=["wm_route", "wm_gemm_run", "wm_set_pdl_t", "wm_set_persist_t", "wm_finish", "wm_pf"],
                              extra_cuda_cflags=["-O3", "--use_fast_math"] + _WM_EXTRA, verbose=False)
        _WM_MOD.wm_set_pdl_t(int(_WM_PDL and torch.cuda.get_device_capability()[0] >= 9))
    return _WM_MOD


def _wm_prepare(weights):
    w13, s13 = _wm_repack(weights["w13"], weights["w13_sf"])
    w2, s2 = _wm_repack(weights["w2"], weights["w2_sf"])
    return {"mod": _wm_module(), "w13": w13, "s13": s13, "gs13": weights["w13_gs"].contiguous(),
            "w2": w2, "s2": s2, "gs2": weights["w2_gs"].contiguous(),
            "nols": torch.empty(0, dtype=torch.int32, device=weights["w13"].device)}


def _experts_wm(x, topk_weights, local_mod_ids, state):
    wm = state["wm"]
    mod = wm["mod"]
    m = x.shape[0]
    x = x.contiguous()
    nch = 1 if m <= 8 else 2 if m <= 16 else 4
    lmax = (m + 7) // 8 * 8
    _c = _wm_cfg(m)
    nt, nw, ring = _c[:3]
    nt1 = _c[3] if len(_c) > 3 else nt
    dev = x.device
    ids = local_mod_ids.to(torch.int32).contiguous()
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(512, dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    mod.wm_route(ids, wm["nols"], 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
    if _WM_PF > 0 and m >= _WM_PF_MINM:
        mod.wm_pf(wm["w2"], wm["s2"], act, nact, _WM_PF, 1)
    maxslots = min(m * TOP_K, 32 + (m * TOP_K) // (nch * 8))
    inter = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=dev)
    mod.wm_gemm_run(0, nch, nt, nw, ring, _WM_DBG, wm["w13"], wm["s13"], wm["gs13"], x, inter, cnt, lst, act, nact, woff, maxslots, lmax)
    mod.wm_gemm_run(1, nch, nt1, nw, ring, _WM_DBG, wm["w2"], wm["s2"], wm["gs2"], inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if _WM_FIN:
        out = torch.empty((m, D_EXPERT), dtype=torch.bfloat16, device=dev)
        mod.wm_finish(y, topk_weights, lids, state["post_expert_w"], out, POST_NORM_EPS)
        return out
    return _finish_experts(
        y.view(m, TOP_K, D_EXPERT), topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=lids.view(m, TOP_K))


WM_MAX_TOKENS = 256
_WM_PF = 0
_WM_PF_MINM = 1
'''.replace("__CFG__", cfg)

wm_block = ("\n# ---- weight-major NVFP4 experts (CUDA) ----\n_WM_CUDA = r'''" + wm_cuda.CUDA_SRC + wm_cuda.TORCH_GLUE +
            "'''\n_WM_CPP = r'''" + wm_cuda.CPP_SRC + "'''\n\n" + repack_src + "\n" + glue + "\n")

# insert block before _experts, hook prepare and _experts
marker = "def _experts(x, topk_weights, topk_ids, state):"
assert marker in src
src = src.replace(marker, wm_block + "\n" + marker, 1)
src = src.replace('        "side_stream": torch.cuda.Stream(device),\n', '        "side_stream": torch.cuda.Stream(device),\n        "wm": _wm_prepare(weights),\n', 1)
old = '    bank = state["bank"]\n    num_tokens, size_k = x.shape\n    size_n = bank["size_n"]'
assert old in src
src = src.replace(old, '    if x.shape[0] <= WM_MAX_TOKENS:\n        return _experts_wm(x, topk_weights, topk_ids, state)\n' + old, 1)
open(sys.argv[1] if len(sys.argv) > 1 else "decoder_candidate.py", "w").write(src)
print("wrote")
