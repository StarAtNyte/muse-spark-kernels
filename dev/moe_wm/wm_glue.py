_WM_MOD = None
_WM_DBG = 0
_WM_PDL = True
_WM_FIN = False
_WM_NOLS = None
_WM_EXTRA = []


def _wm_module():
    global _WM_MOD
    if _WM_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _WM_MOD = load_inline("muse_wm_moe", _WM_CPP, cuda_sources=_WM_CUDA, functions=["wm_route", "wm_gemm_run", "wm_set_pdl_t", "wm_set_persist_t", "wm_finish", "wm_pf"],
                              extra_cuda_cflags=["-O3", "--use_fast_math"] + _WM_EXTRA, verbose=False)
        _WM_MOD.wm_set_pdl_t(int(_WM_PDL and torch.cuda.get_device_capability()[0] >= 9))
    return _WM_MOD


def prepare(weights):
    global _WM_NOLS
    state = _prepare_marlin(weights)
    mod = _wm_module()
    _WM_NOLS = torch.empty(0, dtype=torch.int32, device=weights['w13'].device)
    w13, s13 = _wm_repack(weights["w13"][:32], weights["w13_sf"][:32])
    w2, s2 = _wm_repack(weights["w2"][:32], weights["w2_sf"][:32])
    state["wm"] = {"mod": mod, "w13": w13, "s13": s13, "gs13": weights["w13_gs"][:32].contiguous(),
                   "w2": w2, "s2": s2, "gs2": weights["w2_gs"][:32].contiguous()}
    torch.cuda.synchronize()
    return state


WM_MAX_TOKENS = 256
_WM_PF = 0          # experts whose down-projection weights are prefetched into L2 during the gate/up GEMM
_WM_PF_MINM = 1
# (m upper bound) -> (tiles per warp, warps per CTA, cp.async ring)
_WM_CFG = __WM_CFG__


def _wm_cfg(m):
    for bound, cfg in _WM_CFG:
        if m <= bound:
            return cfg
    return _WM_CFG[-1][1]


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    if (m > WM_MAX_TOKENS or cfg["local_start"] != 0 or cfg["local_count"] != 32 or cfg["d_hidden"] != 4096
            or cfg["d_expert"] != 4096):
        return _marlin_kernel_fn(data, state)
    wm = state["wm"]
    mod = wm["mod"]
    x = x.contiguous()
    nch = 1 if m <= 8 else 2 if m <= 16 else 4
    lmax = (m + 7) // 8 * 8
    _c = _wm_cfg(m)
    nt, nw, ring = _c[:3]
    nt1 = _c[3] if len(_c) > 3 else nt
    dev = x.device
    ids = data["topk_ids"].contiguous()
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(512, dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    mod.wm_route(ids, _WM_NOLS, 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
    if _WM_PF > 0 and m >= _WM_PF_MINM:
        mod.wm_pf(wm["w2"], wm["s2"], act, nact, _WM_PF, 1)
    maxslots = min(m * TOP_K, 32 + (m * TOP_K) // (nch * 8))
    inter = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)
    y = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)
    mod.wm_gemm_run(0, nch, nt, nw, ring, _WM_DBG, wm["w13"], wm["s13"], wm["gs13"], x, inter, cnt, lst, act, nact, woff, maxslots, lmax)
    mod.wm_gemm_run(1, nch, nt1, nw, ring, _WM_DBG, wm["w2"], wm["s2"], wm["gs2"], inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    if _WM_FIN:
        out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
        mod.wm_finish(y, topk_weights, lids, state["norm_w"], out, POST_NORM_EPS)
        return out
    return _finish_experts(
        y.view(m, TOP_K, 4096), topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=lids.view(m, TOP_K))
