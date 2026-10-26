_MQ_MOD = None
_MQ_PDL = True
_MQ_NOLS = None
_MQ_FIN = False
_MQ_S = __MQ_S__
_MQ_SX = 3
_MQ_LAB = False


def _mq_module():
    global _MQ_MOD
    if _MQ_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _MQ_MOD = load_inline("muse_mq_moe", _MQ_CPP, cuda_sources=_MQ_CUDA, functions=["wm_route", "mq_gemm_run", "mq_gemm_k2_run", "wm_set_pdl_t", "wm_finish"],
                              extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
        _MQ_MOD.wm_set_pdl_t(int(_MQ_PDL and torch.cuda.get_device_capability()[0] >= 9))
    return _MQ_MOD


def prepare(weights):
    global _MQ_NOLS
    state = _prepare_marlin(weights)
    mod = _mq_module()
    _MQ_NOLS = torch.empty(0, dtype=torch.int32, device=weights['w13'].device)
    bank = state["banks"][(0, 32)]
    # the Marlin tensors exactly as SGLang keeps them; the global scale is stored as gs * 2^119 in bf16
    state["mq"] = {"mod": mod, "w13": bank["w13"], "s13": bank["w13_s"], "gs13": (bank["w13_gs"].float() * 2.0 ** -111).contiguous(),
                   "w2": bank["w2"], "s2": bank["w2_s"], "gs2": (bank["w2_gs"].float() * 2.0 ** -111).contiguous()}
    torch.cuda.synchronize()
    if _MQ_LAB:
        _mq_lab(state, weights)
    return state


MQ_MAX_TOKENS = 256


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    if (m > MQ_MAX_TOKENS or cfg["local_start"] != 0 or cfg["local_count"] != 32 or cfg["d_hidden"] != 4096
            or cfg["d_expert"] != 4096):
        return _marlin_kernel_fn(data, state)
    mq = state["mq"]
    mod = mq["mod"]
    x = x.contiguous()
    nch = 1 if m <= 8 else 2 if m <= 16 else 4
    lmax = (m + 7) // 8 * 8
    dev = x.device
    ids = data["topk_ids"].contiguous()
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(512, dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    mod.wm_route(ids, _MQ_NOLS, 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
    maxslots = min(m * TOP_K, 32 + (m * TOP_K) // (nch * 8))
    inter = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)
    y = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)
    if m <= 16:
        mod.mq_gemm_k2_run(0, nch, _MQ_SX, mq["w13"], mq["s13"], mq["gs13"], x, inter, lst, act, maxslots, lmax)
        mod.mq_gemm_k2_run(1, nch, _MQ_SX, mq["w2"], mq["s2"], mq["gs2"], inter, y, lst, act, maxslots, lmax)
    else:
        mod.mq_gemm_run(0, nch, _MQ_S, mq["w13"], mq["s13"], mq["gs13"], x, inter, lst, act, maxslots, lmax)
        mod.mq_gemm_run(1, nch, _MQ_S, mq["w2"], mq["s2"], mq["gs2"], inter, y, lst, act, maxslots, lmax)
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    if _MQ_FIN:
        out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
        mod.wm_finish(y, topk_weights, lids, state["norm_w"], out, POST_NORM_EPS)
        return out
    return _finish_experts(
        y.view(m, TOP_K, 4096), topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=lids.view(m, TOP_K))
