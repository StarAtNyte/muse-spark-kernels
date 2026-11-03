_smoe_parent_prepare = prepare


def prepare(weights):
    state = _smoe_parent_prepare(weights)
    lo, cnt = state["lo"], state["cnt"]
    if SMOE_ON and cnt <= 32 and _smoe_scales_ok(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt]):
        dev = weights["w13"].device
        state["smoe"] = _smoe_prepare(weights, lo, cnt, dev)
        _smoe_ext()
        torch.cuda.synchronize()
    return state


_smoe_parent_kernel = kernel_fn


def kernel_fn(data, state):
    x = data["hidden_states"]
    if "smoe" in state and x.shape[0] <= SMOE_MAX_M:
        _bind_context(x)
        cfg = data["config"]
        assert cfg["local_start"] == state["lo"] and cfg["local_count"] == state["cnt"]
        ids = data["topk_ids"].to(torch.int32).contiguous()
        tw = data["topk_weights"].to(torch.float32).contiguous()
        return _smoe_call(state["smoe"], x.contiguous(), ids, tw, state["lo"], state["cnt"], state["grid"])
    return _smoe_parent_kernel(data, state)
