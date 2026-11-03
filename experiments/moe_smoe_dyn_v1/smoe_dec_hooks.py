
_smoe_dec_parent_prepare = prepare


def prepare(weights):
    st = _smoe_dec_parent_prepare(weights)
    if SMOE_ON and _smoe_scales_ok(weights["w13_sf"][:32], weights["w2_sf"][:32]):
        sw = {"w13": weights["w13"].view(torch.uint8), "w13_sf": weights["w13_sf"], "w2": weights["w2"].view(torch.uint8),
              "w2_sf": weights["w2_sf"], "w13_gs": weights["w13_gs"], "w2_gs": weights["w2_gs"],
              "post_norm_w": weights["post_expert_norm"]}
        st["smoe"] = _smoe_prepare(sw, 0, 32, weights["w13"].device)
        _smoe_ext()
        torch.cuda.synchronize()
    return st


class _SmoeExt:
    """okc_run-compatible front for the SMOE kernel (writes the given output tensor)."""
    def __init__(self, sm):
        self.sm = sm

    def okc_set_pdl(self, on):
        pass

    def okc_run(self, w, scratch, x, ids, tw, out, lo, nloc, grid):
        sm = self.sm
        _smoe_ext().smoe_run(sm["w"], sm["gsA"], sm["gsB"], sm["pnw"], x.contiguous(), ids, tw, sm["hbuf"], sm["ybuf"],
                             sm["ss"], sm["ctr"], out, lo, nloc, grid)


_smoe_dec_parent_moe = _moe


def _moe(M=None, state=None):
    if state is not None and M is not None and M <= SMOE_MAX_M and "smoe" in state:
        return _SmoeExt(state["smoe"])
    return _smoe_dec_parent_moe(M, state)
