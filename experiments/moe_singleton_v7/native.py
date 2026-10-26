@triton.jit
def _single_gate_up(X, IDS, LOCAL, W, SF, GS, INTER,
                    H: tl.constexpr, I: tl.constexpr, LO: tl.constexpr,
                    COUNT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    nb, slot = tl.program_id(0), tl.program_id(1)
    expert = tl.load(IDS + slot) - LO
    live = (expert >= 0) & (expert < COUNT)
    if nb == 0:
        tl.store(LOCAL + slot, tl.where(live, expert, -1))
    if live:
        rows = nb * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        gate = tl.full((BN, BK), 0., tl.float32)
        up = tl.full((BN, BK), 0., tl.float32)
        for block in range(H // BK):
            k = block * BK + kk
            x = tl.load(X + k).to(tl.float32)
            wp = expert * (2 * I * H // 2) + rows[:, None] * (H // 2) + k[None, :] // 2
            gp = tl.load(W + wp).to(tl.int32)
            up_packed = tl.load(W + wp + I * H // 2).to(tl.int32)
            shift = (k[None, :] & 1) * 4
            cg, cu = (gp >> shift) & 15, (up_packed >> shift) & 15
            sp = expert * (2 * I * H // 16) + rows[:, None] * (H // 16) + k[None, :] // 16
            sg = tl.load(SF + sp).to(tl.float32)
            su = tl.load(SF + sp + I * H // 16).to(tl.float32)
            wg, wu = _fp4_values(cg) * sg, _fp4_values(cu) * su
            gate = tl.fma(wg, x[None, :], gate)
            up = tl.fma(wu, x[None, :], up)
        gs = tl.load(GS + expert)
        g = (tl.sum(gate, 1) * gs).to(tl.bfloat16).to(tl.float32)
        u = (tl.sum(up, 1) * gs).to(tl.bfloat16).to(tl.float32)
        intermediate = g * tl.sigmoid(g) * u
        tl.store(INTER + slot * I + rows, intermediate)


@triton.jit
def _single_down(INTER, IDS, W, SF, GS, Y,
                 H: tl.constexpr, I: tl.constexpr, LO: tl.constexpr,
                 COUNT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    nb, slot = tl.program_id(0), tl.program_id(1)
    expert = tl.load(IDS + slot) - LO
    if (expert >= 0) & (expert < COUNT):
        rows = nb * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.full((BN, BK), 0., tl.float32)
        for block in range(I // BK):
            k = block * BK + kk
            x = tl.load(INTER + slot * I + k).to(tl.float32)
            wp = expert * (H * I // 2) + rows[:, None] * (I // 2) + k[None, :] // 2
            packed = tl.load(W + wp).to(tl.int32)
            code = (packed >> ((k[None, :] & 1) * 4)) & 15
            sp = expert * (H * I // 16) + rows[:, None] * (I // 16) + k[None, :] // 16
            sf = tl.load(SF + sp).to(tl.float32)
            w = _fp4_values(code) * sf
            acc = tl.fma(w, x[None, :], acc)
        output = tl.sum(acc, 1) * tl.load(GS + expert)
        tl.store(Y + slot * H + rows, output)


def prepare(weights):
    raw = {name: weights[name][:32].clone() for name in
           ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
    state = _prepare_fallback(weights)
    state['raw'] = raw
    return state


def kernel_fn(data, state):
    x = data['hidden_states']
    if x.shape[0] != 1:
        return _fallback_kernel(data, state)
    cfg = data['config']
    h, i, topk = x.shape[1], cfg['d_expert'], cfg['top_k']
    start, count = cfg['local_start'], cfg['local_count']
    raw = state['raw']
    intermediate = torch.empty((topk, i), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((1, topk, h), device=x.device, dtype=torch.bfloat16)
    ids = torch.empty((1, topk), device=x.device, dtype=torch.int32)
    bn, bk = 8, 256
    _single_gate_up[(triton.cdiv(i, bn), topk)](
        x, data['topk_ids'], ids, raw['w13'], raw['w13_sf'], raw['w13_gs'],
        intermediate, h, i, start, count, bn, bk, num_warps=4, num_stages=2)
    _single_down[(triton.cdiv(h, bn), topk)](
        intermediate, data['topk_ids'], raw['w2'], raw['w2_sf'], raw['w2_gs'],
        y, h, i, start, count, bn, bk, num_warps=4, num_stages=2)
    return _finish_experts(y, data['topk_weights'], norm_weight=state['norm_w'],
                           eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=ids)
