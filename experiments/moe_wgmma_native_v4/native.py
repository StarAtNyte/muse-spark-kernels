@triton.jit
def _fp4_values(code):
    u = code & 7
    magnitude_bits = tl.where(u < 2, u * 16128, (u << 6) + 16128)
    bits = magnitude_bits | ((code & 8) << 12)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)



@triton.jit
def _w4_gate_up(X, W, SF, GS, SORTED, EXPERT, NPAD, INTER,
                 M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
                 TOPK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr):
    mb, nb = tl.program_id(0), tl.program_id(1)
    padded = tl.load(NPAD)
    if mb * BM < padded:
        expert = tl.load(EXPERT + mb)
        if expert >= 0:
            rows = mb * BM + tl.arange(0, BM)
            pairs = tl.load(SORTED + rows)
            live = pairs < M * TOPK
            tokens = pairs // TOPK
            cols = nb * BN + tl.arange(0, BN)
            kk = tl.arange(0, BK)
            gate = tl.full((BM, BN), 0., tl.float32)
            up = tl.full((BM, BN), 0., tl.float32)
            for block in range(H // BK):
                k = block * BK + kk
                x = tl.load(X + tokens[:, None] * H + k[None, :], live[:, None], 0.)
                ptr = expert * (2 * I * H // 2) + cols[:, None] * (H // 2) + k[None, :] // 2
                packed_gate = tl.load(W + ptr)
                packed_up = tl.load(W + ptr + I * H // 2)
                shift = (k[None, :] & 1) * 4
                code_g = (packed_gate.to(tl.int32) >> shift) & 15
                code_u = (packed_up.to(tl.int32) >> shift) & 15
                sptr = expert * (2 * I * H // 16) + cols[:, None] * (H // 16) + k[None, :] // 16
                scale_g = tl.load(SF + sptr).to(tl.float32)
                scale_u = tl.load(SF + sptr + I * H // 16).to(tl.float32)
                # E2M1 times E4M3 is exactly representable in BF16. The
                # per-expert FP32 global scale is applied after accumulation.
                wg = (_fp4_values(code_g) * scale_g).to(tl.bfloat16)
                wu = (_fp4_values(code_u) * scale_u).to(tl.bfloat16)
                gate = tl.dot(x, tl.trans(wg), gate)
                up = tl.dot(x, tl.trans(wu), up)
            global_scale = tl.load(GS + expert)
            gate = (gate * global_scale).to(tl.bfloat16).to(tl.float32)
            up = (up * global_scale).to(tl.bfloat16).to(tl.float32)
            activation = gate * tl.sigmoid(gate) * up
            tl.store(INTER + pairs[:, None] * I + cols[None, :], activation, live[:, None])


@triton.jit
def _w4_down(INTER, W, SF, GS, SORTED, EXPERT, NPAD, Y,
              M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
              TOPK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr):
    mb, nb = tl.program_id(0), tl.program_id(1)
    padded = tl.load(NPAD)
    if mb * BM < padded:
        expert = tl.load(EXPERT + mb)
        if expert >= 0:
            rows = mb * BM + tl.arange(0, BM)
            pairs = tl.load(SORTED + rows)
            live = pairs < M * TOPK
            cols = nb * BN + tl.arange(0, BN)
            kk = tl.arange(0, BK)
            acc = tl.full((BM, BN), 0., tl.float32)
            for block in range(I // BK):
                k = block * BK + kk
                x = tl.load(INTER + pairs[:, None] * I + k[None, :], live[:, None], 0.)
                ptr = expert * (H * I // 2) + cols[:, None] * (I // 2) + k[None, :] // 2
                packed = tl.load(W + ptr)
                code = (packed.to(tl.int32) >> ((k[None, :] & 1) * 4)) & 15
                sptr = expert * (H * I // 16) + cols[:, None] * (I // 16) + k[None, :] // 16
                scale = tl.load(SF + sptr).to(tl.float32)
                w = (_fp4_values(code) * scale).to(tl.bfloat16)
                acc = tl.dot(x, tl.trans(w), acc)
            output = acc * tl.load(GS + expert)
            tl.store(Y + pairs[:, None] * H + cols[None, :], output, live[:, None])


def prepare(weights):
    # Preserve the original representation for the native path. Do not
    # pre-dequantize or lower the precision of either the codes or scales.
    raw = {name: weights[name][:32].clone() for name in
           ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
    state = _prepare_marlin(weights)
    state['raw'] = raw
    return state


def kernel_fn(data, state):
    x = data['hidden_states']
    m, h = x.shape
    if m < 64 or m > 512:
        return _marlin_kernel_fn(data, state)
    cfg = data['config']
    i, topk = cfg['d_expert'], cfg['top_k']
    bank = state['banks'][(cfg['local_start'], cfg['local_count'])]
    ids = bank['expert_map'][data['topk_ids']].contiguous()
    bm, bn, bk = 64, 32, 64
    sorted_ids, experts, padded = moe_align_block_size(
        ids, bm, bank['count'], ignore_invalid_expert=True)
    blocks = triton.cdiv(sorted_ids.numel(), bm)
    intermediate = torch.empty((m*topk, i), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((m*topk, h), device=x.device, dtype=torch.bfloat16)
    raw = state['raw']
    _w4_gate_up[(blocks, triton.cdiv(i, bn))](
        x, raw['w13'], raw['w13_sf'], raw['w13_gs'], sorted_ids, experts, padded,
        intermediate, m, h, i, topk, bm, bn, bk, num_warps=4, num_stages=2)
    _w4_down[(blocks, triton.cdiv(h, bn))](
        intermediate, raw['w2'], raw['w2_sf'], raw['w2_gs'], sorted_ids, experts,
        padded, y, m, h, i, topk, bm, bn, bk, num_warps=4, num_stages=2)
    return fused_post_expert_norm_weighted_sum(
        y.view(m, topk, h), data['topk_weights'], norm_weight=state['norm_w'],
        eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=ids)
