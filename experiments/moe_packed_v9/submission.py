# okc: problem=muse_moe_w4a16
"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _prepare_marlin(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


@triton.jit
def _post_expert_norm_weighted_sum_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + tok * TOPK + k) >= 0
        if live:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            acc += y * s

    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _finish_experts(expert_output, topk_scores, norm_weight=None, eps=1e-8,
                    weight_first=True, apply_norm=True, slot_expert_ids=None):
    tokens, topk, hidden = expert_output.shape
    if tokens >= 32:
        return fused_post_expert_norm_weighted_sum(
            expert_output, topk_scores, norm_weight, eps, weight_first,
            apply_norm, slot_expert_ids)
    output = torch.empty((tokens, hidden), device=expert_output.device, dtype=expert_output.dtype)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        expert_output, norm_weight if norm_weight is not None else expert_output,
        topk_scores, output,
        slot_expert_ids if slot_expert_ids is not None else expert_output,
        TOPK=topk, H=hidden, eps=eps,
        APPLY_NORM=apply_norm, HAS_WEIGHT=norm_weight is not None,
        WEIGHT_FIRST=weight_first, HAS_IDS=slot_expert_ids is not None,
        BLOCK_H=triton.next_power_of_2(hidden),
        num_warps=8 if hidden >= 4096 else 4, num_stages=2)
    return output


def _marlin_kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    if num_tokens >= 32:
        block_size_m = 16
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _finish_experts(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


@triton.jit
def _decode_words(words, scales):
    a,b,c,d,e,f,g,h = tl.inline_asm_elementwise(
        r"""{
.reg .b32 packed, q, mag, bits, tmp, sign;
.reg .b16 h, scale;
.reg .pred small;
mov.b32 packed, $8;
mov.b16 scale, $9;
shr.u32 q, packed, 0;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $0, h, scale;
shr.u32 q, packed, 16;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $1, h, scale;
shr.u32 q, packed, 8;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $2, h, scale;
shr.u32 q, packed, 24;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $3, h, scale;
shr.u32 q, packed, 4;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $4, h, scale;
shr.u32 q, packed, 20;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $5, h, scale;
shr.u32 q, packed, 12;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $6, h, scale;
shr.u32 q, packed, 28;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $7, h, scale;
}""",
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h",
        args=[words, scales], dtype=(tl.bfloat16,)*8, is_pure=True, pack=1)
    p0, p1 = tl.join(a,b), tl.join(c,d)
    p2, p3 = tl.join(e,f), tl.join(g,h)
    return tl.join(tl.join(p0,p1),tl.join(p2,p3)).reshape(words.shape[0], words.shape[1]*8)


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
                kw = (block * BK // 8) + tl.arange(0, BK // 8)
                ptr = expert * (2 * I * H // 8) + cols[:, None] * (H // 8) + kw[None, :]
                wg_packed = tl.load(W + ptr).to(tl.uint32)
                wu_packed = tl.load(W + ptr + I * H // 8).to(tl.uint32)
                sptr = expert * (2 * I * H // 16) + cols[:, None] * (H // 16) + kw[None, :] // 2
                sg = tl.load(SF + sptr).to(tl.bfloat16)
                su = tl.load(SF + sptr + I * H // 16).to(tl.bfloat16)
                wg = _decode_words(wg_packed, sg)
                wu = _decode_words(wu_packed, su)
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
                kw = (block * BK // 8) + tl.arange(0, BK // 8)
                ptr = expert * (H * I // 8) + cols[:, None] * (I // 8) + kw[None, :]
                packed = tl.load(W + ptr).to(tl.uint32)
                sptr = expert * (H * I // 16) + cols[:, None] * (I // 16) + kw[None, :] // 2
                sf = tl.load(SF + sptr).to(tl.bfloat16)
                w = _decode_words(packed, sf)
                acc = tl.dot(x, tl.trans(w), acc)
            output = acc * tl.load(GS + expert)
            tl.store(Y + pairs[:, None] * H + cols[None, :], output, live[:, None])


def prepare(weights):
    # Preserve the original representation for the native path. Do not
    # pre-dequantize or lower the precision of either the codes or scales.
    raw = {name: weights[name][:32].clone() for name in
           ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
    raw['w13'] = raw['w13'].view(torch.int32)
    raw['w2'] = raw['w2'].view(torch.int32)
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
    bm, bn, bk = 16, 64, 64
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
