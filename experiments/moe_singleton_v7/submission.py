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


def _prepare_fallback(weights):
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


def _fallback_kernel(data, state):
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
def _fp4_values(code):
    u = code & 7
    magnitude_bits = tl.where(u < 2, u * 16128, (u << 6) + 16128)
    bits = magnitude_bits | ((code & 8) << 12)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)


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
