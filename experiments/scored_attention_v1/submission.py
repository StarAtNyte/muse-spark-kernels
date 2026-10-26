# okc: problem=muse_attention_decode
"""Fuse QK normalization, RoPE, query conversion, and paged cache writes.

FA3 supplies the exact all-key attention path. No cache content or activation
is retained across calls. The only FP8 rounding is that allowed by the task.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


@triton.jit
def _prepare_qkv(X, POS, LOC, KS, VS, Q, KC, VC,
                 HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                 STRIDE: tl.constexpr, THETA: tl.constexpr,
                 SCALE: tl.constexpr, HEADS: tl.constexpr):
    token = tl.program_id(0)
    heads = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    dims = tl.arange(0, D)
    valid = heads < HQ + HKV
    values = tl.load(X + token * STRIDE + heads[:, None] * D + dims[None, :],
                     valid[:, None], 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, 1) / D + 1.e-5)
    normalized = values * inv_rms[:, None]
    if THETA != 0:
        partner = tl.load(X + token * STRIDE + heads[:, None] * D + (dims[None, :] ^ 1),
                          valid[:, None], 0).to(tl.float32) * inv_rms[:, None]
        position = tl.load(POS + token).to(tl.float32)
        frequency = tl.exp((-2.0 * (dims // 2).to(tl.float32) / D) * tl.log(THETA))
        angle = position * frequency
        cs, sn = libdevice.cos(angle), libdevice.sin(angle)
        normalized = normalized * cs[None, :] + tl.where(
            (dims[None, :] & 1) == 0, -partner, partner) * sn[None, :]
    # Query factor 5.473 is folded into FA3's softmax scale, matching SGLang.
    tl.store(Q + token * HQ * D + heads[:, None] * D + dims[None, :],
             normalized, (heads < HQ)[:, None])
    local_head = heads - HQ
    is_key = (heads >= HQ) & valid
    slot = tl.load(LOC + token)
    if SCALE:
        key_scale = tl.load(KS)
        value_scale = tl.load(VS)
    else:
        key_scale, value_scale = 1., 1.
    tl.store(KC + slot * HKV * D + local_head[:, None] * D + dims[None, :],
             normalized / key_scale, is_key[:, None])
    v = tl.load(X + token * STRIDE + (HQ + HKV + local_head[:, None]) * D + dims[None, :],
                is_key[:, None], 0).to(tl.float32)
    tl.store(VC + slot * HKV * D + local_head[:, None] * D + dims[None, :],
             v / value_scale, is_key[:, None])


def kernel_fn(data):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    qkv = data['qkv'].view(tokens, -1)
    q = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=qkv.device)
    scaled = cfg['kv_scale'] != 1.0
    _prepare_qkv[(tokens, triton.cdiv(hq + hkv, 4))](
        qkv, data['positions'], data['cache_loc'], data['k_scale'], data['v_scale'],
        q, data['k_cache'], data['v_cache'], hq, hkv, d, qkv.stride(0),
        cfg['rope_theta'], scaled, 4, num_warps=4, enable_fp_fusion=False)
    descale = {}
    if scaled:
        descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(batch, hkv),
                       v_descale=data['v_scale'].reshape(1, 1).expand(batch, hkv))
    window = cfg['window']
    out = flash_attn_with_kvcache(
        q=q, k_cache=data['k_cache'], v_cache=data['v_cache'],
        page_table=data['page_table'], cache_seqlens=data['cache_seqlens'],
        cu_seqlens_q=data['cu_seqlens_q'], max_seqlen_q=length,
        softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'],
        causal=True, window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data['scheduler_metadata'], **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2*hkv)*d:]
    result = rms_norm_gate(out.reshape(tokens, hq*d).to(torch.bfloat16), gate, None,
                           head_dim=d, eps=1.e-5, has_norm=True)
    return result.view(batch, length, hq*d)
