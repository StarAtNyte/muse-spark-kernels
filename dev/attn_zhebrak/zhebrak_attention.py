# okc: problem=muse_attention_decode
"""Gated attention decode (muse_attention_decode) over a paged FP8 KV cache: warp-specialized kernels with
cluster split-KV and a DSMEM merge, for one-token decode and 8-token verify blocks on sliding-window and full
layers.

kernel_fn is defined in layers; each definition takes the shapes it supports and passes the rest to the previous one:
Triton split-KV / FA3 base path -> FA3 head folding (full layers, batch >= 8) -> CUDA split-KV kernels (decode at
batch >= 8) -> the batch-1 cluster kernels (attn_b1: batch-1 decode and verify, batch-16 / 32 sliding decode, batched
verify) and the warp-specialized batched full-layer decode (wsd_forward: wsd2) -> wsv2 batched verify and batch-1
wsd2 / wsv2 routes (both off)."""
import os

# Base path: Triton split-KV attention with fused input preparation for batch-1 sliding-window layers and, with an
# int32 page-id lookahead, for batch-1 decode on full layers (page size 1, page-table capacity >= 8192); FA3 elsewhere.
# SPDX-License-Identifier: Apache-2.0
# Portions derived from SGLang: Copyright 2023-2024 SGLang Team.
# Source: sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py at commit
# a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0 .
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import triton
import triton.language as tl


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
ROPE_POSITIONS = 1 << 21
_ROPE = {}


def _cos_sin(theta, d, device):
    """Constant RoPE table, generated exactly as the supplied baseline does.

    Only model/configuration constants are cached; never data-derived tensors.
    The key includes dimension and device to prevent cross-configuration reuse.
    """
    key = (theta, d, device)
    cache = _ROPE.get(key)
    if cache is None:
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
        freqs = torch.einsum('i,j -> ij', torch.arange(ROPE_POSITIONS, dtype=torch.float32, device=device), inv)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        _ROPE[key] = cache
    return cache


@triton.jit
def _prepare_qkv(
    QKV, Q_OUT, K_CACHE, V_CACHE, POS, COS_SIN, CACHE_LOC, K_SCALE, V_SCALE,
    stride_qkv: tl.constexpr,
    N_ROWS: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    SCALED: tl.constexpr,
):
    # Derived from SGLang's _qk_rms_norm_fwd_kernel. A row is a token/head;
    # adjacent Q/K/V source heads share one grid and one launch.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BD)
    valid_rows = rows < N_ROWS
    mask = valid_rows[:, None] & (cols < D)[None, :]
    HEADS: tl.constexpr = HQ + 2 * HKV
    token = rows // HEADS
    head = rows % HEADS
    is_q = head < HQ
    is_k = (head >= HQ) & (head < HQ + HKV)
    is_v = head >= HQ + HKV
    src = token * stride_qkv + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    x = raw.to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D
    inv_rms = tl.rsqrt(variance + 1.0e-5)
    x = x * inv_rms[:, None]

    if HAS_ROPE:
        # Preserve the BF16 normalization boundary BEFORE RoPE. The original
        # FusedQKNorm explicitly rounds here to match the separate RMSNorm op.
        x = x.to(tl.bfloat16).to(tl.float32)
        partner = tl.load(QKV + src[:, None] + (cols ^ 1)[None, :], mask=mask, other=0.0).to(tl.float32)
        partner = (partner * inv_rms[:, None]).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(POS + token, mask=valid_rows, other=0).to(tl.int64)
        frequency = cols // 2
        cos = tl.load(COS_SIN + pos[:, None] * D + frequency[None, :], mask=mask, other=1.0)
        sin = tl.load(COS_SIN + pos[:, None] * D + D // 2 + frequency[None, :], mask=mask, other=0.0)
        x = tl.where((cols % 2 == 0)[None, :], x * cos - partner * sin,
                     partner * sin + x * cos)

    # The official baseline writes normalized/rotated Q and K as BF16, then
    # converts them to FP8. Values bypass normalization and RoPE entirely.
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(is_v[:, None], raw.to(tl.float32), x)
    if SCALED:
        k_scale = tl.load(K_SCALE)
        v_scale = tl.load(V_SCALE)
        scale = tl.where(is_k, k_scale, tl.where(is_v, v_scale, 1.0))
        # A BF16 tensor divided by a scalar FP32 scale has BF16 output in the
        # baseline. Preserve that rounding before its subsequent FP8 cast.
        x = (x / scale[:, None]).to(tl.bfloat16).to(tl.float32)

    # Pointer dtype supplies the final FP8 E4M3 conversion; there is no extra
    # quantization, clipping, query scaling or cache precision reduction.
    q_dst = tl.where(is_q, (token * HQ + head) * D, 0)
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])
    loc = tl.load(CACHE_LOC + token, mask=valid_rows, other=0).to(tl.int64)
    kv_head = tl.where(is_v, head - HQ - HKV, head - HQ)
    kv_dst = loc * HKV * D + kv_head * D
    # Unlike SGLang's padding convention, slot zero is a LIVE cache row here.
    tl.store(K_CACHE + kv_dst[:, None] + cols[None, :], x, mask=mask & is_k[:, None])
    tl.store(V_CACHE + kv_dst[:, None] + cols[None, :], x, mask=mask & is_v[:, None])


# Sliding-window split-KV attention. Only the preprocessing above is derived from SGLang.
SLIDING_SPLITS = 4


@triton.jit
def _sliding_partials(
    Q, K_CACHE, V_CACHE, POS, SEQLENS, PAGE_TABLE, K_SCALE,
    PART_M, PART_L, PART_ACC,
    HQ: tl.constexpr, HKV: tl.constexpr, LENGTH: tl.constexpr,
    GROUP: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr,
    TABLE_WIDTH: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    group_id = tl.program_id(0)
    split = tl.program_id(1)
    batch = group_id // HKV
    kv_head = group_id % HKV
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 64)
    row_live = rows < LENGTH * GROUP
    query_token = rows // GROUP
    query_head = kv_head * GROUP + rows % GROUP
    q_offset = ((batch * LENGTH + query_token) * HQ + query_head) * 64
    # E4M3 values are exactly representable in BF16: this conversion adds no
    # quantization before the BF16 tensor-core QK product and FP32 accumulation.
    query = tl.load(Q + q_offset[:, None] + dims[None, :],
                    mask=row_live[:, None], other=0.0).to(tl.bfloat16)
    query_pos = tl.load(POS + batch * LENGTH + query_token,
                        mask=row_live, other=0).to(tl.int64)
    first_pos = tl.load(POS + batch * LENGTH).to(tl.int64)
    first_key = tl.maximum(first_pos - WINDOW + 1, 0)
    sequence_end = tl.load(SEQLENS + batch).to(tl.int64)
    scale = SOFTMAX_SCALE * tl.load(K_SCALE) * 1.4426950408889634

    maximum = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, 64), tl.float32)
    key_columns = tl.arange(0, BLOCK_N)
    # Assign whole tiles evenly: every tile is loaded once, even when the
    # verify block adds L-1 keys beyond the nominal 2048-token window.
    total_tiles: tl.constexpr = (WINDOW + LENGTH + BLOCK_N - 2) // BLOCK_N
    tile_start = split * total_tiles // SPLITS
    tile_end = (split + 1) * total_tiles // SPLITS
    for tile in range(tile_start, tile_end):
        key_pos = first_key + tile * BLOCK_N + key_columns
        logical_page = key_pos // PAGE
        present = (key_pos < sequence_end) & (logical_page < TABLE_WIDTH)
        physical_page = tl.load(PAGE_TABLE + batch * TABLE_STRIDE + logical_page,
                                mask=present, other=-1).to(tl.int64)
        present = present & (physical_page >= 0)
        cache_row = physical_page * PAGE + key_pos % PAGE
        cache_offset = (cache_row * HKV + kv_head) * 64
        key = tl.load(K_CACHE + cache_offset[:, None] + dims[None, :],
                      mask=present[:, None], other=0.0).to(tl.bfloat16)
        value = tl.load(V_CACHE + cache_offset[:, None] + dims[None, :],
                        mask=present[:, None], other=0.0).to(tl.bfloat16)
        score = tl.dot(query, tl.trans(key)) * scale
        # All 2048 required keys, including the exact old edge and each
        # query's own token. Verify queries cannot see later block tokens.
        visible = (row_live[:, None] & present[None, :]
                   & (key_pos[None, :] <= query_pos[:, None])
                   & (key_pos[None, :] > query_pos[:, None] - WINDOW))
        score = tl.where(visible, score, -float('inf'))
        next_maximum = tl.maximum(maximum, tl.max(score, axis=1))
        # An empty split/iteration remains (m=-inf,l=0,acc=0) without NaNs.
        safe_maximum = tl.where(next_maximum == -float('inf'), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(score - safe_maximum[:, None])
        normalizer = normalizer * correction + tl.sum(probability, axis=1)
        accumulator = accumulator * correction[:, None]
        # BF16 probabilities with FP32 accumulation are the conventional
        # tensor-core attention path. No key/value is skipped by its weight.
        accumulator = tl.dot(probability.to(tl.bfloat16), value, accumulator)
        maximum = next_maximum

    partial_row = (group_id * SPLITS + split) * BLOCK_M + rows
    tl.store(PART_M + partial_row, maximum, mask=row_live)
    tl.store(PART_L + partial_row, normalizer, mask=row_live)
    tl.store(PART_ACC + partial_row[:, None] * 64 + dims[None, :],
             accumulator, mask=row_live[:, None])


@triton.jit
def _merge_sliding(
    PART_M, PART_L, PART_ACC, V_SCALE, QKV, OUT,
    HQ: tl.constexpr, HKV: tl.constexpr, LENGTH: tl.constexpr,
    GROUP: tl.constexpr, SPLITS: tl.constexpr, BLOCK_M: tl.constexpr,
    QKV_STRIDE: tl.constexpr, APPLY_GATE: tl.constexpr,
):
    # One real query head per program. Reduction order is fixed by split ID;
    # no atomics or inter-program scheduling dependence enters the sum.
    output_row = tl.program_id(0)
    token = output_row // HQ
    query_head = output_row % HQ
    batch = token // LENGTH
    query_token = token % LENGTH
    group_id = batch * HKV + query_head // GROUP
    row = query_token * GROUP + query_head % GROUP
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, 64)
    partial_row = (group_id * SPLITS + splits) * BLOCK_M + row
    maximum = tl.load(PART_M + partial_row)
    normalizer = tl.load(PART_L + partial_row)
    accumulator = tl.load(PART_ACC + partial_row[:, None] * 64 + dims[None, :])
    global_maximum = tl.max(maximum, axis=0)
    safe_maximum = tl.where(global_maximum == -float('inf'), 0.0, global_maximum)
    correction = tl.exp2(maximum - safe_maximum)
    denominator = tl.sum(normalizer * correction, axis=0)
    numerator = tl.sum(accumulator * correction[:, None], axis=0)
    attention = numerator / tl.maximum(denominator, 1.0e-30) * tl.load(V_SCALE)
    # Preserve the baseline's BF16 attention-output boundary before norm/gate.
    attention = attention.to(tl.bfloat16).to(tl.float32)
    if APPLY_GATE:
        gate_offset = token * QKV_STRIDE + (HQ + 2 * HKV + query_head) * 64
        gate = tl.load(QKV + gate_offset + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(attention * attention, axis=0) / 64 + 1.0e-5)
        attention = attention * inv_rms / (1.0 + tl.exp(-gate))
    tl.store(OUT + output_row * 64 + dims, attention)


def _supports_sliding(cfg):
    return (cfg['window'] == 2048 and cfg['head_dim'] == 64
            and cfg['num_kv_heads'] > 0
            and cfg['num_q_heads'] == 8 * cfg['num_kv_heads']
            and 1 <= cfg['q_len'] <= 8 and cfg['page_size'] in (1, 64))


def sliding_attention(data, q, *, apply_gate=False):
    """Consume prepared FP8 Q and updated caches; read live GPU metadata.

    By default returns raw BF16 attention `[B*L,Hq,64]` before norm/gate (no SGLang
    dependency). `apply_gate=True` fuses the baseline BF16 boundary, RMSNorm and
    sigmoid gate into the merge and returns `[B,L,Hq*64]`.
    """
    cfg = data['config']
    if not _supports_sliding(cfg):
        raise ValueError('sliding_attention requires D64/GQA8/window2048/L1..8/page1or64')
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv = cfg['num_q_heads'], cfg['num_kv_heads']
    if q.dtype != FP8 or tuple(q.shape) != (batch * length, hq, 64) or not q.is_contiguous():
        raise ValueError('sliding_attention requires contiguous FP8 Q [B*L,Hq,64]')
    # Fixed shape-derived work distribution; no GPU metadata is read on CPU.
    groups = batch * hkv
    block_n = 32 if groups < 8 else 64
    splits = max(4, min(64, triton.next_power_of_2(triton.cdiv(128, groups))))
    block_m = max(16, triton.next_power_of_2(length * 8))
    # Queries collectively require at most WINDOW + L - 1 positions. Round
    # the fixed per-split segment up to a tile; masks remove only out-of-window,
    # future, absent-page and padded coordinates. No CPU length read or plan.
    chunk = triton.cdiv(triton.cdiv(cfg['window'] + length - 1, splits), block_n) * block_n
    partial_shape = (batch * hkv, splits, block_m)
    maximum = torch.empty(partial_shape, dtype=torch.float32, device=q.device)
    normalizer = torch.empty_like(maximum)
    accumulator = torch.empty((*partial_shape, 64), dtype=torch.float32, device=q.device)
    output = torch.empty((batch * length, hq, 64), dtype=torch.bfloat16, device=q.device)
    _sliding_partials[(batch * hkv, splits)](
        q, data['k_cache'], data['v_cache'], data['positions'], data['cache_seqlens'],
        data['page_table'], data['k_scale'], maximum, normalizer, accumulator,
        HQ=hq, HKV=hkv, LENGTH=length, GROUP=8, PAGE=cfg['page_size'], WINDOW=cfg['window'],
        TABLE_WIDTH=data['page_table'].shape[1], TABLE_STRIDE=data['page_table'].stride(0),
        SPLITS=splits, CHUNK=chunk, BLOCK_M=block_m, BLOCK_N=block_n,
        SOFTMAX_SCALE=cfg['sm_scale'] * cfg['qk_scale_factor'],
        num_warps=4, num_stages=2,
    )
    qkv = data['qkv'].view(batch * length, -1)
    _merge_sliding[(batch * length * hq,)](
        maximum, normalizer, accumulator, data['v_scale'], qkv, output,
        HQ=hq, HKV=hkv, LENGTH=length, GROUP=8, SPLITS=splits, BLOCK_M=block_m,
        QKV_STRIDE=qkv.stride(0), APPLY_GATE=apply_gate, num_warps=4,
    )
    return output.view(batch, length, hq * 64) if apply_gate else output


def prepare_queries(data):
    """Fused input preparation: returns the FP8 queries and writes the new tokens' cache rows."""
    cfg = data['config']
    tokens = cfg['batch'] * cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    qkv = data['qkv'].view(tokens, -1)
    q = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
    rope = bool(cfg['rope_theta'])
    table = _cos_sin(cfg['rope_theta'], d, qkv.device) if rope else qkv
    n_rows = tokens * (hq + 2 * hkv)
    block_m = 4 if n_rows < 1024 else 8 if n_rows < 8192 else 16 if n_rows < 65536 else 32
    # The scaled FP8 preprocessor is exact with four rows on SM90/Triton3.8.
    # Larger tiles corrupt Q conversion in that specialization; no math changes.
    if cfg['kv_scale'] != 1.0:
        block_m = 4
    _prepare_qkv[(triton.cdiv(n_rows, block_m),)](
        qkv, q, data['k_cache'], data['v_cache'], data['positions'], table,
        data['cache_loc'], data['k_scale'], data['v_scale'],
        stride_qkv=qkv.stride(0), N_ROWS=n_rows, HQ=hq, HKV=hkv, D=d,
        BD=triton.next_power_of_2(d), BLOCK_M=block_m, HAS_ROPE=rope,
        SCALED=cfg['kv_scale'] != 1.0, num_warps=4,
    )
    return q



FULL_TARGET_CTAS = 512
FULL_MIN_KEYS_PER_SPLIT = 512
FULL_MAX_SPLITS = 128
FULL_DISPATCH_MIN_CAPACITY = 8192

@triton.jit
def _full_partials(
    Q, K_CACHE, V_CACHE, POS, SEQLENS, PAGE_TABLE, K_SCALE,
    PART_M, PART_L, PART_ACC,
    HQ: tl.constexpr, HKV: tl.constexpr, LENGTH: tl.constexpr,
    GROUP: tl.constexpr, PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    group_id = tl.program_id(0)
    split = tl.program_id(1)
    batch = group_id // HKV
    kv_head = group_id % HKV
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 64)
    row_live = rows < LENGTH * GROUP
    query_token = rows // GROUP
    query_head = kv_head * GROUP + rows % GROUP
    q_offset = ((batch * LENGTH + query_token) * HQ + query_head) * 64
    # Read existing E4M3 values directly; no new query/key rounding. The
    # supplied key descale is applied to the FP32 logits after the QK dot.
    query = tl.load(Q + q_offset[:, None] + dims[None, :],
                    mask=row_live[:, None], other=0.0)
    query_pos = tl.load(POS + batch * LENGTH + query_token,
                        mask=row_live, other=0).to(tl.int64)
    sequence_end = tl.load(SEQLENS + batch).to(tl.int64)
    scale = SOFTMAX_SCALE * tl.load(K_SCALE) * 1.4426950408889634

    maximum = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, 64), tl.float32)
    key_columns = tl.arange(0, BLOCK_N)
    # The grid and partitions depend only on tensor shapes. The GPU reads live
    # lengths to avoid work beyond the actual context; every live key in this
    # split is visited, irrespective of its attention probability.
    split_start = split * CHUNK
    active_span = tl.maximum(0, tl.minimum(CHUNK, sequence_end - split_start))
    # Prime exactly one page-ID tile. Empty splits perform only masked loads.
    # These are live GPU values; no mapping survives a call or graph replay.
    first_key_pos = split_start + key_columns
    first_logical_page = first_key_pos // PAGE
    first_present = ((first_key_pos < split_start + active_span)
                     & (first_logical_page < TABLE_WIDTH))
    physical_page = tl.load(
        PAGE_TABLE + batch * TABLE_STRIDE + first_logical_page,
        mask=first_present, other=-1,
    )
    for block in range(0, active_span, BLOCK_N):
        key_pos = split_start + block + key_columns
        logical_page = key_pos // PAGE
        present = (key_pos < sequence_end) & (logical_page < TABLE_WIDTH)
        present = present & (physical_page >= 0)
        # Resolve the next tile before current QK/softmax/PV work. Restrict
        # this lookahead to the current split: the final iteration fetches
        # nothing, including when a short live length ends inside a tile.
        next_key_pos = key_pos + BLOCK_N
        next_logical_page = next_key_pos // PAGE
        next_present = ((next_key_pos < split_start + active_span)
                        & (next_logical_page < TABLE_WIDTH))
        next_physical_page = tl.load(
            PAGE_TABLE + batch * TABLE_STRIDE + next_logical_page,
            mask=next_present, other=-1,
        )
        # Page IDs stay int32 across iterations; widen before address arithmetic.
        cache_row = physical_page.to(tl.int64) * PAGE + key_pos % PAGE
        cache_offset = (cache_row * HKV + kv_head) * 64
        key = tl.load(K_CACHE + cache_offset[:, None] + dims[None, :],
                      mask=present[:, None], other=0.0)
        value = tl.load(V_CACHE + cache_offset[:, None] + dims[None, :],
                        mask=present[:, None], other=0.0).to(tl.bfloat16)
        # FP8 inputs are the existing E4M3 values. Hopper's native FP8 WGMMA
        # has a minimum 32-product internal accumulation group. Limit it to
        # that minimum and combine groups in FP32, rather than using Triton's
        # unrestricted default. Softmax/normalizer/PV accumulation remain FP32.
        # WGMMA requires M divisible by 64. Transpose the short-query product
        # so its M dimension is the 64-key tile, with no additional live rows.
        if BLOCK_M < 64:
            score = tl.trans(tl.dot(key, tl.trans(query), out_dtype=tl.float32,
                                    max_num_imprecise_acc=32)) * scale
        else:
            score = tl.dot(query, tl.trans(key), out_dtype=tl.float32,
                           max_num_imprecise_acc=32) * scale
        # Full context from key zero, including every low-attention value.
        # Verify queries cannot see later tokens in their own block.
        visible = (row_live[:, None] & present[None, :]
                   & (key_pos[None, :] <= query_pos[:, None]))
        score = tl.where(visible, score, -float('inf'))
        next_maximum = tl.maximum(maximum, tl.max(score, axis=1))
        # An empty split/iteration remains (m=-inf,l=0,acc=0) without NaNs.
        safe_maximum = tl.where(next_maximum == -float('inf'), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(score - safe_maximum[:, None])
        normalizer = normalizer * correction + tl.sum(probability, axis=1)
        accumulator = accumulator * correction[:, None]
        # BF16 probabilities with FP32 accumulation are the conventional
        # tensor-core attention path. No key/value is skipped by its weight.
        accumulator = tl.dot(probability.to(tl.bfloat16), value, accumulator)
        maximum = next_maximum
        physical_page = next_physical_page

    partial_row = (group_id * SPLITS + split) * BLOCK_M + rows
    tl.store(PART_M + partial_row, maximum, mask=row_live)
    tl.store(PART_L + partial_row, normalizer, mask=row_live)
    tl.store(PART_ACC + partial_row[:, None] * 64 + dims[None, :],
             accumulator, mask=row_live[:, None])


@triton.jit
def _merge_attention(
    PART_M, PART_L, PART_ACC, V_SCALE, QKV, OUT,
    HQ: tl.constexpr, HKV: tl.constexpr, LENGTH: tl.constexpr,
    GROUP: tl.constexpr, SPLITS: tl.constexpr, BLOCK_M: tl.constexpr,
    QKV_STRIDE: tl.constexpr, APPLY_GATE: tl.constexpr,
):
    # One real query head per program. Reduction order is fixed by split ID;
    # no atomics or inter-program scheduling dependence enters the sum.
    output_row = tl.program_id(0)
    token = output_row // HQ
    query_head = output_row % HQ
    batch = token // LENGTH
    query_token = token % LENGTH
    group_id = batch * HKV + query_head // GROUP
    row = query_token * GROUP + query_head % GROUP
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, 64)
    partial_row = (group_id * SPLITS + splits) * BLOCK_M + row
    maximum = tl.load(PART_M + partial_row)
    normalizer = tl.load(PART_L + partial_row)
    accumulator = tl.load(PART_ACC + partial_row[:, None] * 64 + dims[None, :])
    global_maximum = tl.max(maximum, axis=0)
    safe_maximum = tl.where(global_maximum == -float('inf'), 0.0, global_maximum)
    correction = tl.exp2(maximum - safe_maximum)
    denominator = tl.sum(normalizer * correction, axis=0)
    numerator = tl.sum(accumulator * correction[:, None], axis=0)
    attention = numerator / tl.maximum(denominator, 1.0e-30) * tl.load(V_SCALE)
    # Preserve the baseline's BF16 attention-output boundary before norm/gate.
    attention = attention.to(tl.bfloat16).to(tl.float32)
    if APPLY_GATE:
        gate_offset = token * QKV_STRIDE + (HQ + 2 * HKV + query_head) * 64
        gate = tl.load(QKV + gate_offset + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(attention * attention, axis=0) / 64 + 1.0e-5)
        attention = attention * inv_rms / (1.0 + tl.exp(-gate))
    tl.store(OUT + output_row * 64 + dims, attention)


def _supports_full(cfg):
    # Both page sizes are supported here; kernel_fn's dispatch below uses this path
    # only for long page-size-1 layouts.
    return (cfg['window'] == 0 and cfg['head_dim'] == 64
            and cfg['num_kv_heads'] > 0
            and cfg['num_q_heads'] == 8 * cfg['num_kv_heads']
            and 1 <= cfg['q_len'] <= 8 and cfg['page_size'] in (1, 64))


def full_schedule(batch, kv_heads, length, capacity):
    """Pure shape arithmetic; no tensor contents, GPU property query or plan.

    Return fixed launch/temporary dimensions for a graph. The constants are
    fixed heuristics, not tuned at runtime.
    """
    if batch <= 0 or kv_heads <= 0 or not 1 <= length <= 8 or capacity <= 0:
        raise ValueError('invalid full-attention schedule dimensions')
    groups = batch * kv_heads
    desired = max(1, (FULL_TARGET_CTAS + groups - 1) // groups)
    desired = 1 << (desired - 1).bit_length()
    useful = max(1, capacity // FULL_MIN_KEYS_PER_SPLIT)
    useful = 1 << (useful.bit_length() - 1)
    splits = min(FULL_MAX_SPLITS, desired, useful)
    if batch == 1 and kv_heads == 2 and length == 1:
        useful_decode = max(1, capacity // 256)
        useful_decode = 1 << (useful_decode.bit_length() - 1)
        splits = min(256, useful_decode)
    block_n = 64
    chunk = (((capacity + splits - 1) // splits + block_n - 1) // block_n) * block_n
    block_m = max(16, 1 << (length * 8 - 1).bit_length())
    return {
        'splits': splits, 'chunk': chunk, 'block_m': block_m, 'block_n': block_n,
        'groups': groups, 'ctas': groups * splits, 'capacity': capacity,
        'scratch_bytes': groups * splits * block_m * (64 + 2) * 4,
    }


def full_attention(data, q, *, apply_gate=False):
    """Raw BF16 attention [B*L,Hq,64], or fused final output if apply_gate.

    Q must already be FP8 and new cache rows written by prepare_queries. All
    lengths/positions/page mappings are reread on the GPU on every call/replay.
    """
    cfg = data['config']
    if not _supports_full(cfg):
        raise ValueError('full_attention requires full/D64/GQA8/L1..8/page1or64')
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv = cfg['num_q_heads'], cfg['num_kv_heads']
    if q.dtype != FP8 or tuple(q.shape) != (batch * length, hq, 64) or not q.is_contiguous():
        raise ValueError('full_attention requires contiguous FP8 Q [B*L,Hq,64]')
    schedule = full_schedule(batch, hkv, length, data['page_table'].shape[1] * cfg['page_size'])
    splits, chunk = schedule['splits'], schedule['chunk']
    block_m, block_n = schedule['block_m'], schedule['block_n']
    shape = (batch * hkv, splits, block_m)
    maximum = torch.empty(shape, dtype=torch.float32, device=q.device)
    normalizer = torch.empty_like(maximum)
    accumulator = torch.empty((*shape, 64), dtype=torch.float32, device=q.device)
    output = torch.empty((batch * length, hq, 64), dtype=torch.bfloat16, device=q.device)
    _full_partials[(batch * hkv, splits)](
        q, data['k_cache'], data['v_cache'], data['positions'], data['cache_seqlens'],
        data['page_table'], data['k_scale'], maximum, normalizer, accumulator,
        HQ=hq, HKV=hkv, LENGTH=length, GROUP=8, PAGE=cfg['page_size'],
        TABLE_WIDTH=data['page_table'].shape[1], TABLE_STRIDE=data['page_table'].stride(0),
        SPLITS=splits, CHUNK=chunk, BLOCK_M=block_m, BLOCK_N=block_n,
        SOFTMAX_SCALE=cfg['sm_scale'] * cfg['qk_scale_factor'],
        num_warps=4, num_stages=2,
    )
    qkv = data['qkv'].view(batch * length, -1)
    _merge_attention[(batch * length * hq,)](
        maximum, normalizer, accumulator, data['v_scale'], qkv, output,
        HQ=hq, HKV=hkv, LENGTH=length, GROUP=8, SPLITS=splits, BLOCK_M=block_m,
        QKV_STRIDE=qkv.stride(0), APPLY_GATE=apply_gate, num_warps=4,
    )
    return output.view(batch, length, hq * 64) if apply_gate else output


def _use_full(data):
    cfg = data['config']
    return (_supports_full(cfg) and cfg['batch'] == 1 and cfg['q_len'] == 1
            and cfg['num_q_heads'] == 16 and cfg['num_kv_heads'] == 2
            and cfg['page_size'] == 1
            and data['page_table'].shape[1] >= FULL_DISPATCH_MIN_CAPACITY)


def kernel_fn(data):
    cfg = data['config']
    q = prepare_queries(data)
    if _supports_sliding(cfg) and cfg['batch'] == 1 and cfg['num_kv_heads'] == 2:
        return sliding_attention(data, q, apply_gate=True)

    if _use_full(data):
        return full_attention(data, q, apply_gate=True)

    # Other configurations use FA3 with its supplied GPU schedule and the
    # reference output norm/gate implementation.
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    descale = {}
    if cfg['kv_scale'] != 1.0:
        descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(batch, hkv),
                       v_descale=data['v_scale'].reshape(1, 1).expand(batch, hkv))
    window = cfg['window']
    out = flash_attn_with_kvcache(
        q=q, k_cache=data['k_cache'], v_cache=data['v_cache'],
        page_table=data['page_table'], cache_seqlens=data['cache_seqlens'],
        cu_seqlens_q=data['cu_seqlens_q'], max_seqlen_q=length,
        softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data['scheduler_metadata'], **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    qkv = data['qkv'].view(tokens, -1)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    output = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate,
                           None, head_dim=d, eps=RMS_EPS, has_norm=True)
    return output.view(batch, length, hq * d)



# Exact head folding around the public FA3 API for full-attention layers at batch >= 8
# (q_len <= 8, page size 1): the two 64-wide kv heads are attended as one 128-wide head.
_HEADFOLD_BASE_KERNEL = kernel_fn


@triton.jit
def _headfold_query(Q, OUT, ROWS: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dimensions = tl.arange(0, 128)
    # Query heads 0..7 use kv head 0, heads 8..15 kv head 1. Each query keeps its 64
    # FP8 values in its kv head's half and exact zeros in the other half.
    half = (rows % 16) // 8
    query_bytes = Q.to(tl.pointer_type(tl.uint8))
    output_bytes = OUT.to(tl.pointer_type(tl.uint8))
    values = tl.load(query_bytes + rows[:, None] * 64 + (dimensions[None, :] % 64),
                     (rows[:, None] < ROWS) & (dimensions[None, :] // 64 == half[:, None]),
                     other=0)
    tl.store(output_bytes + rows[:, None] * 128 + dimensions[None, :], values,
             rows[:, None] < ROWS)


@triton.jit
def _headfold_norm_gate(ATTENTION, QKV, OUT,
                        ATT_TOKEN_STRIDE: tl.constexpr, ATT_HEAD_STRIDE: tl.constexpr,
                        ATT_DIM_STRIDE: tl.constexpr, QKV_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    token, head = row // 16, row % 16
    dimensions = tl.arange(0, 64)
    selected = (head // 8) * 64 + dimensions
    value = tl.load(ATTENTION + token * ATT_TOKEN_STRIDE + head * ATT_HEAD_STRIDE
                    + selected * ATT_DIM_STRIDE).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(QKV + token * QKV_STRIDE + (20 + head) * 64 + dimensions).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / 64.0 + 1.0e-5)
    result = value * inverse_rms / (1.0 + tl.exp(-gate))
    tl.store(OUT + row * 64 + dimensions, result)


def _headfold_supported(data):
    cfg = data["config"]
    if not (cfg["window"] == 0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["batch"] >= 8 and 1 <= cfg["q_len"] <= 8
            and cfg["page_size"] == 1 and data["page_table"].shape[1] >= 8192):
        return False
    k, v = data["k_cache"], data["v_cache"]
    return (k.ndim == 4 and v.shape == k.shape and tuple(k.shape[1:]) == (1, 2, 64)
            and k.is_contiguous() and v.is_contiguous() and k.dtype == FP8 and v.dtype == FP8
            and k.data_ptr() % 16 == 0 and v.data_ptr() % 16 == 0
            and k.device == v.device == data["qkv"].device
            and data["k_scale"].numel() == 1 and data["v_scale"].numel() == 1)


def headfold_inputs(data, q):
    """Query embedding and zero-copy cache views for head folding.

    Each 128-wide cache token concatenates the two 64-wide KV heads.
    The live page table, cache lengths, positions and append writes are unchanged.
    The helper also accepts page size 64; _headfold_supported dispatches only
    page size 1.
    """
    cfg = data["config"]
    tokens = cfg["batch"] * cfg["q_len"]
    if (cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2 or cfg["head_dim"] != 64
            or tuple(q.shape) != (tokens, 16, 64) or q.dtype != FP8 or not q.is_contiguous()):
        raise ValueError("head folding requires contiguous FP8 TP8 D64 queries")
    k, v = data["k_cache"], data["v_cache"]
    if (not k.is_contiguous() or not v.is_contiguous() or k.shape != v.shape
            or k.dtype != FP8 or v.dtype != FP8 or k.device != q.device or v.device != q.device
            or tuple(k.shape[-2:]) != (2, 64) or data["k_scale"].numel() != 1
            or data["v_scale"].numel() != 1):
        raise ValueError("head folding requires contiguous matching caches and global scalar scales")
    query = torch.empty((tokens, 16, 128), dtype=FP8, device=q.device)
    _headfold_query[(triton.cdiv(tokens * 16, 4),)](q, query, tokens * 16, 4, num_warps=4)
    return query, k.view(k.shape[0], k.shape[1], 1, 128), v.view(v.shape[0], v.shape[1], 1, 128)


def headfold_output(data, attention):
    """Select each head's 64 output channels, then normalize and gate."""
    cfg = data["config"]
    tokens = cfg["batch"] * cfg["q_len"]
    attention = attention.reshape(tokens, 16, 128)
    qkv = data["qkv"].reshape(tokens, -1)
    output = torch.empty((cfg["batch"], cfg["q_len"], 1024),
                         dtype=torch.bfloat16, device=attention.device)
    _headfold_norm_gate[(tokens * 16,)](
        attention, qkv, output, attention.stride(0), attention.stride(1),
        attention.stride(2), qkv.stride(0), num_warps=4)
    return output


def kernel_fn(data):
    if not _headfold_supported(data):
        return _HEADFOLD_BASE_KERNEL(data)
    cfg = data["config"]
    # Norm and the FP8 rounding happen at the original D64 boundary, before
    # zero embedding. prepare_queries also performs every live cache update.
    q = prepare_queries(data)
    query, keys, values = headfold_inputs(data, q)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(cfg["batch"], 1),
                       v_descale=data["v_scale"].reshape(1, 1).expand(cfg["batch"], 1))
    attention = flash_attn_with_kvcache(
        q=query, k_cache=keys, v_cache=values, page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=cfg["q_len"],
        # Keep the D64 softmax scale; 1/sqrt(128) would be wrong.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True, window_size=(-1, -1),
        # The supplied schedule describes D64/Hkv2. FA3 must derive the real
        # D128/Hkv1 schedule from the live GPU lengths on every replay instead.
        scheduler_metadata=None, **descale,
    )
    if isinstance(attention, tuple):
        attention = attention[0]
    return headfold_output(data, attention)


# CUDA split-KV decode (mma.sync) and verify (wgmma) kernels for the shapes _team_v3_supported
# accepts; other shapes fall back to the kernel_fn above.
_TEAM_V3_BASE_KERNEL = kernel_fn

import torch as _TEAMV3_torch

# Decode / verify attention over a paged FP8 E4M3 KV cache as one bandwidth-bound split-KV kernel plus a merge.
# Structure: grid = (splits, batch). A CTA owns one request, BOTH kv heads and one contiguous range of cached keys, so
# every token's K (and V) row is one 128-byte transaction: the H200 streams random 128-B rows at twice the rate of random
# 64-B rows. All threads stream 16-key tiles of K and V rows (gathered through the page table) into a shared-memory ring
# with 16-byte cp.async copies, four tiles per commit group and three groups in flight; the page ids of the CTA's range
# are staged in shared memory first. Each warp serves one kv head: for decode the four warps of a head take one tile each
# per group (one 16-row m-tile: the 8 GQA heads + 8 zero rows) and merge in shared memory at the end; for verify each warp
# owns one m-tile (2 of the head's 8 queries) and consumes every tile. The recurrence is FA2 on mma.sync.m16n8k16 (f16 x
# f16 -> f32): S = Q K^T with a per-lane permutation of the head dimension so every lane's K bytes are one 16-byte shared
# load, e4m3 -> f16 by cvt (exact), online softmax in base 2, P reused from the S accumulators as the A operand, O += P V
# with a permuted output dimension so each lane's V bytes are 8-byte loads. CTAs write (m, l, O) partials; the merge
# kernel (one CTA per query head, 8 warps over the splits) combines them, scales by 1/l and v_scale, applies the output
# RMS norm and the sigmoid gate, and writes bf16. The block's new tokens are normalized (RoPE on sliding layers),
# quantized to e4m3 with the cache scales, written to their cache rows by the last split's CTA and attended from those
# same bytes, so the step sees exactly what it stored.
import torch as _TEAMV3_torch

_TEAMV3_ATTN_DECODE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace okcattn {

constexpr int NW = 8;             // warps per CTA (4 per kv head)
constexpr int NWH = NW / 2;       // warps per kv head
constexpr int HKV = 2;            // kv heads per CTA (the model's TP8 layout)
constexpr int G = 4;              // tiles per cp.async group
constexpr int NG = __OKC_ATTN_NG__;  // groups in flight
constexpr int PR_ST = 4;          // decode: stages per warp pair (the two kv-head warps of one tile index share a ring)
constexpr int HALF_FILL = __OKC_ATTN_HALF__;  // 1: each warp copies only its own head's 64-B halves and waits on itself (no pair barrier)
// ring slots for cached tiles: decode pairs need 4 x PR_ST, verify groups NG x G; + 1 slot for the new-token tile
template <int L> struct Ring {
  static constexpr int RS = (L == 1 && 4 * PR_ST > NG * G) ? 4 * PR_ST : NG * G;
  static constexpr int RD = RS + 1;
};
constexpr int MAX_CHUNK = __OKC_ATTN_MAXCHUNK__;   // keys per CTA at most (page ids staged in shared memory)
constexpr int KROW = 64;          // K tile row stride (bytes)
constexpr int VROW = 80;          // V tile row stride (bytes): conflict-free 8-byte fragment loads
constexpr int KHEAD = 16 * KROW;  // 1024: one head's K tile
constexpr int VHEAD = 16 * VROW;  // 1280: one head's V tile
constexpr int SLOT = HKV * (KHEAD + VHEAD);  // 4608 B
constexpr int MW = 8;             // merge warps (split lanes)
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
// PDL (no-ops without the programmatic-serialization launch attribute)
__device__ __forceinline__ void gdc_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

struct Params {
  const __nv_bfloat16* qkv;   // [B, L, (2 Hq + 2 Hkv) * 64]
  const int64_t* cache_loc;   // [B * L]
  uint8_t* k_cache;           // [pages, ps, Hkv, 64] e4m3
  uint8_t* v_cache;
  const int32_t* page_table;  // [B, max_pages]
  const int32_t* seq_lens;    // [B]
  const float* inv_freq;      // [32] RoPE inverse frequencies (as the reference computes them)
  float* part_o;              // [B, S, HKV * 8L, 64]
  float* part_ml;             // [B, S, HKV * 8L, 2]
  __nv_bfloat16* out;         // [B, L, Hq * 64]
  int B, L, Hq, page_size, max_pages, window, splits, chunk;
  float qk_scale, q_mult, inv_k_scale, inv_v_scale;  // q_mult = sm_scale * k_scale (exact powers of two)
  int rope;                   // 1 on sliding layers
  int mode;                   // 0 normal; 1 = stream only (a bandwidth probe)
};

template <int L> struct Smem {
  static constexpr int QROWS = HKV * 16 * ((L + 1) / 2);  // decode: 2 x 16 (8 real + 8 zero); verify: 2 x 64
  static constexpr int BYTES = MAX_CHUNK * 4 + Ring<L>::RD * SLOT + QROWS * 64 * 2;
};

// One CTA: request b, split s, both kv heads. Dynamic smem: page ids, ring [RD][SLOT], Q [QROWS][64] f16.
template <int L>
__global__ void __launch_bounds__(NW * 32) attn_decode_kernel(const Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  int32_t* ids = reinterpret_cast<int32_t*>(smem);
  unsigned char* ring = smem + MAX_CHUNK * 4;
  constexpr int RS = Ring<L>::RS, RD = Ring<L>::RD;
  uint16_t* qs = reinterpret_cast<uint16_t*>(ring + RD * SLOT);  // [HKV][16 or 64 rows][64] f16
  float* red = reinterpret_cast<float*>(ring);                    // decode merge scratch (after the loop)
  constexpr int QH = 16 * ((L + 1) / 2);  // Q rows per head in smem (16 or 64)

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, b = blockIdx.y;
  const int hh = warp & 1;             // this warp's kv head
  const int wq = warp >> 1;            // decode: the tile of each group this warp consumes; verify: its m-tile
  const int group = p.Hq / HKV;        // 8 query heads per kv head
  const int seq = p.seq_lens[b];
  const int first = p.window ? max(0, seq - p.window + 1) : 0;  // oldest key the block's first query sees
  const int k0 = first + s * p.chunk;
  const int k1 = min(seq, k0 + p.chunk);
  const bool tail = (s == p.splits - 1);  // this CTA also attends (and stores) the block's new tokens
  const int ncached_here = max(0, k1 - k0);
  const int ntiles_c = (ncached_here + 15) / 16;
  constexpr int ROW_BYTES = HKV * 64;  // one flat cache row (both kv heads): 128 B
  const int qkv_stride = (2 * p.Hq + 2 * HKV) * 64;

  // --- page ids of this CTA's cached keys -------------------------------------------------------------------------
  {  // all page-table loads of this thread in flight at once (a dependent loop would cost one latency per 256 keys)
    constexpr int PT_PER_THREAD = MAX_CHUNK / (NW * 32);
    int32_t pt[PT_PER_THREAD];
#pragma unroll
    for (int r = 0; r < PT_PER_THREAD; ++r) {
      const int i = tid + r * NW * 32;
      const int pos = k0 + min(i, max(ncached_here - 1, 0));
      pt[r] = p.page_table[static_cast<size_t>(b) * p.max_pages + pos / p.page_size];
    }
#pragma unroll
    for (int r = 0; r < PT_PER_THREAD; ++r) {
      const int i = tid + r * NW * 32;
      if (i < ncached_here) ids[i] = pt[r] * p.page_size + (k0 + i) % p.page_size;
    }
  }
  // Everything above reads only the step's inputs; the queries and new keys below come from the previous kernel.
  gdc_launch();
  gdc_wait();
  // --- queries: rms * qk_scale (RoPE on sliding) -> bf16 -> * sm_scale * k_scale -> f16 ---------------------------
  // rows per head r = 8 j + gq (query j, group head gq); TPR consecutive threads per row, DPT dims each.
  {
    constexpr int ROWS = HKV * 8 * L;                       // 16 or 128 real rows
    constexpr int TPR = (NW * 32 / ROWS) >= 2 ? (NW * 32 / ROWS) : 2;  // decode: 16 threads x 4 dims; verify: 2 x 32
    constexpr int DPT = 64 / TPR;
    for (int base = 0; base < ROWS * TPR; base += NW * 32) {
      const int it = base + tid;
      const int rr = it / TPR, part = it % TPR;              // rr: head-major row index
      const int head_kv = rr / (8 * L), r = rr % (8 * L), j = r >> 3, gq = r & 7;
      const __nv_bfloat16* q = p.qkv + (static_cast<size_t>(b) * L + j) * qkv_stride + (head_kv * group + gq) * 64 + part * DPT;
      float v[DPT];
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < DPT; i += 4) {
        const uint2 u = *reinterpret_cast<const uint2*>(q + i);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const float2 f = __bfloat1622float2(b2[k]);
          v[i + 2 * k] = f.x; v[i + 2 * k + 1] = f.y;
          ss += f.x * f.x + f.y * f.y;
        }
      }
#pragma unroll
      for (int o2 = 1; o2 < TPR; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);  // TPR lanes share a row
      const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f) * p.qk_scale;
      if (p.rope) {
        const float pos = static_cast<float>(seq + j);
#pragma unroll
        for (int i = 0; i < DPT; i += 2) {
          float sn, cs;
          sincosf(pos * p.inv_freq[(part * DPT + i) >> 1], &sn, &cs);
          const float e = v[i] * inv, o = v[i + 1] * inv;
          v[i] = e * cs - o * sn;
          v[i + 1] = e * sn + o * cs;
        }
      } else {
#pragma unroll
        for (int i = 0; i < DPT; ++i) v[i] *= inv;
      }
      uint16_t* dst = qs + (head_kv * QH + r) * 64 + part * DPT;
#pragma unroll
      for (int i = 0; i < DPT; i += 2) {
        *reinterpret_cast<uint32_t*>(dst + i) = pack_f16x2(bf16_round(v[i]) * p.q_mult, bf16_round(v[i + 1]) * p.q_mult);
      }
    }
    if (L == 1) {  // rows 8..15 of each head's m-tile are padding
      for (int i = tid; i < 2 * 8 * 64 / 2; i += NW * 32) {
        const int head_kv = i / (8 * 32), w = i % (8 * 32);
        reinterpret_cast<uint32_t*>(qs + (head_kv * QH + 8) * 64)[w] = 0u;
      }
    }
  }
  // --- the block's new tokens: normalized keys (RoPE on sliding), quantized with the cache scales; stored to the
  //     cache rows and to the ring's last slot as a 16-key tile (keys >= L zero), both heads ----------------------
  unsigned char* newslot = ring + RS * SLOT;
  if (tail) {
    for (int i = tid; i < 16 * HKV * 8; i += NW * 32) {  // (key, head, 8-byte chunk)
      const int key = i >> 4, head_kv = (i >> 3) & 1, c = i & 7;
      uint2 kb = make_uint2(0u, 0u), vb = make_uint2(0u, 0u);
      if (key < L) {
        const __nv_bfloat16* row = p.qkv + (static_cast<size_t>(b) * L + key) * qkv_stride;
        const __nv_bfloat16* kin = row + p.Hq * 64 + head_kv * 64;
        const __nv_bfloat16* vin = row + p.Hq * 64 + HKV * 64 + head_kv * 64;
        float ss = 0.f;
#pragma unroll
        for (int d = 0; d < 64; d += 8) {
          const uint4 u = *reinterpret_cast<const uint4*>(kin + d);
          const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
          for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b2[k]); ss += f.x * f.x + f.y * f.y; }
        }
        const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
        float kv[8], vv[8];
        {
          const uint4 u = *reinterpret_cast<const uint4*>(kin + 8 * c);
          const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
          for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b2[k]); kv[2 * k] = f.x * inv; kv[2 * k + 1] = f.y * inv; }
          const uint4 w = *reinterpret_cast<const uint4*>(vin + 8 * c);
          const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
          for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(c2[k]); vv[2 * k] = f.x; vv[2 * k + 1] = f.y; }
        }
        if (p.rope) {
          const float pos = static_cast<float>(seq + key);
#pragma unroll
          for (int k = 0; k < 8; k += 2) {
            float sn, cs;
            sincosf(pos * p.inv_freq[(8 * c + k) >> 1], &sn, &cs);
            const float e = kv[k], o = kv[k + 1];
            kv[k] = e * cs - o * sn;
            kv[k + 1] = e * sn + o * cs;
          }
        }
        uint8_t kq[8], vq[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          kq[k] = __nv_cvt_float_to_fp8(kv[k] * p.inv_k_scale, __NV_SATFINITE, __NV_E4M3);
          vq[k] = __nv_cvt_float_to_fp8(vv[k] * p.inv_v_scale, __NV_SATFINITE, __NV_E4M3);
        }
        kb = *reinterpret_cast<uint2*>(kq);
        vb = *reinterpret_cast<uint2*>(vq);
        const int64_t loc = p.cache_loc[b * L + key];
        *reinterpret_cast<uint2*>(p.k_cache + static_cast<size_t>(loc) * ROW_BYTES + head_kv * 64 + 8 * c) = kb;
        *reinterpret_cast<uint2*>(p.v_cache + static_cast<size_t>(loc) * ROW_BYTES + head_kv * 64 + 8 * c) = vb;
      }
      *reinterpret_cast<uint2*>(newslot + head_kv * KHEAD + key * KROW + 8 * c) = kb;
      *reinterpret_cast<uint2*>(newslot + HKV * KHEAD + head_kv * VHEAD + key * VROW + 8 * c) = vb;
    }
  }
  __syncthreads();

  // --- this warp's m-tile: decode: rows = the head's 8 query heads + padding; verify: queries 2 wq, 2 wq + 1
  constexpr int MT = 1;
  const int mt0 = (L == 1) ? 0 : wq;
  uint32_t qa[MT][8], qb[MT][8];  // rows g / g+8: dims 16 t .. 16 t + 15 as 8 f16x2
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const uint4* r0 = reinterpret_cast<const uint4*>(qs + (hh * QH + 16 * (mt0 + m) + g) * 64 + 16 * t);
    const uint4* r1 = reinterpret_cast<const uint4*>(qs + (hh * QH + 16 * (mt0 + m) + g + 8) * 64 + 16 * t);
    const uint4 a = r0[0], bq = r0[1], c = r1[0], d = r1[1];
    qa[m][0] = a.x; qa[m][1] = a.y; qa[m][2] = a.z; qa[m][3] = a.w; qa[m][4] = bq.x; qa[m][5] = bq.y; qa[m][6] = bq.z; qa[m][7] = bq.w;
    qb[m][0] = c.x; qb[m][1] = c.y; qb[m][2] = c.z; qb[m][3] = c.w; qb[m][4] = d.x; qb[m][5] = d.y; qb[m][6] = d.z; qb[m][7] = d.w;
  }
  // query index of rows g / g+8 per m-tile, and the key bounds per row: valid iff lo <= key <= hi
  int lo_lo[MT], lo_hi[MT], hi_lo[MT], hi_hi[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int j_lo = (L == 1) ? 0 : 2 * (mt0 + m), j_hi = (L == 1) ? 0 : 2 * (mt0 + m) + 1;
    lo_lo[m] = p.window ? seq + j_lo - p.window + 1 : 0; lo_hi[m] = p.window ? seq + j_hi - p.window + 1 : 0;
    hi_lo[m] = seq + j_lo; hi_hi[m] = seq + j_hi;
  }
  // a tile needs masking when some row's bounds cut into it: the latest lower bound / the earliest upper bound
  const int lo_max = p.window ? lo_hi[MT - 1] : 0;
  const int hi_min = hi_lo[0];

  float o[MT][8][4];
  float m_lo[MT], m_hi[MT], l_lo[MT], l_hi[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[m][j][0] = o[m][j][1] = o[m][j][2] = o[m][j][3] = 0.f; }
    m_lo[m] = -INFINITY; m_hi[m] = -INFINITY; l_lo[m] = 0.f; l_hi[m] = 0.f;
  }

  // --- ring fill: tile i -> slot i % RS; threads 0..127 copy 16 B of K each (key tid / 8, 128-B row piece tid % 8),
  //     threads 128..255 the V rows
  auto fill = [&](int i) {
    if (i >= ntiles_c) return;
    unsigned char* slot = ring + (i % RS) * SLOT;
    const int tt = tid & 127, key = tt >> 3, piece = tt & 7, head_kv = piece >> 2, sub = piece & 3;
    const int idx = i * 16 + key;
    const bool valid = idx < ncached_here;
    const size_t row = valid ? static_cast<size_t>(ids[idx]) : 0;
    if (tid < 128) cp_async16(slot + head_kv * KHEAD + key * KROW + sub * 16, p.k_cache + row * ROW_BYTES + piece * 16, valid ? 16 : 0);
    else cp_async16(slot + HKV * KHEAD + head_kv * VHEAD + key * VROW + sub * 16, p.v_cache + row * ROW_BYTES + piece * 16, valid ? 16 : 0);
  };
  // --- one 16-key tile of this warp's head -------------------------------------------------------------------------
  auto consume = [&](int i, const unsigned char* slot) {
    if (p.mode == 1) return;
    const unsigned char* ks = slot + hh * KHEAD;
    const unsigned char* vs = slot + HKV * KHEAD + hh * VHEAD;
    const int tile_key0 = (i < ntiles_c) ? k0 + 16 * i : seq;  // absolute position of the tile's key 0
    const int nvalid = (i < ntiles_c) ? min(16, ncached_here - 16 * i) : L;
    const bool edge = (nvalid < 16) || (tile_key0 < lo_max) || (tile_key0 + 15 > hi_min);
    // K fragments: n-tile u (keys 8u .. 8u+7): lane's 16 bytes of key 8u+g at dims 16t .. 16t+15
    uint32_t kf[2][8];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const uint4 kraw = *reinterpret_cast<const uint4*>(ks + (8 * u + g) * KROW + 16 * t);
      const uint16_t* kp = reinterpret_cast<const uint16_t*>(&kraw);
#pragma unroll
      for (int q = 0; q < 8; ++q) kf[u][q] = e4m3x2_to_f16x2(kp[q]);
    }
    // V fragments: rows 2t, 2t+1, 2t+8, 2t+9 at physical dims 8g .. 8g+7
    uint32_t vf[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int key = (r < 2) ? 2 * t + r : 2 * t + 8 + (r - 2);
      const uint2 vraw = *reinterpret_cast<const uint2*>(vs + key * VROW + 8 * g);
      const uint16_t* vp = reinterpret_cast<const uint16_t*>(&vraw);
#pragma unroll
      for (int q = 0; q < 4; ++q) vf[r][q] = e4m3x2_to_f16x2(vp[q]);
    }
    uint32_t vb0[8], vb1[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t sel = (j & 1) ? 0x7632u : 0x5410u;
      vb0[j] = __byte_perm(vf[0][j >> 1], vf[1][j >> 1], sel);
      vb1[j] = __byte_perm(vf[2][j >> 1], vf[3][j >> 1], sel);
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float sacc[2][4];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        sacc[u][0] = sacc[u][1] = sacc[u][2] = sacc[u][3] = 0.f;
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          const uint32_t a[4] = {qa[m][2 * st], qb[m][2 * st], qa[m][2 * st + 1], qb[m][2 * st + 1]};
          mma_f16(sacc[u], a, kf[u][2 * st], kf[u][2 * st + 1]);
        }
      }
      float tmax_lo = -INFINITY, tmax_hi = -INFINITY;
      if (edge) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int kk = 8 * u + 2 * t + (c & 1);
            const int key = tile_key0 + kk;
            const bool row_hi = c >= 2;
            const bool ok = (kk < nvalid) && key >= (row_hi ? lo_hi[m] : lo_lo[m]) && key <= (row_hi ? hi_hi[m] : hi_lo[m]);
            const float v = ok ? sacc[u][c] * LOG2E : -INFINITY;
            sacc[u][c] = v;
            if (row_hi) tmax_hi = fmaxf(tmax_hi, v); else tmax_lo = fmaxf(tmax_lo, v);
          }
        }
      } else {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const float v = sacc[u][c] * LOG2E;
            sacc[u][c] = v;
            if (c >= 2) tmax_hi = fmaxf(tmax_hi, v); else tmax_lo = fmaxf(tmax_lo, v);
          }
        }
      }
      tmax_lo = fmaxf(tmax_lo, __shfl_xor_sync(0xffffffffu, tmax_lo, 1));
      tmax_lo = fmaxf(tmax_lo, __shfl_xor_sync(0xffffffffu, tmax_lo, 2));
      const float mn_lo = fmaxf(m_lo[m], tmax_lo);
      const float a_lo = (mn_lo == -INFINITY) ? 1.f : exp2f(m_lo[m] - mn_lo);
      m_lo[m] = mn_lo;
      l_lo[m] *= a_lo;
      float mn_hi = 0.f, a_hi = 1.f;
      if (L > 1) {  // decode's rows 8..15 are padding: no softmax bookkeeping for them
        tmax_hi = fmaxf(tmax_hi, __shfl_xor_sync(0xffffffffu, tmax_hi, 1));
        tmax_hi = fmaxf(tmax_hi, __shfl_xor_sync(0xffffffffu, tmax_hi, 2));
        mn_hi = fmaxf(m_hi[m], tmax_hi);
        a_hi = (mn_hi == -INFINITY) ? 1.f : exp2f(m_hi[m] - mn_hi);
        m_hi[m] = mn_hi;
        l_hi[m] *= a_hi;
      }
      if (__any_sync(0xffffffffu, a_lo != 1.f || a_hi != 1.f)) {  // the running max moved for some row: rescale O
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          o[m][j][0] *= a_lo; o[m][j][1] *= a_lo;
          if (L > 1) { o[m][j][2] *= a_hi; o[m][j][3] *= a_hi; }
        }
      }
      uint32_t pa[4];
      {
        float pv[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const bool row_hi = c >= 2;
            if (row_hi && L == 1) { pv[u][c] = 0.f; continue; }
            const float mm = row_hi ? mn_hi : mn_lo;
            const float e = (sacc[u][c] == -INFINITY) ? 0.f : exp2f(sacc[u][c] - mm);
            pv[u][c] = e;
            if (row_hi) l_hi[m] += e; else l_lo[m] += e;
          }
        }
        pa[0] = pack_f16x2(pv[0][0], pv[0][1]);
        pa[1] = (L == 1) ? 0u : pack_f16x2(pv[0][2], pv[0][3]);
        pa[2] = pack_f16x2(pv[1][0], pv[1][1]);
        pa[3] = (L == 1) ? 0u : pack_f16x2(pv[1][2], pv[1][3]);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) mma_f16(o[m][j], pa, vb0[j], vb1[j]);
    }
  };

  // --- main loop. Decode (L == 1): the two warps that share tile index wq (one per kv head) form a pair with a
  //     private PR_ST-stage ring; the pair's 64 threads fill their own tiles (4 x 16 B each), wait per pair with a
  //     named barrier and consume independently of the other pairs (no CTA-wide barrier in the loop). Verify (L == 8):
  //     groups of G tiles, NG groups in flight, every warp of a head consumes every tile (different rows). ----------
  if constexpr (L == 1) {
    const int tp = tid & 63;                              // thread within the pair (warps 2 wq and 2 wq + 1)
    unsigned char* pring = ring + (wq * PR_ST) * SLOT;    // this pair's stages
    const int ntp = (ntiles_c > wq) ? (ntiles_c - wq + 3) / 4 : 0;  // this pair's tiles: wq, wq + 4, ...
    auto fill_pair = [&](int k) {                         // k-th tile of the pair -> stage k % PR_ST
      if (k >= ntp) return;
      const int i = wq + 4 * k;
      unsigned char* slot = pring + (k % PR_ST) * SLOT;
      if constexpr (HALF_FILL) {
        // this warp copies its head's halves: 16 keys x (K, V) x 4 pieces of 16 B = 128 copies, 4 per lane
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int it = r * 32 + lane;
          const int tensor = it >> 6, key = (it >> 2) & 15, sub = it & 3;
          const int idx = i * 16 + key;
          const bool valid = idx < ncached_here;
          const size_t row = valid ? static_cast<size_t>(ids[idx]) : 0;
          unsigned char* dst = tensor ? slot + HKV * KHEAD + hh * VHEAD + key * VROW + sub * 16
                                      : slot + hh * KHEAD + key * KROW + sub * 16;
          const uint8_t* src = (tensor ? p.v_cache : p.k_cache) + row * ROW_BYTES + hh * 64 + sub * 16;
          cp_async16(dst, src, valid ? 16 : 0);
        }
      } else {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int it = r * 64 + tp;                       // 256 copies: tensor (K/V), key, 16-byte piece
          const int tensor = it >> 7, key = (it >> 3) & 15, piece = it & 7, head_kv = piece >> 2, sub = piece & 3;
          const int idx = i * 16 + key;
          const bool valid = idx < ncached_here;
          const size_t row = valid ? static_cast<size_t>(ids[idx]) : 0;
          unsigned char* dst = tensor ? slot + HKV * KHEAD + head_kv * VHEAD + key * VROW + sub * 16
                                      : slot + head_kv * KHEAD + key * KROW + sub * 16;
          const uint8_t* src = (tensor ? p.v_cache : p.k_cache) + row * ROW_BYTES + piece * 16;
          cp_async16(dst, src, valid ? 16 : 0);
        }
      }
    };
#pragma unroll
    for (int k = 0; k < PR_ST - 1; ++k) { fill_pair(k); cp_async_commit(); }
    if (p.mode != 2) for (int k = 0; k < ntp; ++k) {
      cp_async_wait<PR_ST - 2>();                         // tile k landed (k + 1, k + 2 may still be in flight)
      if constexpr (!HALF_FILL) asm volatile("bar.sync %0, 64;" ::"r"(1 + wq) : "memory");  // both warps of the pair see all 64 threads' copies
      else __syncwarp();
      consume(wq + 4 * k, pring + (k % PR_ST) * SLOT);
      fill_pair(k + PR_ST - 1);                           // into the stage consumed one iteration ago (both warps passed the barrier since)
      cp_async_commit();
    }
  } else {
    const int ngroups = (ntiles_c + G - 1) / G;
#pragma unroll
    for (int pg = 0; pg < NG - 1; ++pg) {
      for (int q = 0; q < G; ++q) fill(pg * G + q);
      cp_async_commit();
    }
    for (int grp = 0; grp < ngroups; ++grp) {
      for (int q = 0; q < G; ++q) fill((grp + NG - 1) * G + q);
      cp_async_commit();
      cp_async_wait<NG - 1>();
      __syncthreads();
#pragma unroll 1
      for (int q = 0; q < G; ++q) {
        const int i = grp * G + q;
        if (i >= ntiles_c) break;
        consume(i, ring + (i % RS) * SLOT);
      }
      __syncthreads();
    }
  }
  cp_async_wait<0>();
  if (tail) {
    if (L == 1) { if (wq == 0) consume(ntiles_c, newslot); }
    else consume(ntiles_c, newslot);
  }
  if (p.mode == 1) return;
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    l_lo[m] += __shfl_xor_sync(0xffffffffu, l_lo[m], 1); l_lo[m] += __shfl_xor_sync(0xffffffffu, l_lo[m], 2);
    l_hi[m] += __shfl_xor_sync(0xffffffffu, l_hi[m], 1); l_hi[m] += __shfl_xor_sync(0xffffffffu, l_hi[m], 2);
  }

  // --- partials: lane holds row g dims 16t + j (o[j][0]) and 16t + 8 + j (o[j][1]); row g+8 in [2], [3] ------------
  constexpr int ROWS = HKV * 8 * L;
  const size_t pbase = (static_cast<size_t>(b) * p.splits + s) * ROWS;
  if (L == 1) {
    __syncthreads();  // ring reads are done; reuse it as scratch: [NW warps][8 rows] m, l and [NW][8][64] O
    float* rm = red;
    float* rl = red + NW * 8;
    float* ro = red + 2 * NW * 8;
    if (t == 0) { rm[warp * 8 + g] = m_lo[0]; rl[warp * 8 + g] = l_lo[0]; }
#pragma unroll
    for (int j = 0; j < 8; ++j) { ro[(warp * 8 + g) * 64 + 16 * t + j] = o[0][j][0]; ro[(warp * 8 + g) * 64 + 16 * t + 8 + j] = o[0][j][1]; }
    __syncthreads();
    if (warp < 2) {  // warp hh merges the head's NWH warps (hh, hh + 2, ...)
      float mm = -INFINITY;
#pragma unroll
      for (int k = 0; k < NWH; ++k) mm = fmaxf(mm, rm[(warp + 2 * k) * 8 + g]);
      float cc[NWH], ll = 0.f;
#pragma unroll
      for (int k = 0; k < NWH; ++k) {
        const float mk = rm[(warp + 2 * k) * 8 + g];
        cc[k] = (mk == -INFINITY) ? 0.f : exp2f(mk - mm);
        ll += rl[(warp + 2 * k) * 8 + g] * cc[k];
      }
      const int row = warp * 8 + g;
      float* po = p.part_o + (pbase + row) * 64 + 16 * t;
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < NWH; ++k) acc += ro[((warp + 2 * k) * 8 + g) * 64 + 16 * t + d] * cc[k];
        po[d] = acc;
      }
      if (t == 0) { p.part_ml[(pbase + row) * 2] = mm; p.part_ml[(pbase + row) * 2 + 1] = ll; }
    }
  } else {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const int r_lo = hh * 8 * L + 16 * (mt0 + m) + g, r_hi = r_lo + 8;
      float* po_lo = p.part_o + (pbase + r_lo) * 64 + 16 * t;
      float* po_hi = p.part_o + (pbase + r_hi) * 64 + 16 * t;
#pragma unroll
      for (int j = 0; j < 8; ++j) { po_lo[j] = o[m][j][0]; po_lo[8 + j] = o[m][j][1]; po_hi[j] = o[m][j][2]; po_hi[8 + j] = o[m][j][3]; }
      if (t == 0) {
        p.part_ml[(pbase + r_lo) * 2] = m_lo[m]; p.part_ml[(pbase + r_lo) * 2 + 1] = l_lo[m];
        p.part_ml[(pbase + r_hi) * 2] = m_hi[m]; p.part_ml[(pbase + r_hi) * 2 + 1] = l_hi[m];
      }
    }
  }
}

// Merge: one CTA of MW warps per (b, j, kv head, query head): lane -> dims 2 lane, 2 lane + 1; warp w folds splits
// w, w + MW, ... (branch-free so the loads pipeline); the warps combine through shared memory and warp 0 applies 1/l,
// v_scale, the output RMS norm and the sigmoid gate.
__global__ void __launch_bounds__(MW * 32) attn_merge_kernel(const Params p) {
  __shared__ float sm_m[MW], sm_l[MW], sm_o[MW][64];
  const int gq = blockIdx.x & 7, hkv = (blockIdx.x >> 3) % HKV, j = (blockIdx.x / (8 * HKV)) % p.L, b = blockIdx.x / (8 * HKV * p.L);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int group = p.Hq / HKV;
  const int rows = HKV * 8 * p.L, row = hkv * 8 * p.L + 8 * j + gq;
  const size_t base = static_cast<size_t>(b) * p.splits;
  gdc_launch();
  gdc_wait();
  float mm = -INFINITY, ll = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll 4
  for (int s = warp; s < p.splits; s += MW) {
    const float2 ml = *reinterpret_cast<const float2*>(p.part_ml + ((base + s) * rows + row) * 2);
    const float2 v = *reinterpret_cast<const float2*>(p.part_o + ((base + s) * rows + row) * 64 + 2 * lane);
    const float mn = fmaxf(mm, ml.x);
    const float a = (mn == -INFINITY) ? 0.f : exp2f(mm - mn), c = (mn == -INFINITY) ? 0.f : exp2f(ml.x - mn);
    mm = mn;
    ll = ll * a + ml.y * c;
    a0 = a0 * a + v.x * c;
    a1 = a1 * a + v.y * c;
  }
  if (lane == 0) { sm_m[warp] = mm; sm_l[warp] = ll; }
  sm_o[warp][2 * lane] = a0; sm_o[warp][2 * lane + 1] = a1;
  __syncthreads();
  if (warp == 0) {
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < MW; ++w) M = fmaxf(M, sm_m[w]);
    float Lsum = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
    for (int w = 0; w < MW; ++w) {
      const float c = (sm_m[w] == -INFINITY) ? 0.f : exp2f(sm_m[w] - M);
      Lsum += sm_l[w] * c;
      o0 += sm_o[w][2 * lane] * c;
      o1 += sm_o[w][2 * lane + 1] * c;
    }
    const float inv = (Lsum > 0.f ? 1.f / Lsum : 0.f) / p.inv_v_scale;  // / l, * v_scale
    o0 *= inv; o1 *= inv;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const int qkv_stride = (2 * p.Hq + 2 * HKV) * 64;
    const int head = hkv * group + gq;
    const __nv_bfloat162 gate = *reinterpret_cast<const __nv_bfloat162*>(
        p.qkv + (static_cast<size_t>(b) * p.L + j) * qkv_stride + (p.Hq + 2 * HKV) * 64 + head * 64 + 2 * lane);
    const float2 gf = __bfloat1622float2(gate);
    const __nv_bfloat162 r = __floats2bfloat162_rn(o0 * rn * (1.f / (1.f + expf(-gf.x))), o1 * rn * (1.f / (1.f + expf(-gf.y))));
    *reinterpret_cast<__nv_bfloat162*>(p.out + (static_cast<size_t>(b) * p.L + j) * (p.Hq * 64) + head * 64 + 2 * lane) = r;
  }
}

}  // namespace okcattn
using namespace okcattn;

static bool g_attn_pdl = true;
void attn_decode(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                 torch::Tensor seq_lens, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor out,
                 int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t page_size, int64_t window, bool rope,
                 double qk_scale, double sm_scale, double k_scale, double v_scale, int64_t splits, int64_t chunk, int64_t mode) {
  TORCH_CHECK(L == 1 || L == 8, "attn decode: q_len 1 or 8");
  TORCH_CHECK(Hkv == HKV && Hq == 8 * HKV, "attn decode: 16 query heads over 2 kv heads");
  TORCH_CHECK(chunk % 16 == 0 && chunk <= MAX_CHUNK && chunk > 0, "attn decode: chunk");
  TORCH_CHECK(qkv.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous() && page_table.is_contiguous(), "attn decode: contiguous");
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = page_table.data_ptr<int32_t>();
  p.seq_lens = seq_lens.data_ptr<int32_t>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.part_o = part_o.data_ptr<float>();
  p.part_ml = part_ml.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.B = B; p.L = L; p.Hq = Hq; p.page_size = page_size; p.max_pages = page_table.size(1);
  p.window = window; p.splits = splits; p.chunk = chunk; p.rope = rope ? 1 : 0; p.mode = mode;
  p.qk_scale = static_cast<float>(qk_scale);
  p.q_mult = static_cast<float>(sm_scale * k_scale);
  p.inv_k_scale = static_cast<float>(1.0 / k_scale); p.inv_v_scale = static_cast<float>(1.0 / v_scale);
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(attn_decode_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<1>::BYTES);
    cudaFuncSetAttribute(attn_decode_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<8>::BYTES);
    attr_set = true;
  }
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = g_attn_pdl ? 1 : 0;
  lc.attrs = at; lc.numAttrs = 1; lc.stream = stream;
  lc.gridDim = dim3(splits, B); lc.blockDim = dim3(NW * 32);
  cudaError_t err;
  if (L == 1) { lc.dynamicSmemBytes = Smem<1>::BYTES; err = cudaLaunchKernelEx(&lc, attn_decode_kernel<1>, p); }
  else { lc.dynamicSmemBytes = Smem<8>::BYTES; err = cudaLaunchKernelEx(&lc, attn_decode_kernel<8>, p); }
  TORCH_CHECK(err == cudaSuccess, "attn decode launch: ", cudaGetErrorString(err));
  if (mode == 0) {
    lc.gridDim = dim3(B * L * HKV * 8); lc.blockDim = dim3(MW * 32); lc.dynamicSmemBytes = 0;
    err = cudaLaunchKernelEx(&lc, attn_merge_kernel, p);
    TORCH_CHECK(err == cudaSuccess, "attn merge launch: ", cudaGetErrorString(err));
  }
}
void attn_set_pdl(bool on) { g_attn_pdl = on; }
"""

_TEAMV3_ATTN_DECODE_CPP = r"""
void attn_set_pdl(bool on);
void attn_decode(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                 torch::Tensor seq_lens, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor out,
                 int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t page_size, int64_t window, bool rope,
                 double qk_scale, double sm_scale, double k_scale, double v_scale, int64_t splits, int64_t chunk, int64_t mode);
"""

_TEAMV3__ATTN_EXT = None
_TEAMV3__INV_FREQ = {}
_TEAMV3_ATTN_NG = 3  # groups in flight (tuning knob; the extension name carries it)
_TEAMV3_ATTN_MAX_CHUNK = 4096  # keys per CTA at most (page ids in shared memory: 16 KB)
_TEAMV3_ATTN_WAVE_KEYS = 512  # per-CTA fixed cost in key-equivalents (prologue), for the wave-aware split choice
_TEAMV3_ATTN_MIN_CHUNK = 128
_TEAMV3_ATTN_TARGET_CTAS = 264  # ~2 CTAs per SM on the H200


def _TEAMV3_attn_ext():
    global _TEAMV3__ATTN_EXT
    if _TEAMV3__ATTN_EXT is None:
        from torch.utils.cpp_extension import load_inline as _TEAMV3_load_inline
        major, minor = _TEAMV3_torch.cuda.get_device_capability()
        arch = f"{major}{minor}{'a' if major == 9 else ''}"
        import os
        ng = int(os.environ.get("OKC_ATTN_NG", _TEAMV3_ATTN_NG))
        half = int(os.environ.get("OKC_ATTN_HALF", "0"))
        maxc = int(os.environ.get("OKC_ATTN_MAXCHUNK", _TEAMV3_ATTN_MAX_CHUNK))
        src = _TEAMV3_ATTN_DECODE_SRC.replace("__OKC_ATTN_NG__", str(ng)).replace("__OKC_ATTN_HALF__", str(half)).replace("__OKC_ATTN_MAXCHUNK__", str(maxc))
        _TEAMV3__ATTN_EXT = _TEAMV3_load_inline(f"okc_attn_decode_pdl_ng{ng}_h{half}_c{maxc}", cpp_sources=_TEAMV3_ATTN_DECODE_CPP, cuda_sources=src, functions=["attn_decode", "attn_set_pdl"],
                                extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo"], verbose=False)
    return _TEAMV3__ATTN_EXT


def _TEAMV3_attn_inv_freq(theta: float, device) -> _TEAMV3_torch.Tensor:
    """RoPE inverse frequencies exactly as the reference computes them (fp32); a constant of the layer's config."""
    key = (float(theta), str(device))
    if key not in _TEAMV3__INV_FREQ:
        if theta:
            _TEAMV3__INV_FREQ[key] = (1.0 / (theta ** (_TEAMV3_torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
        else:
            _TEAMV3__INV_FREQ[key] = _TEAMV3_torch.zeros(32, dtype=_TEAMV3_torch.float32, device=device)
    return _TEAMV3__INV_FREQ[key]


def _TEAMV3_attn_splits(batch: int, max_keys: int, window: int):
    """(splits, chunk): enough CTAs to cover the GPU, chunks between _TEAMV3_ATTN_MIN_CHUNK and _TEAMV3_ATTN_MAX_CHUNK keys. When the
    grid needs more than one wave of _TEAMV3_ATTN_TARGET_CTAS CTAs, the split count is chosen to fill whole waves: the cost
    model is ceil(CTAs / slots) x (chunk + _TEAMV3_ATTN_WAVE_KEYS)."""
    keys = min(max_keys, window) if window else max_keys
    keys = max(keys, 16)
    splits = max(1, -(-_TEAMV3_ATTN_TARGET_CTAS // batch))
    splits = min(splits, max(1, -(-keys // _TEAMV3_ATTN_MIN_CHUNK)))
    splits = max(splits, -(-keys // _TEAMV3_ATTN_MAX_CHUNK))
    if batch * splits > _TEAMV3_ATTN_TARGET_CTAS and keys >= 16384:  # wave-filling split choice for long contexts only
        best = None
        for s in range(splits, 2 * splits + 1):
            ch = -(-keys // s)
            ch = -(-ch // 16) * 16
            if ch < _TEAMV3_ATTN_MIN_CHUNK:
                break
            s_eff = max(1, -(-keys // ch))
            cost = -(-(batch * s_eff) // _TEAMV3_ATTN_TARGET_CTAS) * (ch + _TEAMV3_ATTN_WAVE_KEYS)
            if best is None or cost < best[0]:
                best = (cost, s_eff)
        splits = best[1]
    chunk = -(-keys // splits)
    chunk = -(-chunk // 16) * 16
    splits = max(1, -(-keys // chunk))
    return splits, chunk


def _TEAMV3_attn_decode_forward(data, max_keys: int, out=None, mode: int = 0, splits_chunk=None):
    """One decode / verify step from the problem's inputs. max_keys bounds cache_seqlens (shape-derived: the page
    table's capacity x page size); it only sizes the split."""
    cfg = data["config"]
    B, L, Hq, Hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    splits, chunk = splits_chunk or _TEAMV3_attn_splits(B, max_keys, cfg["window"])
    dev = qkv.device
    rows = Hkv * 8 * L
    part_o = _TEAMV3_torch.empty((B, splits, rows, 64), dtype=_TEAMV3_torch.float32, device=dev)
    part_ml = _TEAMV3_torch.empty((B, splits, rows, 2), dtype=_TEAMV3_torch.float32, device=dev)
    if out is None:
        out = _TEAMV3_torch.empty((B, L, Hq * 64), dtype=_TEAMV3_torch.bfloat16, device=dev)
    theta = float(cfg["rope_theta"])
    _TEAMV3_attn_ext().attn_decode(qkv.contiguous(), data["cache_loc"], data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"],
                           _TEAMV3_attn_inv_freq(theta, dev), part_o, part_ml, out, B, L, Hq, Hkv, cfg["page_size"], cfg["window"], theta != 0.0,
                           float(cfg["qk_scale_factor"]), float(cfg["sm_scale"]), float(cfg["kv_scale"]), float(cfg["kv_scale"]),
                           splits, chunk, mode)
    return out
# Verify-block attention (8 queries per request) over the paged FP8 E4M3 KV cache on Hopper's warpgroup tensor cores.
# Same split-KV skeleton as the decode kernel above (grid = (splits, batch); one CTA per request, both kv heads, a
# contiguous key range; raw K/V rows gathered through the page table into a shared-memory ring with cp.async; the
# block's new tokens normalized, quantized, stored and attended from the same bytes; partials merged by a second kernel
# that applies the output RMS norm and the sigmoid gate). The math differs: each 64-key tile is converted once from e4m3
# to f16 into 128B-swizzled shared-memory tiles, and each warpgroup (one per kv head) runs the FA2 recurrence on
# wgmma.m64n64k16 (f16 x f16 -> f32): S = Q K^T with Q (8 queries x 8 heads = 64 rows) in registers and K as a K-major
# descriptor, softmax on the accumulators, P packed to f16 and reused as the register A operand, O += P V with V as an
# MN-major descriptor. One wgmma covers 64 rows x 64 keys x 16 dims, where mma.sync needs 32 instructions.
import torch as _TEAMV3_torch

_TEAMV3_ATTN_VERIFY_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace okcattnv {

constexpr int L = 8;               // queries per request (a verify block)
constexpr int HKV = 2;             // kv heads per CTA
constexpr int NW = 8;              // warps per CTA: warpgroup w / 4 = kv head
constexpr int TK = 64;             // keys per tile (= wgmma N)
constexpr int NG = 2;              // raw tiles in flight (cp.async)
constexpr int RD = NG;             // raw slots (the new tokens go straight into an f16 buffer)
constexpr int MAX_CHUNK = 2048;    // keys per CTA at most (page ids staged in shared memory)
constexpr int RAW_ROW = HKV * 64;  // 128 B: one token's K (or V) row, both heads
constexpr int RAW_SLOT = 2 * TK * RAW_ROW;  // 16 KB: K rows then V rows
constexpr int T16 = TK * 128;      // 8 KB: one head's f16 tile (64 keys x 64 dims), 128B-swizzled rows
constexpr int QROWS = HKV * 64;    // 8 queries x 8 heads per kv head
constexpr int MW = 8;              // merge warps
constexpr float LOG2E = 1.4426950408889634f;
constexpr int F16BUF = 2 * HKV * T16;  // 32 KB: K tiles (both heads) then V tiles
constexpr int OFF_IDS = 0, OFF_RAW = MAX_CHUNK * 4, OFF_F16 = OFF_RAW + RD * RAW_SLOT, SMEM_BYTES = OFF_F16 + 2 * F16BUF;
constexpr int OFF_Q = OFF_F16 + F16BUF;  // Q is staged in f16 buffer 1 before the loop (tile 0 converts into buffer 0)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ uint64_t make_desc(const void* smem, uint32_t lbo_bytes, uint32_t sbo_bytes, int swz) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint64_t d = 0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)swz << 62;  // 1 = 128B swizzle
  return d;
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void wg_barrier(int id) { asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory"); }  // one warpgroup
// D[64 x 64] f32 (32 regs) (+)= A[64 x 16] f16 (registers) * B (smem descriptor; TRANS_B = 1: MN-major)
template <int TRANS_B>
__device__ __forceinline__ void wgmma_rs(float d[32], const uint32_t a[4], uint64_t descb, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
      "{%32,%33,%34,%35}, %36, p, 1, 1, %38;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(descb), "r"(scale_d), "n"(TRANS_B));
}

struct Params {
  const __nv_bfloat16* qkv;   // [B, L, (2 Hq + 2 Hkv) * 64]
  const int64_t* cache_loc;   // [B * L]
  uint8_t* k_cache;           // [pages, ps, Hkv, 64] e4m3
  uint8_t* v_cache;
  const int32_t* page_table;  // [B, max_pages]
  const int32_t* seq_lens;    // [B]
  const float* inv_freq;      // [32]
  float* part_o;              // [B, S, QROWS, 64]
  float* part_ml;             // [B, S, QROWS, 2]
  __nv_bfloat16* out;         // [B, L, Hq * 64]
  int B, Hq, page_size, max_pages, window, splits, chunk;
  float qk_scale, q_mult, inv_k_scale, inv_v_scale;  // q_mult = sm_scale * k_scale
  int rope, mode;
};

__global__ void __launch_bounds__(NW * 32, 2) attn_verify_kernel(const Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  int32_t* ids = reinterpret_cast<int32_t*>(smem + OFF_IDS);
  unsigned char* raw = smem + OFF_RAW;
  unsigned char* f16 = smem + OFF_F16;  // buffer bb: K tiles at bb * F16BUF + hh * T16, V tiles at + HKV * T16
  uint16_t* qs = reinterpret_cast<uint16_t*>(smem + OFF_Q);  // [HKV][64 rows][64] f16, natural order

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int hh = warp >> 2, w4 = warp & 3;  // kv head (warpgroup), warp within it: rows 16 w4 .. +15 (queries 2 w4, 2 w4 + 1)
  const int s = blockIdx.x, b = blockIdx.y;
  const int group = p.Hq / HKV;
  const int seq = p.seq_lens[b];
  const int first = p.window ? max(0, seq - p.window + 1) : 0;
  const int k0 = first + s * p.chunk;
  const int k1 = min(seq, k0 + p.chunk);
  const bool tail = (s == p.splits - 1);
  const int ncached_here = max(0, k1 - k0);
  const int ntiles_c = (ncached_here + TK - 1) / TK;
  const int qkv_stride = (2 * p.Hq + 2 * HKV) * 64;

  for (int i = tid; i < ncached_here; i += NW * 32) {
    const int pos = k0 + i;
    ids[i] = p.page_table[static_cast<size_t>(b) * p.max_pages + pos / p.page_size] * p.page_size + pos % p.page_size;
  }
  // --- queries: rms * qk_scale (RoPE on sliding) -> bf16 -> * sm_scale * k_scale -> f16; 2 threads x 32 dims per row
  {
    constexpr int TPR = 2, DPT = 32;
    for (int base = 0; base < QROWS * TPR; base += NW * 32) {
      const int it = base + tid;
      const int rr = it / TPR, part = it % TPR;
      const int head_kv = rr / 64, r = rr % 64, j = r >> 3, gq = r & 7;
      const __nv_bfloat16* q = p.qkv + (static_cast<size_t>(b) * L + j) * qkv_stride + (head_kv * group + gq) * 64 + part * DPT;
      float v[DPT];
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < DPT; i += 8) {
        const uint4 u = *reinterpret_cast<const uint4*>(q + i);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 f = __bfloat1622float2(b2[k]);
          v[i + 2 * k] = f.x; v[i + 2 * k + 1] = f.y;
          ss += f.x * f.x + f.y * f.y;
        }
      }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f) * p.qk_scale;
      if (p.rope) {
        const float pos = static_cast<float>(seq + j);
#pragma unroll
        for (int i = 0; i < DPT; i += 2) {
          float sn, cs;
          sincosf(pos * p.inv_freq[(part * DPT + i) >> 1], &sn, &cs);
          const float e = v[i] * inv, o = v[i + 1] * inv;
          v[i] = e * cs - o * sn;
          v[i + 1] = e * sn + o * cs;
        }
      } else {
#pragma unroll
        for (int i = 0; i < DPT; ++i) v[i] *= inv;
      }
      uint16_t* dst = qs + (head_kv * 64 + r) * 64 + part * DPT;
#pragma unroll
      for (int i = 0; i < DPT; i += 2) {
        *reinterpret_cast<uint32_t*>(dst + i) = pack_f16x2(bf16_round(v[i]) * p.q_mult, bf16_round(v[i + 1]) * p.q_mult);
      }
    }
  }
  // --- the block's new tokens: normalized keys (RoPE on sliding), quantized with the cache scales, written to the cache;
  //     the tail CTA keeps the e4m3 bytes in registers for its f16 tail tile (built after the loop) ----------------
  uint2 newk = make_uint2(0u, 0u), newv = make_uint2(0u, 0u);  // this thread's (key, head, 8-byte chunk) of the tail tile
  const int nt_key = tid >> 4, nt_head = (tid >> 3) & 1, nt_c = tid & 7;  // 256 threads = 16 keys x 2 heads x 8 chunks
  if (tail && nt_key < L) {
    const __nv_bfloat16* row = p.qkv + (static_cast<size_t>(b) * L + nt_key) * qkv_stride;
    const __nv_bfloat16* kin = row + p.Hq * 64 + nt_head * 64;
    const __nv_bfloat16* vin = row + p.Hq * 64 + HKV * 64 + nt_head * 64;
    float ss = 0.f;
#pragma unroll
    for (int d = 0; d < 64; d += 8) {
      const uint4 u = *reinterpret_cast<const uint4*>(kin + d);
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b2[k]); ss += f.x * f.x + f.y * f.y; }
    }
    const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    float kv[8], vv[8];
    {
      const uint4 u = *reinterpret_cast<const uint4*>(kin + 8 * nt_c);
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b2[k]); kv[2 * k] = f.x * inv; kv[2 * k + 1] = f.y * inv; }
      const uint4 w = *reinterpret_cast<const uint4*>(vin + 8 * nt_c);
      const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
      for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(c2[k]); vv[2 * k] = f.x; vv[2 * k + 1] = f.y; }
    }
    if (p.rope) {
      const float pos = static_cast<float>(seq + nt_key);
#pragma unroll
      for (int k = 0; k < 8; k += 2) {
        float sn, cs;
        sincosf(pos * p.inv_freq[(8 * nt_c + k) >> 1], &sn, &cs);
        const float e = kv[k], o = kv[k + 1];
        kv[k] = e * cs - o * sn;
        kv[k + 1] = e * sn + o * cs;
      }
    }
    uint8_t kq[8], vq[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      kq[k] = __nv_cvt_float_to_fp8(kv[k] * p.inv_k_scale, __NV_SATFINITE, __NV_E4M3);
      vq[k] = __nv_cvt_float_to_fp8(vv[k] * p.inv_v_scale, __NV_SATFINITE, __NV_E4M3);
    }
    newk = *reinterpret_cast<uint2*>(kq);
    newv = *reinterpret_cast<uint2*>(vq);
    const int64_t loc = p.cache_loc[b * L + nt_key];
    *reinterpret_cast<uint2*>(p.k_cache + static_cast<size_t>(loc) * RAW_ROW + nt_head * 64 + 8 * nt_c) = newk;
    *reinterpret_cast<uint2*>(p.v_cache + static_cast<size_t>(loc) * RAW_ROW + nt_head * 64 + 8 * nt_c) = newv;
  }
  __syncthreads();

  // --- Q as the register A operand: k-step st: a0 = Q[row g][16 st + 2t, +1], a1 = row g+8, a2 = cols + 8, a3
  uint32_t qa[4][4];
  {
    const uint16_t* r0 = qs + (hh * 64 + 16 * w4 + g) * 64;
    const uint16_t* r1 = qs + (hh * 64 + 16 * w4 + g + 8) * 64;
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      qa[st][0] = *reinterpret_cast<const uint32_t*>(r0 + 16 * st + 2 * t);
      qa[st][1] = *reinterpret_cast<const uint32_t*>(r1 + 16 * st + 2 * t);
      qa[st][2] = *reinterpret_cast<const uint32_t*>(r0 + 16 * st + 8 + 2 * t);
      qa[st][3] = *reinterpret_cast<const uint32_t*>(r1 + 16 * st + 8 + 2 * t);
    }
  }
  const int j_lo = 2 * w4, j_hi = 2 * w4 + 1;  // query index of rows g / g+8
  const int lo_lo = p.window ? seq + j_lo - p.window + 1 : 0, lo_hi = p.window ? seq + j_hi - p.window + 1 : 0;
  const int hi_lo = seq + j_lo, hi_hi = seq + j_hi;
  const int lo_max = p.window ? lo_hi : 0, hi_min = hi_lo;

  float o[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.f;
  float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.f, l_hi = 0.f;

  // --- raw fill: tile i -> slot i % NG; each thread copies 32 B of K and 32 B of V: key tid / 4, 32-B piece tid % 4
  auto fill = [&](int i) {
    if (i >= ntiles_c) return;
    unsigned char* slot = raw + (i % NG) * RAW_SLOT;
    const int key = tid >> 2, piece = tid & 3;
    const int idx = i * TK + key;
    const bool valid = idx < ncached_here;
    const size_t row = valid ? static_cast<size_t>(ids[idx]) : 0;
    const uint8_t* ksrc = p.k_cache + row * RAW_ROW + piece * 32;
    const uint8_t* vsrc = p.v_cache + row * RAW_ROW + piece * 32;
    cp_async16(slot + key * RAW_ROW + piece * 32, ksrc, valid ? 16 : 0);
    cp_async16(slot + key * RAW_ROW + piece * 32 + 16, ksrc + 16, valid ? 16 : 0);
    cp_async16(slot + TK * RAW_ROW + key * RAW_ROW + piece * 32, vsrc, valid ? 16 : 0);
    cp_async16(slot + TK * RAW_ROW + key * RAW_ROW + piece * 32 + 16, vsrc + 16, valid ? 16 : 0);
  };
  // --- convert a raw slot's rows of this warpgroup's head to an f16 tile: row = key (128 B), chunk c -> c ^ (key % 8)
  auto convert = [&](const unsigned char* src_rows, unsigned char* dst_tile) {
    const int wt = tid & 127;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int it = wt + r * 128;                 // 256 items: (key, 16-byte chunk)
      const int c2 = it & 3, key = it >> 2;
      const int sw = key & 7;
      const uint4 rawv = *reinterpret_cast<const uint4*>(src_rows + key * RAW_ROW + hh * 64 + c2 * 16);
      const uint16_t* rp = reinterpret_cast<const uint16_t*>(&rawv);
      uint4 fa, fb;
      fa.x = e4m3x2_to_f16x2(rp[0]); fa.y = e4m3x2_to_f16x2(rp[1]); fa.z = e4m3x2_to_f16x2(rp[2]); fa.w = e4m3x2_to_f16x2(rp[3]);
      fb.x = e4m3x2_to_f16x2(rp[4]); fb.y = e4m3x2_to_f16x2(rp[5]); fb.z = e4m3x2_to_f16x2(rp[6]); fb.w = e4m3x2_to_f16x2(rp[7]);
      unsigned char* d = dst_tile + key * 128;
      *reinterpret_cast<uint4*>(d + (((2 * c2) ^ sw) << 4)) = fa;
      *reinterpret_cast<uint4*>(d + (((2 * c2 + 1) ^ sw) << 4)) = fb;
    }
  };
  // --- one 64-key tile for this warpgroup's head: S = Q K^T issued async (the V conversion overlaps it), then softmax
  //     and O += P V (its wait is deferred to the next tile) --------------------------------------------------------
  float sacc[32];
  auto qk_issue = [&](int bb) {
    const unsigned char* kt = f16 + bb * F16BUF + hh * T16;
    wgmma_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      wgmma_rs<0>(sacc, qa[st], make_desc(kt + st * 32, 16, 1024, 1), st > 0 ? 1 : 0);
    }
    wgmma_commit();
  };
  auto softmax_pv = [&](int bb, int tile_key0, int nvalid) {
    const bool edge = (nvalid < TK) || (tile_key0 < lo_max) || (tile_key0 + TK - 1 > hi_min);
    wgmma_wait<0>();  // S (and the previous tile's P V)
    // C layout: sacc[4j + c]: rows g (c < 2) / g+8, keys 8j + 2t + (c & 1)
    float tmax_lo = -INFINITY, tmax_hi = -INFINITY;
    if (edge) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * t + (c & 1);
          const int key = tile_key0 + kk;
          const bool row_hi = c >= 2;
          const bool ok = (kk < nvalid) && key >= (row_hi ? lo_hi : lo_lo) && key <= (row_hi ? hi_hi : hi_lo);
          const float v = ok ? sacc[4 * j + c] * LOG2E : -INFINITY;
          sacc[4 * j + c] = v;
          if (row_hi) tmax_hi = fmaxf(tmax_hi, v); else tmax_lo = fmaxf(tmax_lo, v);
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const float v = sacc[4 * j + c] * LOG2E;
          sacc[4 * j + c] = v;
          if (c >= 2) tmax_hi = fmaxf(tmax_hi, v); else tmax_lo = fmaxf(tmax_lo, v);
        }
      }
    }
    tmax_lo = fmaxf(tmax_lo, __shfl_xor_sync(0xffffffffu, tmax_lo, 1));
    tmax_lo = fmaxf(tmax_lo, __shfl_xor_sync(0xffffffffu, tmax_lo, 2));
    tmax_hi = fmaxf(tmax_hi, __shfl_xor_sync(0xffffffffu, tmax_hi, 1));
    tmax_hi = fmaxf(tmax_hi, __shfl_xor_sync(0xffffffffu, tmax_hi, 2));
    const float mn_lo = fmaxf(m_lo, tmax_lo), mn_hi = fmaxf(m_hi, tmax_hi);
    const float a_lo = (mn_lo == -INFINITY) ? 1.f : exp2f(m_lo - mn_lo);
    const float a_hi = (mn_hi == -INFINITY) ? 1.f : exp2f(m_hi - mn_hi);
    m_lo = mn_lo; m_hi = mn_hi;
    l_lo *= a_lo; l_hi *= a_hi;
    if (__any_sync(0xffffffffu, a_lo != 1.f || a_hi != 1.f)) {
#pragma unroll
      for (int j = 0; j < 8; ++j) { o[4 * j] *= a_lo; o[4 * j + 1] *= a_lo; o[4 * j + 2] *= a_hi; o[4 * j + 3] *= a_hi; }
    }
    uint32_t pa[4][4];  // P as the A operand per k-step st (keys 16 st .. 16 st + 15): n-tiles j = 2 st, 2 st + 1
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float e[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float mm = (c >= 2) ? mn_hi : mn_lo;
        e[c] = (sacc[4 * j + c] == -INFINITY) ? 0.f : exp2f(sacc[4 * j + c] - mm);
      }
      l_lo += e[0] + e[1]; l_hi += e[2] + e[3];
      const int st = j >> 1, hi = j & 1;
      pa[st][hi ? 2 : 0] = pack_f16x2(e[0], e[1]);
      pa[st][hi ? 3 : 1] = pack_f16x2(e[2], e[3]);
    }
    const unsigned char* vt = f16 + bb * F16BUF + HKV * T16 + hh * T16;
    wgmma_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      wgmma_rs<1>(o, pa[st], make_desc(vt + st * 16 * 128, 16, 1024, 1), 1);
    }
    wgmma_commit();  // waited for one iteration later (wait<1>), before its buffer is overwritten
  };

  // --- the tail tile (the block's new tokens) straight into an f16 buffer: 64 keys x 2 heads x 2 (K, V) ----------
  auto build_tail = [&](int bb) {
    // zero the K and V tiles of both heads (32 KB = 256 threads x 128 B), then write the L real keys
    unsigned char* base = f16 + bb * F16BUF;
#pragma unroll
    for (int r = 0; r < 8; ++r) reinterpret_cast<uint4*>(base)[tid + r * 256] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
    if (nt_key < L) {
      const uint16_t* kp = reinterpret_cast<const uint16_t*>(&newk);
      const uint16_t* vp = reinterpret_cast<const uint16_t*>(&newv);
      const uint2 kf = make_uint2(e4m3x2_to_f16x2(kp[0]), e4m3x2_to_f16x2(kp[1]));
      const uint2 kg = make_uint2(e4m3x2_to_f16x2(kp[2]), e4m3x2_to_f16x2(kp[3]));
      const uint2 vf = make_uint2(e4m3x2_to_f16x2(vp[0]), e4m3x2_to_f16x2(vp[1]));
      const uint2 vg = make_uint2(e4m3x2_to_f16x2(vp[2]), e4m3x2_to_f16x2(vp[3]));
      // dims 8 nt_c .. 8 nt_c + 7 of key nt_key: chunk 16-B index (dims / 8) = nt_c, swizzled by key % 8
      const int sw = nt_key & 7;
      unsigned char* kd = base + nt_head * T16 + nt_key * 128 + ((nt_c ^ sw) << 4);
      unsigned char* vd = base + HKV * T16 + nt_head * T16 + nt_key * 128 + ((nt_c ^ sw) << 4);
      *reinterpret_cast<uint4*>(kd) = make_uint4(kf.x, kf.y, kg.x, kg.y);
      *reinterpret_cast<uint4*>(vd) = make_uint4(vf.x, vf.y, vg.x, vg.y);
    }
  };

  // --- main loop: software-pipelined. Invariant at the top of iteration i: tile i's f16 is in buffer i % 2 and its
  //     Q K^T is in flight; raw tile i + 1 is landing. Convert tile i + 1 (after the P V that read its buffer is done),
  //     then softmax(i), issue P V(i) and Q K^T(i + 1) -----------------------------------------------------------------
  const int ntiles = ntiles_c + (tail ? 1 : 0);
  fill(0); cp_async_commit();
  fill(1); cp_async_commit();
  // prologue: tile 0 -> buffer 0, Q K^T(0)
  if (ntiles_c > 0) {
    cp_async_wait<1>();
    __syncthreads();
    convert(raw, f16 + hh * T16);
    convert(raw + TK * RAW_ROW, f16 + HKV * T16 + hh * T16);
  } else if (tail) {
    build_tail(0);
  }
  fence_proxy_async();
  __syncthreads();            // tile 0 complete (and Q's staging buffer is now free to be overwritten by tile 1)
  if (ntiles_c > 0) { fill(2); cp_async_commit(); }
  if (p.mode != 1 && ntiles > 0) qk_issue(0);
  for (int i = 0; i < ntiles; ++i) {
    const int bb = i & 1;
    const bool has_next = (i + 1 < ntiles);
    if (has_next) {
      const bool next_cached = (i + 1 < ntiles_c);
      if (next_cached) cp_async_wait<1>();   // raw tile i + 1 landed (tile i + 2 may be in flight)
      wgmma_wait<1>();                       // P V(i - 1), which read buffer (i + 1) % 2, is done
      __syncthreads();
      if (next_cached) {
        const unsigned char* slot = raw + ((i + 1) % NG) * RAW_SLOT;
        convert(slot, f16 + (bb ^ 1) * F16BUF + hh * T16);
        convert(slot + TK * RAW_ROW, f16 + (bb ^ 1) * F16BUF + HKV * T16 + hh * T16);
        fence_proxy_async();
        __syncthreads();                     // raw slot (i + 1) % NG consumed by everyone
        if (i + 3 <= ntiles_c) { fill(i + 3); cp_async_commit(); }  // (an empty group when i + 3 == ntiles_c keeps the wait<1> accounting)
      } else {
        build_tail(bb ^ 1);
        fence_proxy_async();
        __syncthreads();
      }
    }
    if (p.mode != 1) {
      const int tile_key0 = (i < ntiles_c) ? k0 + TK * i : seq;
      const int nvalid = (i < ntiles_c) ? min(TK, ncached_here - TK * i) : L;
      softmax_pv(bb, tile_key0, nvalid);
      if (has_next) qk_issue(bb ^ 1);
    }
  }
  cp_async_wait<0>();
  wgmma_wait<0>();
  if (p.mode == 1) return;
  l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 1); l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 2);
  l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 1); l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 2);

  // --- partials: o[4j + c]: rows 16 w4 + g (c < 2) / +8, dims 8j + 2t + (c & 1) -------------------------------------
  const size_t pbase = (static_cast<size_t>(b) * p.splits + s) * QROWS;
  const int r_lo = hh * 64 + 16 * w4 + g, r_hi = r_lo + 8;
  float* po_lo = p.part_o + (pbase + r_lo) * 64 + 2 * t;
  float* po_hi = p.part_o + (pbase + r_hi) * 64 + 2 * t;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    *reinterpret_cast<float2*>(po_lo + 8 * j) = make_float2(o[4 * j], o[4 * j + 1]);
    *reinterpret_cast<float2*>(po_hi + 8 * j) = make_float2(o[4 * j + 2], o[4 * j + 3]);
  }
  if (t == 0) {
    p.part_ml[(pbase + r_lo) * 2] = m_lo; p.part_ml[(pbase + r_lo) * 2 + 1] = l_lo;
    p.part_ml[(pbase + r_hi) * 2] = m_hi; p.part_ml[(pbase + r_hi) * 2 + 1] = l_hi;
  }
}

// Merge: one CTA of MW warps per (b, j, kv head, query head): lane -> dims 2 lane, 2 lane + 1; warp w folds splits
// w, w + MW, ...; the warps combine through shared memory and warp 0 applies 1/l, v_scale, the output RMS norm and gate.
__global__ void __launch_bounds__(MW * 32) attn_merge_kernel(const Params p) {
  __shared__ float sm_m[MW], sm_l[MW], sm_o[MW][64];
  const int gq = blockIdx.x & 7, hkv = (blockIdx.x >> 3) % HKV, j = (blockIdx.x / (8 * HKV)) % L, b = blockIdx.x / (8 * HKV * L);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int group = p.Hq / HKV;
  const int row = hkv * 64 + 8 * j + gq;
  const size_t base = static_cast<size_t>(b) * p.splits;
  float mm = -INFINITY, ll = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll 4
  for (int s = warp; s < p.splits; s += MW) {
    const float2 ml = *reinterpret_cast<const float2*>(p.part_ml + ((base + s) * QROWS + row) * 2);
    const float2 v = *reinterpret_cast<const float2*>(p.part_o + ((base + s) * QROWS + row) * 64 + 2 * lane);
    const float mn = fmaxf(mm, ml.x);
    const float a = (mn == -INFINITY) ? 0.f : exp2f(mm - mn), c = (mn == -INFINITY) ? 0.f : exp2f(ml.x - mn);
    mm = mn;
    ll = ll * a + ml.y * c;
    a0 = a0 * a + v.x * c;
    a1 = a1 * a + v.y * c;
  }
  if (lane == 0) { sm_m[warp] = mm; sm_l[warp] = ll; }
  sm_o[warp][2 * lane] = a0; sm_o[warp][2 * lane + 1] = a1;
  __syncthreads();
  if (warp == 0) {
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < MW; ++w) M = fmaxf(M, sm_m[w]);
    float Lsum = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
    for (int w = 0; w < MW; ++w) {
      const float c = (sm_m[w] == -INFINITY) ? 0.f : exp2f(sm_m[w] - M);
      Lsum += sm_l[w] * c;
      o0 += sm_o[w][2 * lane] * c;
      o1 += sm_o[w][2 * lane + 1] * c;
    }
    const float inv = (Lsum > 0.f ? 1.f / Lsum : 0.f) / p.inv_v_scale;
    o0 *= inv; o1 *= inv;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const int qkv_stride = (2 * p.Hq + 2 * HKV) * 64;
    const int head = hkv * group + gq;
    const __nv_bfloat162 gate = *reinterpret_cast<const __nv_bfloat162*>(
        p.qkv + (static_cast<size_t>(b) * L + j) * qkv_stride + (p.Hq + 2 * HKV) * 64 + head * 64 + 2 * lane);
    const float2 gf = __bfloat1622float2(gate);
    const __nv_bfloat162 r = __floats2bfloat162_rn(o0 * rn * (1.f / (1.f + expf(-gf.x))), o1 * rn * (1.f / (1.f + expf(-gf.y))));
    *reinterpret_cast<__nv_bfloat162*>(p.out + (static_cast<size_t>(b) * L + j) * (p.Hq * 64) + head * 64 + 2 * lane) = r;
  }
}

}  // namespace okcattnv
using namespace okcattnv;

void attn_verify(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                 torch::Tensor seq_lens, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor out,
                 int64_t B, int64_t Hq, int64_t Hkv, int64_t page_size, int64_t window, bool rope,
                 double qk_scale, double sm_scale, double k_scale, double v_scale, int64_t splits, int64_t chunk, int64_t mode) {
  TORCH_CHECK(Hkv == HKV && Hq == 8 * HKV, "attn verify: 16 query heads over 2 kv heads");
  TORCH_CHECK(chunk % TK == 0 && chunk <= MAX_CHUNK && chunk > 0, "attn verify: chunk");
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = page_table.data_ptr<int32_t>();
  p.seq_lens = seq_lens.data_ptr<int32_t>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.part_o = part_o.data_ptr<float>();
  p.part_ml = part_ml.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.B = B; p.Hq = Hq; p.page_size = page_size; p.max_pages = page_table.size(1);
  p.window = window; p.splits = splits; p.chunk = chunk; p.rope = rope ? 1 : 0; p.mode = mode;
  p.qk_scale = static_cast<float>(qk_scale);
  p.q_mult = static_cast<float>(sm_scale * k_scale);
  p.inv_k_scale = static_cast<float>(1.0 / k_scale); p.inv_v_scale = static_cast<float>(1.0 / v_scale);
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(attn_verify_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    attr_set = true;
  }
  attn_verify_kernel<<<dim3(splits, B), NW * 32, SMEM_BYTES, stream>>>(p);
  if (mode != 1) attn_merge_kernel<<<B * L * HKV * 8, MW * 32, 0, stream>>>(p);
}
"""

_TEAMV3_ATTN_VERIFY_CPP = r"""
void attn_verify(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                 torch::Tensor seq_lens, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor out,
                 int64_t B, int64_t Hq, int64_t Hkv, int64_t page_size, int64_t window, bool rope,
                 double qk_scale, double sm_scale, double k_scale, double v_scale, int64_t splits, int64_t chunk, int64_t mode);
"""

_TEAMV3__VERIFY_EXT = None
_TEAMV3__INV_FREQ_V = {}
_TEAMV3_VERIFY_MAX_CHUNK = 2048
_TEAMV3_VERIFY_MIN_CHUNK = 128
_TEAMV3_VERIFY_TARGET_CTAS = 264


def _TEAMV3_attn_verify_ext():
    global _TEAMV3__VERIFY_EXT
    if _TEAMV3__VERIFY_EXT is None:
        from torch.utils.cpp_extension import load_inline as _TEAMV3_load_inline
        _TEAMV3__VERIFY_EXT = _TEAMV3_load_inline("okc_attn_verify_v4", cpp_sources=_TEAMV3_ATTN_VERIFY_CPP, cuda_sources=_TEAMV3_ATTN_VERIFY_SRC, functions=["attn_verify"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"], verbose=False)
    return _TEAMV3__VERIFY_EXT


def _TEAMV3_attn_verify_inv_freq(theta: float, device) -> _TEAMV3_torch.Tensor:
    key = (float(theta), str(device))
    if key not in _TEAMV3__INV_FREQ_V:
        if theta:
            _TEAMV3__INV_FREQ_V[key] = (1.0 / (theta ** (_TEAMV3_torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
        else:
            _TEAMV3__INV_FREQ_V[key] = _TEAMV3_torch.zeros(32, dtype=_TEAMV3_torch.float32, device=device)
    return _TEAMV3__INV_FREQ_V[key]


def _TEAMV3_attn_verify_splits(batch: int, max_keys: int, window: int):
    keys = min(max_keys, window) if window else max_keys
    keys = max(keys, 64)
    splits = max(1, -(-_TEAMV3_VERIFY_TARGET_CTAS // batch))
    splits = min(splits, max(1, -(-keys // _TEAMV3_VERIFY_MIN_CHUNK)))
    splits = max(splits, -(-keys // _TEAMV3_VERIFY_MAX_CHUNK))
    chunk = -(-keys // splits)
    chunk = -(-chunk // 64) * 64
    splits = max(1, -(-keys // chunk))
    return splits, chunk


def _TEAMV3_attn_verify_forward(data, max_keys: int, out=None, mode: int = 0, splits_chunk=None):
    """One 8-query verify step from the problem's inputs (sm_90a only)."""
    cfg = data["config"]
    B, L, Hq, Hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert L == 8, "attn_verify_forward: q_len 8"
    qkv = data["qkv"]
    splits, chunk = splits_chunk or _TEAMV3_attn_verify_splits(B, max_keys, cfg["window"])
    dev = qkv.device
    rows = Hkv * 64
    part_o = _TEAMV3_torch.empty((B, splits, rows, 64), dtype=_TEAMV3_torch.float32, device=dev)
    part_ml = _TEAMV3_torch.empty((B, splits, rows, 2), dtype=_TEAMV3_torch.float32, device=dev)
    if out is None:
        out = _TEAMV3_torch.empty((B, L, Hq * 64), dtype=_TEAMV3_torch.bfloat16, device=dev)
    theta = float(cfg["rope_theta"])
    _TEAMV3_attn_verify_ext().attn_verify(qkv.contiguous(), data["cache_loc"], data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"],
                                  _TEAMV3_attn_verify_inv_freq(theta, dev), part_o, part_ml, out, B, Hq, Hkv, cfg["page_size"], cfg["window"], theta != 0.0,
                                  float(cfg["qk_scale_factor"]), float(cfg["sm_scale"]), float(cfg["kv_scale"]), float(cfg["kv_scale"]),
                                  splits, chunk, mode)
    return out
# Attention step dispatch: decode (q_len 1) on the mma.sync split-KV kernel, verify blocks (q_len 8) on the wgmma
# kernel (sm_90a). Both take the problem's inputs and return the gated, normalized output [B, L, Hq * 64] bf16,
# writing the new tokens' cache rows.
import torch as _TEAMV3_torch


def _TEAMV3_attn_step(data, max_keys: int, out=None):
    if data["config"]["q_len"] == 8 and _TEAMV3_torch.cuda.get_device_capability()[0] == 9:
        return _TEAMV3_attn_verify_forward(data, max_keys, out=out)
    return _TEAMV3_attn_decode_forward(data, max_keys, out=out)


def _TEAMV3_kernel_fn(data):
    cfg = data["config"]
    max_keys = data["page_table"].shape[1] * cfg["page_size"]  # shape-derived bound on cache_seqlens
    return _TEAMV3_attn_step(data, max_keys)


def _team_v3_supported(data):
    cfg = data['config']
    if not (cfg['num_q_heads'] == 16 and cfg['num_kv_heads'] == 2 and
            cfg['head_dim'] == 64 and cfg['q_len'] in (1, 8) and cfg['batch'] >= 1 and
            cfg['page_size'] in (1, 64) and cfg['window'] in (0, 2048) and
            cfg['kv_scale'] == 1.0):
        return False
    # Batch-1 sliding decode stays on the base path, like scaled caches (kv_scale != 1).
    if cfg['batch'] == 1 and cfg['q_len'] == 1 and cfg['window'] != 0:
        return False
    if (cfg['window'] == 0 and cfg['rope_theta'] != 0.0) or (
            cfg['window'] != 0 and cfg['rope_theta'] != 500000.0):
        return False
    if cfg['qk_scale_factor'] != 5.4730064863875 or cfg['sm_scale'] != 0.125:
        return False
    qkv, k, v = data['qkv'], data['k_cache'], data['v_cache']
    batch, length, page = cfg['batch'], cfg['q_len'], cfg['page_size']
    if not (qkv.is_cuda and qkv.dtype == _TEAMV3_torch.bfloat16 and
            tuple(qkv.shape) == (batch, length, 2304) and qkv.is_contiguous() and
            qkv.data_ptr() % 16 == 0 and k.is_cuda and v.is_cuda and
            k.ndim == 4 and tuple(k.shape[1:]) == (page, 2, 64) and v.shape == k.shape and
            k.dtype == _TEAMV3_torch.float8_e4m3fn and v.dtype == k.dtype and
            k.is_contiguous() and v.is_contiguous() and
            k.data_ptr() % 16 == 0 and v.data_ptr() % 16 == 0 and
            k.device == v.device == qkv.device):
        return False
    table, sequence, locations = data['page_table'], data['seq_lens'], data['cache_loc']
    if not (table.ndim == 2 and table.shape[0] == batch and table.shape[1] > 0 and
            table.shape[1] * page < (1 << 31) and table.dtype == _TEAMV3_torch.int32 and
            tuple(sequence.shape) == (batch,) and sequence.dtype == _TEAMV3_torch.int32 and
            tuple(locations.shape) == (batch * length,) and locations.dtype == _TEAMV3_torch.int64):
        return False
    for tensor in (table, sequence, locations, data['k_scale'], data['v_scale']):
        if not tensor.is_cuda or tensor.device != qkv.device or not tensor.is_contiguous():
            return False
    if not (data['k_scale'].dtype == _TEAMV3_torch.float32 and data['v_scale'].dtype == _TEAMV3_torch.float32 and
            data['k_scale'].numel() == 1 and data['v_scale'].numel() == 1):
        return False
    return _TEAMV3_torch.cuda.get_device_capability(qkv.device) == (9, 0)


def kernel_fn(data):
    if not _team_v3_supported(data):
        return _TEAM_V3_BASE_KERNEL(data)
    with _TEAMV3_torch.cuda.device(data['qkv'].device):
        return _TEAMV3_kernel_fn(data)


_DECODE_TEAM_COMPOSITION = kernel_fn


def kernel_fn(data):
    cfg = data['config']
    # The CUDA split-KV kernels serve decode (q_len 1) at batch >= 8; verify blocks and smaller batches take the
    # paths above.
    if cfg['q_len'] != 1 or cfg['batch'] < 8:
        return _TEAM_V3_BASE_KERNEL(data)
    return _DECODE_TEAM_COMPOSITION(data)
# load_inline without torch in the CUDA translation unit. torch.utils.cpp_extension.load_inline prepends
# `#include <torch/types.h>` to the .cu (and kernels often add torch/extension.h), and nvcc's host pass over the torch
# headers costs ~8-16 s per extension while the device code of a typical kernel file compiles in ~1 s. This helper takes
# load_inline's arguments, writes the .cu verbatim (the kernels include only CUDA headers and export plain functions)
# and a .cpp with the declarations plus generated pybind bindings, and builds both with cpp_extension.load (ninja
# compiles the two in parallel; torch is parsed once, by the .cpp). When the .cpp would hold only plain declarations
# (int64_t / bool / double / int / std::vector<int64_t> parameters), even that is skipped (LEAN_CT): torch's loader
# builds the .cu alone, with extern "C" trampolines (C++ exceptions -> an error string), into a plain shared library that
# ctypes opens; the returned object has the same method names, raising RuntimeError on errors.
import ctypes
import os
import re

LEAN_CT = True

# A torch-free stand-in for TORCH_CHECK in a lean .cu's host code (skipped when torch is included): it throws
# std::runtime_error, which pybind11 raises as RuntimeError (TORCH_CHECK's c10::Error is a RuntimeError too).
LEAN_CHECK = r"""
#ifndef TORCH_CHECK
#include <stdexcept>
#include <string>
namespace okclean {
inline void app(std::string& s, const char* v) { s += v; }
inline void app(std::string& s, const std::string& v) { s += v; }
template <class T> inline void app(std::string& s, T v) { s += std::to_string(v); }
template <class... A> [[noreturn]] inline void fail(const A&... a) {
  std::string s;
  (app(s, a), ...);
  throw std::runtime_error(s);
}
}  // namespace okclean
#define TORCH_CHECK(cond, ...) do { if (!(cond)) okclean::fail(__VA_ARGS__); } while (0)
#endif
"""


def lean_load_inline(name, cpp_sources, cuda_sources, functions, extra_cflags=None, extra_cuda_cflags=None,
                     extra_ldflags=None, verbose=False):
    from torch.utils.cpp_extension import _get_build_directory, load
    if isinstance(cpp_sources, (list, tuple)):
        cpp_sources = "\n".join(cpp_sources)
    if isinstance(cuda_sources, (list, tuple)):
        cuda_sources = "\n".join(cuda_sources)
    if isinstance(functions, str):
        functions = [functions]
    if LEAN_CT and not extra_cflags and not extra_ldflags:
        sigs = _ct_sigs(cpp_sources, functions)
        if sigs is not None:
            return _ct_load(name, cuda_sources, sigs, extra_cuda_cflags, verbose)
    binds = "\n".join(f'  m.def("{f}", &{f}, "{f}");' for f in functions)
    cpp = ("#include <torch/extension.h>\n" + cpp_sources + "\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n" + binds + "\n}\n")
    build_dir = _get_build_directory(name, verbose)
    os.makedirs(build_dir, exist_ok=True)
    paths = []
    for fname, text in (("main.cpp", cpp), ("cuda.cu", cuda_sources)):
        path = os.path.join(build_dir, fname)
        old = None
        if os.path.exists(path):
            with open(path) as f:
                old = f.read()
        if old != text:  # rewrite only on change (load() rebuilds when a source's hash changes)
            with open(path, "w") as f:
                f.write(text)
        paths.append(path)
    return load(name, paths, extra_cflags=extra_cflags, extra_cuda_cflags=extra_cuda_cflags, extra_ldflags=extra_ldflags,
                build_directory=build_dir, verbose=verbose)


_CT_TYPES = {"int64_t": ctypes.c_int64, "bool": ctypes.c_bool, "double": ctypes.c_double, "int": ctypes.c_int, "void": None}
_CT_VEC = "std::vector<int64_t>"  # (a Python list of ints; passed as pointer + length)
_CT_DECL = re.compile(r"\b(void|int64_t|bool|double|int)\s+(\w+)\s*\(([^()]*)\)\s*;", re.S)


def _ct_sigs(cpp_sources, functions):
    """{name: (ret, [param types])} if cpp_sources is nothing but plain declarations of `functions`, else None."""
    body = re.sub(r"//[^\n]*|#include <\w+>", "", cpp_sources)
    sigs = {}
    for m in _CT_DECL.finditer(body):
        ret, name, params = m.groups()
        types = []
        for prm in (q.strip() for q in params.split(",")):
            if not prm:
                continue
            t = prm.rsplit(None, 1)[0] if len(prm.split()) > 1 else prm
            if (t not in _CT_TYPES or t == "void") and t != _CT_VEC:
                return None
            types.append(t)
        sigs[name] = (ret, types)
    if _CT_DECL.sub("", body).strip() or set(sigs) != set(functions):
        return None
    return sigs


def _ct_trampolines(sigs):
    out = ["", "#include <cstdio>", "#include <exception>", "#include <vector>", "static char okcct_msg[512];"]
    for name, (ret, types) in sigs.items():
        params = [f"const int64_t* a{i}p, int64_t a{i}n" if t == _CT_VEC else f"{t} a{i}" for i, t in enumerate(types)]
        params += [f"{ret}* okc_ret"] if ret != "void" else []
        args = ", ".join(f"std::vector<int64_t>(a{i}p, a{i}p + a{i}n)" if t == _CT_VEC else f"a{i}" for i, t in enumerate(types))
        call = f"*okc_ret = {name}({args});" if ret != "void" else f"{name}({args});"
        out.append(f'extern "C" const char* okcct_{name}({", ".join(params)}) {{\n'
                   f"  try {{ {call} return nullptr; }}\n"
                   f'  catch (const std::exception& e) {{ snprintf(okcct_msg, sizeof(okcct_msg), "%s", e.what()); return okcct_msg; }}\n'
                   f'  catch (...) {{ return "unknown C++ exception"; }}\n}}')
    return "\n".join(out) + "\n"


class _CtExt:
    """The ctypes-loaded library: one method per exported function (same names and arguments as the pybind module)."""

    def __init__(self, lib, sigs, path):
        self._lib, self.__file__ = lib, path
        for name, (ret, types) in sigs.items():
            f = getattr(lib, "okcct_" + name)
            rt = _CT_TYPES[ret]
            argt = []
            for t in types:
                argt += [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64] if t == _CT_VEC else [_CT_TYPES[t]]
            f.argtypes = argt + ([ctypes.POINTER(rt)] if rt else [])
            f.restype = ctypes.c_char_p
            setattr(self, name, _ct_call(f, rt, name, [i for i, t in enumerate(types) if t == _CT_VEC]))


def _ct_call(f, rt, name, vecs):
    def args(a):
        if not vecs:
            return a
        out = []
        for i, v in enumerate(a):
            if i in vecs:
                v = list(v)
                out += [(ctypes.c_int64 * max(1, len(v)))(*v), len(v)]
            else:
                out.append(v)
        return out

    if rt is None:
        def call(*a):
            e = f(*args(a))
            if e is not None:
                raise RuntimeError(e.decode())
    else:
        def call(*a):
            r = rt()
            e = f(*args(a), ctypes.byref(r))
            if e is not None:
                raise RuntimeError(e.decode())
            return r.value
    call.__name__ = name
    return call


def _ct_load(name, cuda_sources, sigs, extra_cuda_cflags, verbose):
    """torch's own build (cpp_extension.load, is_python_module=False: the .cu alone, torch's nvcc flags, no pybind) of the
    .cu + trampolines into a plain shared library, opened with ctypes."""
    from torch.utils.cpp_extension import _get_build_directory, load
    src = cuda_sources + _ct_trampolines(sigs)
    build_dir = _get_build_directory(name, verbose)
    os.makedirs(build_dir, exist_ok=True)
    path = os.path.join(build_dir, "cuda.cu")
    old = None
    if os.path.exists(path):
        with open(path) as f:
            old = f.read()
    if old != src:
        with open(path, "w") as f:
            f.write(src)
    so = load(name, [path], extra_cuda_cflags=extra_cuda_cflags, build_directory=build_dir, verbose=verbose,
              is_python_module=False)
    return _CtExt(ctypes.CDLL(so), sigs, so)
import os

import torch

# Batch-1 decode attention (q_len 1, 16 query heads over 2 kv heads, paged FP8 cache) in ONE launch.
# Numerics follow the split-KV decode kernel above: queries rms-normed * qk_scale (RoPE on sliding layers), rounded to
# bf16, times sm_scale * k_scale in f16; keys/values e4m3 -> f16 (exact); FA2 online softmax in base 2 on mma.m16n8k16
# with the head-dim permutation that lets a lane's 16 bytes of a key row be its B fragment; the new token is
# normalized/quantized once, written to its cache row by the tail CTA and attended from the same bytes; output =
# rmsnorm(O / l * v_scale) * sigmoid(gate).
# What differs (batch 1 is latency-bound, and with a cold L2 every kernel's code is refetched from DRAM each step, so
# the code is kept small: no sincosf slow paths, rolled loops off the hot path, one copy of the tile math):
# - all step inputs (length, the CTA's page ids, the query / new-token / gate rows) are requested first; full layers
#   fetch their page ids without waiting for the length;
# - each warp owns a private cp.async ring of whole 128-B cache rows (both kv heads: a full line per outstanding miss)
#   deep enough to keep its share of the split in flight, with no CTA barrier in the loop; the new token enters the
#   tail CTA's reduce as one more partial (m = q . k_new, l = 1, O = v_new);
# - the split partials are merged in the same launch: within each thread-block cluster every rank pushes its partial
#   of row r into rank (r mod CL)'s shared memory (DSMEM stores, one release/acquire cluster barrier), rank r folds its
#   rows; then per row the last-arriving owner across clusters folds the cluster partials (one ticket per row, reset by
#   that owner, so graph replays are self-contained).
ATTN_B1_SRC = r"""
// (no torch headers in the .cu: they cost ~16 s of host compile; the torch-typed attn_b1() binding is in ATTN_B1_CPP)
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <climits>
#include <cooperative_groups.h>

namespace okcb1 {
namespace cg = cooperative_groups;
constexpr int HKV = 2, NW = 8;
constexpr int ROWB = HKV * 64;                   // one flat cache row (both kv heads): 128 B
constexpr int GS = 16;                           // at most 16 CTAs per cluster and 16 clusters
constexpr int ROWS = 16;                         // 2 kv heads x 8 query heads
constexpr int TILE = 4096;                       // a warp's 16-key tile: K [16][128] then V [16][128] (both kv heads)
constexpr int MAXCHUNK = 2048;                   // keys per CTA (verify kernel; the decode kernel sizes its ids on the host)
constexpr int OSTR = 68;                         // row stride (floats) of the warp partials in smem
constexpr int LMAX = 8;                          // query tokens per step (decode kernel: one token per cluster set)
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ uint64_t l2_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void cp_async16s(uint32_t s, const void* gmem, uint64_t pol) {  // shared address, full 16 B
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "l"(pol));
}
__device__ __forceinline__ void cp_async16sz(uint32_t s, const void* gmem, int src_bytes, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;" ::"r"(s), "l"(gmem), "r"(src_bytes), "l"(pol));
}
__device__ __forceinline__ void cp_async16n(uint32_t s, const void* gmem, int src_bytes) {  // shared address, no policy
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void touch(const void* p) {  // a real load (warms the TLB; prefetches may be dropped on a miss)
  uint32_t v;
  asm volatile("ld.global.relaxed.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p));
}
// the CTA's page ids -> smem: every thread issues its loads of a batch before any store (one DRAM trip per 8 NT ids;
// a load-store loop would wait a trip per NT ids)
template <int NT>
__device__ __forceinline__ void load_ids(int* ids, const int32_t* ptab, int k0, int chunk, int kcap, int shift, int tid) {
  const int pmask = (1 << shift) - 1;
#pragma unroll 1
  for (int base = 0; base < chunk; base += 8 * NT) {
    int v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int i = base + tid + j * NT, key = k0 + i;
      v[j] = (i < chunk && key < kcap) ? (ptab[key >> shift] << shift) | (key & pmask) : 0;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int i = base + tid + j * NT;
      if (i < chunk) ids[i] = v[j];
    }
  }
}
// flag-in-data words for the cross-cluster merge (LL mode): (epoch << 32) | float bits, 8-B single-copy atomic
__device__ __forceinline__ unsigned long long ld_vol_u64(const unsigned long long* a) {
  unsigned long long v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(a));
  return v;
}
__device__ __forceinline__ void ld_vol_v2u64(const unsigned long long* a, unsigned long long& x, unsigned long long& y) {
  asm volatile("ld.volatile.global.v2.u64 {%0, %1}, [%2];" : "=l"(x), "=l"(y) : "l"(a));
}
__device__ __forceinline__ void st_vol_v2u64(unsigned long long* a, unsigned long long x, unsigned long long y) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1, %2};" ::"l"(a), "l"(x), "l"(y) : "memory");
}
__device__ __forceinline__ unsigned long long llw(unsigned long long e, float v) { return e | __float_as_uint(v); }
__device__ __forceinline__ float llf(unsigned long long w) { return __uint_as_float(static_cast<uint32_t>(w)); }
// L2 prefetch of one 128-B row: plain (bit 28) or through the bulk-copy unit (bit 29)
__device__ __forceinline__ void l2pf(const void* p, bool bulk) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (bulk) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" ::"l"(p) : "memory");
    return;
  }
#endif
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
// loads pinned at their position (the compiler may otherwise sink a load to its use, after the cold misses it hides)
__device__ __forceinline__ uint4 vld16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 vld8(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t vld4(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ long long vld64(const void* p) {
  long long v;
  asm volatile("ld.global.nc.s64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
}
// coherent (L2) loads for data a PDL predecessor writes while this grid is already running: the non-coherent path may
// serve lines cached before the producer's writes even after griddepcontrol.wait
__device__ __forceinline__ uint4 cld16(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 cld8(const void* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t cld4(const void* p) {
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int atom_add_acq_rel(int* p, int v) {
  int old;
  asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ int atom_add_relaxed(int* p, int v) {
  int old;
  asm volatile("atom.add.relaxed.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void red_add_release(int* p, int v) {
  asm volatile("red.add.release.gpu.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// PDL (no-ops without the programmatic-serialization launch attribute)
__device__ __forceinline__ void gdc_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
// st.async merge: remote stores that count their bytes on the destination rank's mbarrier (sm_90+)
__device__ __forceinline__ uint32_t sa_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t mapa_u32(uint32_t a, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_async2(uint32_t a, float x, float y, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
               ::"r"(a), "f"(x), "f"(y), "r"(bar) : "memory");
}
__device__ __forceinline__ void st_async4(uint32_t a, float4 v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(bar) : "memory");
}
__device__ __forceinline__ void st_async4u(uint32_t a, uint4 v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t a, int parity) {
  uint32_t done;
  do {
    asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                 : "=r"(done) : "r"(a), "r"(parity) : "memory");
  } while (!done);
}
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v & 0xffffu), __NV_E4M3);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2(float x) {  // 2^x, one MUFU (ex2(-inf) = 0)
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
// (queries are stored times sm_scale * k_scale * log2 e in f16: the MMA scores are already base-2 logits)
// RoPE sin/cos of x = fl(pos * inv_freq) (the reference's fp32 angle): exact reduction mod 2 pi in double, then the
// fast intrinsic on |r| <= pi (abs error 2^-21.4); sincosf would inline its Payne-Hanek slow path at every call site.
__device__ __forceinline__ void rope_sincos(float x, float* s, float* c) {
  const double xd = static_cast<double>(x);
  const double k = rint(xd * 0.15915494309189535);
  __sincosf(static_cast<float>(fma(-k, 6.283185307179586, xd)), s, c);
}
// rms-normalize (64 dims over the 16 lanes of a half-warp, 4 dims each) and rotate one 4-dim piece (dims 4 part ..)
__device__ __forceinline__ void norm_rope4(float v[4], float scale, int rope, float pos, float2 ifr) {
  float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f) * scale;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= inv;
  if (rope) {
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
      float sn, cs;
      rope_sincos(pos * (i ? ifr.y : ifr.x), &sn, &cs);
      const float e = v[i], o = v[i + 1];
      v[i] = e * cs - o * sn;
      v[i + 1] = e * sn + o * cs;
    }
  }
}
__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}

struct P {
  const __nv_bfloat16* qkv;   // [1, 1, (2 Hq + 2 Hkv) * 64]
  const int64_t* cache_loc;   // [1]
  uint8_t* k_cache;
  uint8_t* v_cache;
  const int32_t* page_table;  // [1, max_pages]
  const int32_t* seq_lens;    // [1]
  const float* inv_freq;      // [32]
  float* grp_ml;              // [clusters][16][2]
  float* grp_o;               // [clusters][16][64]
  int* cnt;                   // [16] per-row tickets across clusters, zero between launches
  __nv_bfloat16* out;         // [1, 1, Hq * 64]
  unsigned long long* dbg;    // probe builds: [S][16] phase timestamps (globaltimer ns, then SM clock)
  long long cache_bytes, table_bytes;  // TLB warm-up extents (0: off)
  int mode;                   // probe builds: ablation mask (1 no query/new-token math, 2 no K/V loads, 4 no tile math)
  int page_shift, max_pages, window, S, chunk, rope, Hq;
  int B;                      // requests
  int L;                      // query tokens per step
  int Stok;                   // b1_decode_kernel: CTAs per token (grid = L x Stok)
  int pdl;                    // launched with programmatic stream serialization (qkv read after griddepcontrol.wait)
  float qk_scale, q_mult, inv_k_scale, inv_v_scale;
};

#ifdef OKC_B1_PROBE
__device__ __forceinline__ void stamp(const P& p, int i) {
  if (p.dbg != nullptr && threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.dbg[blockIdx.x * 16 + i] = t;
    p.dbg[blockIdx.x * 16 + 8 + i] = clock64();
  }
}
__device__ __forceinline__ void tstamp(const P& p, int stage, int slot) {  // per-stage trace of CTA 0 (mode 512)
  if ((p.mode & 512) && p.dbg != nullptr && blockIdx.x == 0 && stage >= 0 && stage < 256) {
    p.dbg[(slot >= 12 ? 60000 + stage : 50000 + stage * 12 + slot)] = clock64();
  }
}
__device__ __forceinline__ void stamp_t(const P& p, int i, int who) {  // stamp from thread `who`
  if (p.dbg != nullptr && threadIdx.x == who) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.dbg[blockIdx.x * 16 + i] = t;
    p.dbg[blockIdx.x * 16 + 8 + i] = clock64();
  }
}
__device__ __forceinline__ void stampx(const P& p, int i) {  // extra decode-kernel phase stamps (globaltimer)
  if (p.dbg != nullptr && threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.dbg[49152 + blockIdx.x * 16 + i] = t;
  }
}
#define PROBE_MODE(bit) (p.mode & (bit))
#else
__device__ __forceinline__ void stamp(const P&, int) {}
__device__ __forceinline__ void stampx(const P&, int) {}
__device__ __forceinline__ void stamp_t(const P&, int, int) {}
__device__ __forceinline__ void tstamp(const P&, int, int) {}
#define PROBE_MODE(bit) 0
#endif

// One 16-key tile for row g of one kv head: kr = this lane's K bytes of keys g, 8 + g at dims 16 t .. 16 t + 15;
// vr = its V bytes of keys 2t, 2t+1, 2t+8, 2t+9 at dims 8 g .. 8 g + 7. Keys >= nvalid are masked.
struct Acc {
  float o[8][2];
  float m, l;
};
__device__ __forceinline__ void consume(Acc& A, const uint32_t qa[8], const uint4 kr[2], const uint2 vr[4], int nvalid) {
  const int t = threadIdx.x & 3;
  float sacc[2][4];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const uint32_t kw[4] = {kr[u].x, kr[u].y, kr[u].z, kr[u].w};
    sacc[u][0] = sacc[u][1] = sacc[u][2] = sacc[u][3] = 0.f;
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint32_t a[4] = {qa[2 * st], 0u, qa[2 * st + 1], 0u};
      mma_f16(sacc[u], a, e4m3x2_to_f16x2(kw[st]), e4m3x2_to_f16x2(kw[st] >> 16));
    }
  }
  float tmax = -INFINITY;
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float v = (8 * u + 2 * t + c) < nvalid ? sacc[u][c] : -INFINITY;  // q carries log2 e
      sacc[u][c] = v;
      tmax = fmaxf(tmax, v);
    }
  tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
  tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
  const float mn = fmaxf(A.m, tmax);
  const float mref = (mn == -INFINITY) ? 0.f : mn;  // a row with no valid key yet: every term below is ex2(-inf) = 0
  const float a = ex2(A.m - mref);
  A.m = mn;
  A.l *= a;
  // (unconditional: a warp vote + branch here would split the two kv heads' chains into basic blocks)
#pragma unroll
  for (int j = 0; j < 8; ++j) { A.o[j][0] *= a; A.o[j][1] *= a; }
  float pv[2][2];
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float e = ex2(sacc[u][c] - mref);
      pv[u][c] = e;
      A.l += e;
    }
  const uint32_t pa[4] = {pack_f16x2(pv[0][0], pv[0][1]), 0u, pack_f16x2(pv[1][0], pv[1][1]), 0u};
  // V: n-tile j = dims 8 g + j; B rows 2t, 2t+1 (vr 0, 1) and 2t+8, 2t+9 (vr 2, 3)
  const uint32_t vw[4][2] = {{vr[0].x, vr[0].y}, {vr[1].x, vr[1].y}, {vr[2].x, vr[2].y}, {vr[3].x, vr[3].y}};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t sel = (j & 3) | ((4 + (j & 3)) << 4);  // byte j of each key's 8 bytes -> (key 2t(+8), key 2t+1(+9))
    const uint32_t b0 = e4m3x2_to_f16x2(__byte_perm(vw[0][j >> 2], vw[1][j >> 2], sel));
    const uint32_t b1 = e4m3x2_to_f16x2(__byte_perm(vw[2][j >> 2], vw[3][j >> 2], sel));
    float c4[4] = {A.o[j][0], A.o[j][1], 0.f, 0.f};
    mma_f16(c4, pa, b0, b1);
    A.o[j][0] = c4[0]; A.o[j][1] = c4[1];
  }
}

// transposed (B1_DT): thread (g, t) owns query heads 2t, 2t+1: S^T keys g, g+8; O^T dims 8g + 2mt (+1), mt = 0..3
struct AccT {
  float o[4][4];     // m-tile mt: (dim 8g+2mt, head 2t), (8g+2mt, 2t+1), (8g+2mt+1, 2t), (8g+2mt+1, 2t+1)
  float m[2], l[2];  // heads 2t, 2t+1 (l: this thread's keys; reduced over g at the end)
};
__device__ __forceinline__ void consume_t(AccT& A, const uint32_t qa[8], const uint4 kr[2], const uint2 vr[4], int nvalid) {
  const int g = (threadIdx.x & 31) >> 2;
  const uint32_t k0[4] = {kr[0].x, kr[0].y, kr[0].z, kr[0].w}, k1[4] = {kr[1].x, kr[1].y, kr[1].z, kr[1].w};
  float st[4] = {0.f, 0.f, 0.f, 0.f};  // (key g, head 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1); q carries log2 e
#pragma unroll
  for (int s4 = 0; s4 < 4; ++s4) {
    const uint32_t a[4] = {e4m3x2_to_f16x2(k0[s4]), e4m3x2_to_f16x2(k1[s4]), e4m3x2_to_f16x2(k0[s4] >> 16), e4m3x2_to_f16x2(k1[s4] >> 16)};
    mma_f16(st, a, qa[2 * s4], qa[2 * s4 + 1]);
  }
  const bool v0 = g < nvalid, v1 = g + 8 < nvalid;
  const float s0 = v0 ? st[0] : -INFINITY, s1 = v0 ? st[1] : -INFINITY, s2 = v1 ? st[2] : -INFINITY, s3 = v1 ? st[3] : -INFINITY;
  float tm0 = fmaxf(s0, s2), tm1 = fmaxf(s1, s3);
#pragma unroll
  for (int o2 = 4; o2 < 32; o2 <<= 1) {
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, o2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, o2));
  }
  const float mn0 = fmaxf(A.m[0], tm0), mn1 = fmaxf(A.m[1], tm1);
  const float r0 = (mn0 == -INFINITY) ? 0.f : mn0, r1 = (mn1 == -INFINITY) ? 0.f : mn1;
  const float a0 = ex2(A.m[0] - r0), a1 = ex2(A.m[1] - r1);
  A.m[0] = mn0; A.m[1] = mn1;
  const float e0 = ex2(s0 - r0), e1 = ex2(s1 - r1), e2 = ex2(s2 - r0), e3 = ex2(s3 - r1);
  A.l[0] = A.l[0] * a0 + (e0 + e2);
  A.l[1] = A.l[1] * a1 + (e1 + e3);
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) { A.o[mt][0] *= a0; A.o[mt][1] *= a1; A.o[mt][2] *= a0; A.o[mt][3] *= a1; }
  uint32_t pb0, pb1;  // P^T B fragments: (head g; keys 2t, 2t+1) / (head g; keys 2t+8, 2t+9)
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(pb0) : "r"(pack_f16x2(e0, e1)));
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(pb1) : "r"(pack_f16x2(e2, e3)));
  const uint32_t vw[4][2] = {{vr[0].x, vr[0].y}, {vr[1].x, vr[1].y}, {vr[2].x, vr[2].y}, {vr[3].x, vr[3].y}};
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {  // V^T m-tile mt: rows g / g+8 = dims 8g + 2mt / + 1 (bytes 2mt, 2mt+1 of the 8)
    const int j0 = 2 * mt, j1 = 2 * mt + 1;
    const uint32_t q0 = (j0 & 3) | ((4 + (j0 & 3)) << 4), q1 = (j1 & 3) | ((4 + (j1 & 3)) << 4);
    const uint32_t a[4] = {e4m3x2_to_f16x2(__byte_perm(vw[0][j0 >> 2], vw[1][j0 >> 2], q0)), e4m3x2_to_f16x2(__byte_perm(vw[0][j1 >> 2], vw[1][j1 >> 2], q1)),
                           e4m3x2_to_f16x2(__byte_perm(vw[2][j0 >> 2], vw[3][j0 >> 2], q0)), e4m3x2_to_f16x2(__byte_perm(vw[2][j1 >> 2], vw[3][j1 >> 2], q1))};
    mma_f16(A.o[mt], a, pb0, pb1);
  }
}

// online fold of one (m, l, O[4]) into a running one
__device__ __forceinline__ void fold(float& M, float& L, float4& acc, float m, float l, float4 o) {
  if (m == -INFINITY) return;
  const float mn = fmaxf(M, m);
  const float a = (M == -INFINITY) ? 0.f : ex2(M - mn), b = ex2(m - mn);
  M = mn;
  L = L * a + l * b;
  acc.x = acc.x * a + o.x * b; acc.y = acc.y * a + o.y * b; acc.z = acc.z * a + o.z * b; acc.w = acc.w * a + o.w * b;
}

// online fold of one (m, l, O[16]) into a running one
__device__ __forceinline__ void fold16(float& M, float& L, float4 acc[4], float m, float l, const float4 o[4]) {
  if (m == -INFINITY) return;
  const float mn = fmaxf(M, m);
  const float a = (M == -INFINITY) ? 0.f : ex2(M - mn), b = ex2(m - mn);
  M = mn;
  L = L * a + l * b;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc[i].x = acc[i].x * a + o[i].x * b; acc[i].y = acc[i].y * a + o[i].y * b;
    acc[i].z = acc[i].z * a + o[i].z * b; acc[i].w = acc[i].w * a + o[i].w * b;
  }
}

// fold the (m, l, O[4]) held by n consecutive lanes (n a power of two <= 16): every lane gets the result
__device__ __forceinline__ void group_fold(int n, float& M, float& L, float4& acc) {
  float mx = M;
#pragma unroll
  for (int o = 1; o < 16; o <<= 1)
    if (o < n) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const float c = (M == -INFINITY) ? 0.f : ex2(M - mx);
  float l = L * c;
  float4 a = make_float4(acc.x * c, acc.y * c, acc.z * c, acc.w * c);
#pragma unroll
  for (int o = 1; o < 16; o <<= 1)
    if (o < n) {
      l += __shfl_xor_sync(0xffffffffu, l, o);
      a.x += __shfl_xor_sync(0xffffffffu, a.x, o); a.y += __shfl_xor_sync(0xffffffffu, a.y, o);
      a.z += __shfl_xor_sync(0xffffffffu, a.z, o); a.w += __shfl_xor_sync(0xffffffffu, a.w, o);
    }
  M = mx; L = l; acc = a;
}

// smem: qs [16][64] f16 | newt: the new tokens as one ring tile (keys 0..LMAX-1, the rest zero) | recv (the cluster ranks' partials of this rank's rows:
// [CL][16 / CL] (m, l), then [CL][16 / CL][64] O) | per-warp rings [NW][RD][TILE] | ids [chunk]. After the loop the rings hold the
// warp partials (m, l [NW][16], O [NW][16][OSTR]); the cross-cluster fold scratch reuses that area.
constexpr int RECV_BYTES = ROWS * 66 * 4;
template <int RD>
__global__ void __launch_bounds__(NW * 32, 1) b1_decode_kernel(const P p) {
  static_assert(RD * TILE >= (2 * ROWS + ROWS * OSTR) * 4 && NW * RD * TILE >= 256 * 6 * 4, "ring");
  extern __shared__ __align__(16) unsigned char smem[];
  __half* qs = reinterpret_cast<__half*>(smem);
  unsigned char* newt = smem + ROWS * 64 * 2;  // [TILE]: K [16][128] then V [16][128], the ring's swizzle
  float* recv = reinterpret_cast<float*>(newt + TILE);
  unsigned char* ring = newt + TILE + RECV_BYTES;
  int* ids = reinterpret_cast<int*>(ring + NW * RD * TILE);  // [chunk] (sized by the host)
  __shared__ int s_last[ROWS];
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  // q_len L: token tok's Stok CTAs (its own clusters) treat it as a decode step over its own window; the token's new
  // keys are the block's tokens 0..tok (causal)
  // batch: problem = (request rb, token tok), each with its own Stok CTAs
  const int prob = blockIdx.x / p.Stok, s = blockIdx.x - prob * p.Stok;
  const int rb = prob / p.L, tok = prob - rb * p.L;
  const int QS = (2 * p.Hq + 2 * HKV) * 64;  // qkv row stride (per token)
  const __nv_bfloat16* qkv = p.qkv + prob * QS;
  const int32_t* ptab = p.page_table + static_cast<size_t>(rb) * p.max_pages;
  const bool tailcta = s == p.Stok - 1;
  const int nidx = NW - 1 - warp;  // new token handled by this warp of the tail CTA (last warps: fewest tiles)
  const bool newtok = tailcta && nidx <= tok;
  // st.async merge (mode bit 18): this rank's receive mbarrier expects every source's bytes (one phase per launch)
  // (in the dynamic area after the ids: a static one shifts the dynamic base off its alignment, and the K/V ring and
  //  the ids lose their bank / sector alignment)
  uint64_t& s_rbar = *reinterpret_cast<uint64_t*>(reinterpret_cast<unsigned char*>(ids) + ((p.chunk * 4 + 15) & ~15));
  const bool samode = (p.mode & (1 << 18)) != 0;
  if (samode && threadIdx.x == 0) {
    const int ncb = static_cast<int>(cg::this_cluster().num_blocks());
    const uint32_t ba = sa_u32(&s_rbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(ba) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(ba), "r"(ncb * (ROWS / ncb) * 264) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_arrive_relaxed();  // "this CTA has started" (DSMEM stores into it are allowed once every rank has arrived)
  stamp(p, 0);

  // --- first touches that do not depend on the previous kernel (PDL: before griddepcontrol.wait): length, TLB warm-up,
  //     RoPE frequencies, the new token's cache row; then the page ids and the first K/V copies
  const int seq = p.seq_lens[rb];
  if (p.mode & (1 << 10)) {  // batch-1 sliding: the page-table row -> L2 while the length loads (B1_PTPF)
    const int nct = p.B > 1 ? p.Stok : static_cast<int>(gridDim.x), ci = p.B > 1 ? s : static_cast<int>(blockIdx.x);
    const int rowb = p.max_pages * 4, per = (rowb + nct - 1) / nct;  // batch: this request's CTAs split its row
    const int beg = ci * per, end = min(rowb, beg + per);
    for (int off = beg + 128 * tid; off < end; off += 128 * NW * 32) l2pf(reinterpret_cast<const unsigned char*>(ptab) + off, false);
  }
  if (p.cache_bytes > 0 && tid < 96) {
    const int which = tid >> 5;
    const long long off = static_cast<long long>(tid & 31) << 21;
    const long long lim = which == 2 ? p.table_bytes : p.cache_bytes;
    const uint8_t* base = which == 0 ? p.k_cache : which == 1 ? p.v_cache : reinterpret_cast<const uint8_t*>(p.page_table);
    if (off < lim) touch(base + off);
  }
  const int part = tid & 15;  // 4-dim piece of a 64-dim row (query prep, new token, output)
  const uint2 ifr_raw = vld8(p.inv_freq + 2 * part);
  const float2 ifr = make_float2(__uint_as_float(ifr_raw.x), __uint_as_float(ifr_raw.y));
  // output slot (see the merge): cluster rank r owns rows r, r + CL, ..; output thread tid -> (row orow, piece q);
  // the stored dims 4q .. 4q + 3 are natural d0, d0 + 8, d0 + 1, d0 + 9 (d0 = 16 (q >> 2) + 2 (q & 3))
  const int CLs = static_cast<int>(cluster.num_blocks()), crank = static_cast<int>(cluster.block_rank());
  const int q = part, orow = crank + CLs * (tid >> 4);  // output threads: tid < 16 x rows per rank
  const int mrow = orow < ROWS ? orow : crank;
  const int d0 = 16 * (q >> 2) + 2 * (q & 3);
  const int qkv_gate = (p.Hq + 2 * HKV) * 64;
  long long loc = 0;
  if (newtok && nidx == tok) loc = vld64(p.cache_loc + prob);
  // the step's qkv rows: without PDL (standalone) requested here, ahead of the K/V burst they would queue behind;
  // with PDL they come from the previous kernel and are read after griddepcontrol.wait below
  uint2 qin = make_uint2(0, 0), kin = qin, vin = qin;
  uint32_t gate0 = 0, gate1 = 0;
  auto load_qkv = [&]() {
    // coherent loads: under PDL this kernel has already issued work, and .nc reads of the producer's output may be stale
    qin = cld8(qkv + (tid >> 4) * 64 + 4 * part);  // query head tid >> 4
    gate0 = cld4(qkv + qkv_gate + mrow * 64 + d0);
    gate1 = cld4(qkv + qkv_gate + mrow * 64 + d0 + 8);
    if (newtok) {  // new token nidx: lane -> (kv head lane >> 4, piece)
      const __nv_bfloat16* nrow = p.qkv + (rb * p.L + nidx) * QS;
      kin = cld8(nrow + p.Hq * 64 + (lane >> 4) * 64 + 4 * part);
      vin = cld8(nrow + p.Hq * 64 + HKV * 64 + (lane >> 4) * 64 + 4 * part);
    }
  };
  if (!p.pdl) load_qkv();

  // --- the CTA's page ids -> smem (full layers: without waiting for the length)
  const int kcap = p.max_pages << p.page_shift, pmask = (1 << p.page_shift) - 1;
  int k0 = s * p.chunk;
  if (p.window) k0 += max(0, seq + tok - p.window + 1);  // a branch, not a select: full layers do not wait for the length
  if (PROBE_MODE(1 << 30)) {  // (probe mode bit 30: the serial loop, one DRAM trip per NW * 32 ids)
#pragma unroll 1
    for (int i = tid; i < p.chunk; i += NW * 32) {
      const int key = k0 + i;
      ids[i] = key < kcap ? (ptab[key >> p.page_shift] << p.page_shift) | (key & pmask) : 0;
    }
  } else {
    load_ids<NW * 32>(ids, ptab, k0, p.chunk, kcap, p.page_shift, tid);
  }
  const int k1 = min(seq, k0 + p.chunk);
  const int ncached = max(0, k1 - k0), ntiles = (ncached + 15) >> 4;
  const int nmine = ntiles > warp ? (ntiles - warp + NW - 1) / NW : 0;  // this warp's tiles: warp, warp + 8, ...

  stampx(p, 0);  // thread 0's page ids landed
  if (tailcta) reinterpret_cast<uint4*>(newt)[tid] = make_uint4(0u, 0u, 0u, 0u);  // keys past the block's tokens
  __syncthreads();  // ids are visible
  stamp(p, 1);

  // --- this warp's ring: its k-th tile (16 whole 128-B rows of K, then of V) -> slot k % RD; 16-byte chunks swizzled
  //     (K: chunk ^ 4 (row & 1); V: chunk ^ 2 ((row >> 1) & 3)) so the fragment reads below are conflict-free
  unsigned char* myring = ring + warp * RD * TILE;
  auto issue = [&](int k) {
    if (k < nmine && !PROBE_MODE(2)) {
      // lane -> keys q4 + 4 m (m < 4), chunk ch of each K and V row: the 4 page ids at once, one row offset per key
      // for both copies; swizzles (K: chunk ^ 4 (key & 1) = ^ 4 (q4 & 1); V: chunk ^ 2 ((key >> 1) & 3), which
      // alternates between two values with m)
      const int base = 16 * (warp + NW * k), q4 = lane >> 3, ch = lane & 7;
      const uint32_t sb = sa_u32(myring + (k % RD) * TILE) + q4 * ROWB;
      const uint32_t kx = (ch ^ ((q4 & 1) << 2)) << 4;
      const uint32_t vx0 = TILE / 2 + ((ch ^ (((q4 >> 1) & 3) << 1)) << 4), vx1 = TILE / 2 + ((ch ^ ((((q4 >> 1) + 2) & 3) << 1)) << 4);
      int rw[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) rw[m] = base + q4 + 4 * m < ncached ? ids[base + q4 + 4 * m] : -1;
      const unsigned char* kc = p.k_cache + 16 * ch;
      const unsigned char* vc = p.v_cache + 16 * ch;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const size_t off = rw[m] >= 0 ? static_cast<size_t>(rw[m]) * ROWB : 0;
        const int nb = rw[m] >= 0 ? 16 : 0;
        cp_async16n(sb + 4 * m * ROWB + kx, kc + off, nb);
        cp_async16n(sb + 4 * m * ROWB + ((m & 1) ? vx1 : vx0), vc + off, nb);
      }
    }
    cp_async_commit();
  };
#pragma unroll 1
  for (int k = 0; k < RD; ++k) issue(k);
  if ((p.mode & (3 << 28)) && nmine > RD && !PROBE_MODE(2)) {
    // the warp's later tiles -> L2 now (lane: tensor lane >> 4, key lane & 15), so their ring copies hit L2: memory-level
    // parallelism beyond the ring
    const bool bulk = (p.mode & (1 << 29)) != 0;
    const unsigned char* pool = (lane >> 4) ? p.v_cache : p.k_cache;
#pragma unroll 1
    for (int k = RD; k < nmine; ++k) {
      const int idx = 16 * (warp + NW * k) + (lane & 15);
      if (idx < ncached) l2pf(pool + static_cast<size_t>(ids[idx]) * ROWB, bulk);
    }
  }
  stamp(p, 2);
  // --- PDL: the dependent kernel may start (its weight prefetch overlaps this kernel); the step's qkv comes from the
  //     previous kernel: wait for it before the first read
  gdc_launch();
  gdc_wait();
  stampx(p, 1);
  if (p.pdl) load_qkv();
  // LL cross-cluster merge: this launch's epoch (read after the wait: every earlier launch has finished)
  const bool llmode = (p.mode & (1 << 27)) != 0 && p.Stok > static_cast<int>(cluster.num_blocks());
  unsigned long long* const llbuf = reinterpret_cast<unsigned long long*>(p.grp_o);
  const unsigned long long gen = llmode ? ld_vol_u64(llbuf + prob) : 0ull;  // per problem: its merger advances it
  // --- query rows -> smem: rms * qk_scale (RoPE on sliding) -> bf16 -> * sm_scale * k_scale -> f16 (row tid >> 4)
  {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qin.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qin.y));
    float v[4] = {a.x, a.y, b.x, b.y};
    if (!PROBE_MODE(1)) norm_rope4(v, p.qk_scale, p.rope, static_cast<float>(seq + tok), ifr);
    *reinterpret_cast<uint2*>(qs + (tid >> 4) * 64 + 4 * part) =
        make_uint2(pack_f16x2(bf16_round(v[0]) * p.q_mult, bf16_round(v[1]) * p.q_mult),
                   pack_f16x2(bf16_round(v[2]) * p.q_mult, bf16_round(v[3]) * p.q_mult));
  }
  // --- the new tokens 0..tok (tail CTA, warp NW-1-i takes token i, while its copies fly): key normalized (RoPE at
  //     seq + i on sliding) and quantized, value quantized; token tok's row is written to the cache here (single
  //     writer; no CTA reads it); all go to the new tile (key i), which a warp of the tail CTA consumes like a ring tile
  if (newtok) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kin.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kin.y));
    float k4[4] = {a.x, a.y, b.x, b.y};
    if (!PROBE_MODE(1)) norm_rope4(k4, 1.f, p.rope, static_cast<float>(seq + nidx), ifr);
    const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vin.x));
    const float2 d = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vin.y));
    const float ik = p.inv_k_scale, iv = p.inv_v_scale;
    const uint32_t kq = fp8x4(k4[0] * ik, k4[1] * ik, k4[2] * ik, k4[3] * ik);
    const uint32_t vq = fp8x4(c.x * iv, c.y * iv, d.x * iv, d.y * iv);
    const int off = (lane >> 4) * 64 + 4 * part;  // byte offset in the row
    if (nidx == tok) {
      *reinterpret_cast<uint32_t*>(p.k_cache + static_cast<size_t>(loc) * ROWB + off) = kq;
      *reinterpret_cast<uint32_t*>(p.v_cache + static_cast<size_t>(loc) * ROWB + off) = vq;
    }
    const int ch = off >> 4, wi = off & 15;  // 16-B chunk of the row, byte in it (the ring's swizzles)
    *reinterpret_cast<uint32_t*>(newt + nidx * ROWB + ((ch ^ ((nidx & 1) << 2)) << 4) + wi) = kq;
    *reinterpret_cast<uint32_t*>(newt + TILE / 2 + nidx * ROWB + ((ch ^ (((nidx >> 1) & 3) << 1)) << 4) + wi) = vq;
  }

  stampx(p, 2);  // thread 0's query prep done
  // the output gates' sigmoids now, off the epilogue's dependency chain
  float sgate[4];
  {
    const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate0));
    const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate1));
    sgate[0] = 1.f / (1.f + expf(-ga.x)); sgate[1] = 1.f / (1.f + expf(-ga.y));
    sgate[2] = 1.f / (1.f + expf(-gb.x)); sgate[3] = 1.f / (1.f + expf(-gb.y));
  }
  __syncthreads();  // query rows (and new-token rows) are visible
  stampx(p, 3);
  uint32_t qa[HKV][8];  // row g of each kv head's 8 query heads, dims 16 t .. 16 t + 15
#pragma unroll
  for (int h = 0; h < HKV; ++h) {
    const uint4 x = *reinterpret_cast<const uint4*>(qs + (h * 8 + g) * 64 + 16 * t);
    const uint4 y = *reinterpret_cast<const uint4*>(qs + (h * 8 + g) * 64 + 16 * t + 8);
    qa[h][0] = x.x; qa[h][1] = x.y; qa[h][2] = x.z; qa[h][3] = x.w; qa[h][4] = y.x; qa[h][5] = y.y; qa[h][6] = y.z; qa[h][7] = y.w;
  }
#ifdef B1_DT
  AccT A[HKV];
#pragma unroll
  for (int h = 0; h < HKV; ++h) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) A[h].o[mt][0] = A[h].o[mt][1] = A[h].o[mt][2] = A[h].o[mt][3] = 0.f;
    A[h].m[0] = A[h].m[1] = -INFINITY; A[h].l[0] = A[h].l[1] = 0.f;
  }
#define B1_CONSUME consume_t
#else
  Acc A[HKV];
#pragma unroll
  for (int h = 0; h < HKV; ++h) {
#pragma unroll
    for (int j = 0; j < 8; ++j) A[h].o[j][0] = A[h].o[j][1] = 0.f;
    A[h].m = -INFINITY; A[h].l = 0.f;
  }
#define B1_CONSUME consume
#endif

  // --- main loop: wait for tile k (this lane's copies, then the warp's), read both heads' fragments, refill the slot
  auto frags_at = [&](const unsigned char* slot, uint4 (&kr)[HKV][2], uint2 (&vr)[HKV][4]) {
#pragma unroll
    for (int h = 0; h < HKV; ++h) {
#pragma unroll
      for (int u = 0; u < 2; ++u) kr[h][u] = *reinterpret_cast<const uint4*>(slot + (8 * u + g) * ROWB + (((4 * h + t) ^ ((g & 1) << 2)) << 4));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int key = (i < 2) ? 2 * t + i : 2 * t + 8 + (i - 2);
        const int ch = (4 * h + (g >> 1)) ^ (((key >> 1) & 3) << 1);
        vr[h][i] = *reinterpret_cast<const uint2*>(slot + TILE / 2 + key * ROWB + (ch << 4) + (g & 1) * 8);
      }
    }
  };
  auto frags = [&](int k, uint4 (&kr)[HKV][2], uint2 (&vr)[HKV][4]) { frags_at(myring + (k % RD) * TILE, kr, vr); };
  // the block's new tokens 0..tok go into the free slots of the tail CTA's last cached tile when they fit (sliding:
  // the window's cached keys and the block's tokens fill whole tiles), else they are one more tile (below)
  const int r0n = ncached & 15, lt = ntiles - 1;  // keys in the last cached tile (0: full) and its index
  const bool mergenew = tailcta && r0n != 0 && r0n + tok + 1 <= 16;
  if (tailcta && warp == NW - 1 && !mergenew) {
    // the block's new tokens 0..tok as one more tile (this warp has the fewest ring tiles; its first copies are in flight)
    uint4 kr[HKV][2];
    uint2 vr[HKV][4];
    frags_at(newt, kr, vr);
    if (!PROBE_MODE(4)) {
#pragma unroll
      for (int h = 0; h < HKV; ++h) B1_CONSUME(A[h], qa[h], kr[h], vr[h], tok + 1);
    }
  }
  {
#pragma unroll 1
  for (int k = 0; k < nmine; ++k) {
    cp_async_wait<RD - 1>();
    if (k == 0) stampx(p, 4);  // warp 0's first tile landed
    __syncwarp();
    const bool here = mergenew && warp + NW * k == lt;
    if (here) {
      // new token i -> row r0n + i of this tile (K and V rows, 16-B chunks re-swizzled for their new row)
      unsigned char* slot = myring + (k % RD) * TILE;
      for (int e = lane; e < 16 * (tok + 1); e += 32) {
        const int i = e >> 4, c = e & 7, isv = (e >> 3) & 1, r = r0n + i, base = isv ? TILE / 2 : 0;
        const int dsw = isv ? (((r >> 1) & 3) << 1) : ((r & 1) << 2), ssw = isv ? (((i >> 1) & 3) << 1) : ((i & 1) << 2);
        *reinterpret_cast<uint4*>(slot + base + r * ROWB + ((c ^ dsw) << 4)) = *reinterpret_cast<const uint4*>(newt + base + i * ROWB + ((c ^ ssw) << 4));
      }
      __syncwarp();
    }
    uint4 kr[HKV][2];
    uint2 vr[HKV][4];
    frags(k, kr, vr);
    __syncwarp();
    issue(k + RD);
    const int nvalid = here ? r0n + tok + 1 : min(16, ncached - 16 * (warp + NW * k));
    if (!PROBE_MODE(4)) {
#pragma unroll
      for (int h = 0; h < HKV; ++h) B1_CONSUME(A[h], qa[h], kr[h], vr[h], nvalid);
    }
  }
  }
  cp_async_wait<0>();
  stamp(p, 3);

  // --- CTA partial: each warp writes its partial into its own ring region as soon as its copies landed (no CTA barrier
  //     before: the region is private) - [m 16][l 16][o 16][OSTR] in the stored dim order (t * 8 + j) * 2 + c - then
  //     thread (row tid >> 4, piece q) folds the 8 warps for stored dims 4q .. 4q + 3
  __syncwarp();
  stampx(p, 5);
  auto wpart = [&](int w) { return reinterpret_cast<float*>(ring + w * RD * TILE); };
  {
    float* mine = wpart(warp);
#ifdef B1_DT
#pragma unroll
    for (int h = 0; h < HKV; ++h) {  // the same partial layout: row h*8 + head, stored dim ((d/8)/2 * 8 + d%8) * 2 + (d/8)%2
#pragma unroll
      for (int o2 = 4; o2 < 32; o2 <<= 1) {
        A[h].l[0] += __shfl_xor_sync(0xffffffffu, A[h].l[0], o2);
        A[h].l[1] += __shfl_xor_sync(0xffffffffu, A[h].l[1], o2);
      }
      if (g == 0) {
        mine[h * 8 + 2 * t] = A[h].m[0]; mine[h * 8 + 2 * t + 1] = A[h].m[1];
        mine[ROWS + h * 8 + 2 * t] = A[h].l[0]; mine[ROWS + h * 8 + 2 * t + 1] = A[h].l[1];
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int c = 0; c < 4; ++c)
          mine[2 * ROWS + (h * 8 + 2 * t + (c & 1)) * OSTR + ((g >> 1) * 8 + 2 * mt + (c >> 1)) * 2 + (g & 1)] = A[h].o[mt][c];
    }
#else
#pragma unroll
    for (int h = 0; h < HKV; ++h) {
      A[h].l += __shfl_xor_sync(0xffffffffu, A[h].l, 1);
      A[h].l += __shfl_xor_sync(0xffffffffu, A[h].l, 2);
      const int row = h * 8 + g;
      if (t == 0) { mine[row] = A[h].m; mine[ROWS + row] = A[h].l; }
#pragma unroll
      for (int j = 0; j < 8; ++j)
        *reinterpret_cast<float2*>(mine + 2 * ROWS + row * OSTR + (t * 8 + j) * 2) = make_float2(A[h].o[j][0], A[h].o[j][1]);
    }
#endif
  }
#undef B1_CONSUME
  stampx(p, 6);  // warp partials written
  __syncthreads();
  stampx(p, 7);
  const int RPC = ROWS / CLs;  // rows per cluster rank
  {
    // thread (row tid >> 4, piece q) folds the 8 warps, then pushes the CTA partial of that row into the receive buffer
    // of its owner rank (row % CL), slot (this rank, row / CL)
    const int row = tid >> 4;
    float mw[NW];
#pragma unroll
    for (int w = 0; w < NW; ++w) mw[w] = wpart(w)[row];
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < NW; ++w) M = fmaxf(M, mw[w]);
    float Ls = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (M != -INFINITY) {
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        const float c = mw[w] == -INFINITY ? 0.f : ex2(mw[w] - M);
        const float4 x = *reinterpret_cast<const float4*>(wpart(w) + 2 * ROWS + row * OSTR + 4 * q);
        Ls += wpart(w)[ROWS + row] * c;
        acc.x += x.x * c; acc.y += x.y * c; acc.z += x.z * c; acc.w += x.w * c;
      }
    }
    stamp(p, 4);
    cluster_wait();  // every rank has started
    stampx(p, 8);
    const int slot = crank * RPC + row / CLs;
    if (samode) {
      const uint32_t rd = mapa_u32(sa_u32(recv), row % CLs), rbar = mapa_u32(sa_u32(&s_rbar), row % CLs);
      if (q == 0) st_async2(rd + 8 * slot, M, Ls, rbar);
      st_async4(rd + (2 * ROWS + slot * 64 + 4 * q) * 4, acc, rbar);
    } else {
      float* dst = cluster.map_shared_rank(recv, row % CLs);
      if (q == 0) *reinterpret_cast<float2*>(dst + 2 * slot) = make_float2(M, Ls);
      *reinterpret_cast<float4*>(dst + 2 * ROWS + slot * 64 + 4 * q) = acc;
    }
  }
  stampx(p, 9);  // pushes issued
  asm volatile("" ::"l"(gen));  // the epoch load has returned before this rank's release below
  if (samode) {
    mbar_wait_cluster(sa_u32(&s_rbar), 0);  // every source's bytes landed in this rank (no cluster-wide barrier)
  } else {
    cluster_arrive_release();
    cluster_wait_acquire();  // every rank's pushes into this rank are visible; nobody touches remote smem after this
  }
  stamp(p, 7);
  // --- merge. Thread -> (source src = tid % CL, slot = tid / CL = (row index ri, piece qq)): the CL sources of a
  //     (row, piece) sit in CL consecutive lanes and are folded with shuffles. Level 1, within the cluster: source rank
  //     src's partial from recv. Level 2, across clusters: the slot owners (src 0) publish the row's cluster partial;
  //     per row, the last-arriving owner CTA reads the ncl partials (lane src takes clusters src, src + CL) and folds
  //     them the same way. No thread leaves before the end (uniform barriers).
  const int src = tid & (CLs - 1), slot = tid / CLs, ri = slot >> 4, qq = slot & 15;
  const int frow = crank + CLs * ri;  // the row this thread folds
  float M, Ls;
  float4 acc;
  {
    const int sl = src * RPC + ri;
    const float2 ml = *reinterpret_cast<const float2*>(recv + 2 * sl);
    M = ml.x; Ls = ml.y;
    acc = *reinterpret_cast<const float4*>(recv + 2 * ROWS + sl * 64 + 4 * qq);
    group_fold(CLs, M, Ls, acc);
  }
  const int ncl = p.Stok / CLs, cid = s / CLs;  // clusters of this token
  float* xs = reinterpret_cast<float*>(ring);  // scratch: [256] (m, l, O[4]) per slot (the warp partials are consumed)
  if (ncl > 1 && llmode) {
    // across clusters, flag in data: every word of a cluster partial carries the epoch (gen + 1); cluster 0 polls the
    // other clusters' words (lane src: clusters src, src + CL) and folds them; the other clusters publish and leave
    const unsigned long long e = static_cast<unsigned long long>(static_cast<uint32_t>(gen + 1)) << 32;
    if (cid != 0) {
      if (src == 0) {
        unsigned long long* w = llbuf + 256 + static_cast<size_t>((prob * ncl + cid) * ROWS + frow) * 68;
        st_vol_v2u64(w + 4 + 4 * qq, llw(e, acc.x), llw(e, acc.y));
        st_vol_v2u64(w + 6 + 4 * qq, llw(e, acc.z), llw(e, acc.w));
        if (qq == 0) st_vol_v2u64(w, llw(e, M), llw(e, Ls));
      }
      return;  // no remote access after the cluster barrier above
    }
    float Mx = -INFINITY, Lx = 0.f;
    float4 ax = make_float4(0.f, 0.f, 0.f, 0.f);
    if (src == 0) { Mx = M; Lx = Ls; ax = acc; }
#pragma unroll 1
    for (int r = src; r < ncl; r += CLs) {
      if (r == 0) continue;
      const unsigned long long* w = llbuf + 256 + static_cast<size_t>((prob * ncl + r) * ROWS + frow) * 68;
      unsigned long long a0, a1, b0, b1, c0, c1;
      do {
        ld_vol_v2u64(w, a0, a1);
        ld_vol_v2u64(w + 4 + 4 * qq, b0, b1);
        ld_vol_v2u64(w + 6 + 4 * qq, c0, c1);
      } while (((a0 ^ e) | (a1 ^ e) | (b0 ^ e) | (b1 ^ e) | (c0 ^ e) | (c1 ^ e)) >> 32);
      fold(Mx, Lx, ax, llf(a0), llf(a1), make_float4(llf(b0), llf(b1), llf(c0), llf(c1)));
    }
    M = Mx; Ls = Lx; acc = ax;
    group_fold(CLs, M, Ls, acc);
    if (tid < ROWS) s_last[tid] = 1;
  } else if (ncl > 1) {  // across clusters
    if (src == 0) {
      if (qq == 0) *reinterpret_cast<float2*>(p.grp_ml + ((prob * ncl + cid) * ROWS + frow) * 2) = make_float2(M, Ls);
      *reinterpret_cast<float4*>(p.grp_o + ((prob * ncl + cid) * ROWS + frow) * 64 + 4 * qq) = acc;
    }
    __syncthreads();
    if (src == 0 && qq == 0) s_last[frow] = atom_add_acq_rel(&p.cnt[prob * ROWS + frow], 1) == ncl - 1;
    __syncthreads();
    M = -INFINITY; Ls = 0.f;
    acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (s_last[frow]) {  // lane src: clusters src and src + CL (host: ncl <= 2 CL)
      const int r0 = src, r1 = src + CLs;
      const float* gml = p.grp_ml + prob * ncl * ROWS * 2;
      const float* go = p.grp_o + prob * ncl * ROWS * 64;
      const float2 m0 = r0 < ncl ? __ldcg(reinterpret_cast<const float2*>(gml + (r0 * ROWS + frow) * 2)) : make_float2(-INFINITY, 0.f);
      const float2 m1 = r1 < ncl ? __ldcg(reinterpret_cast<const float2*>(gml + (r1 * ROWS + frow) * 2)) : make_float2(-INFINITY, 0.f);
      const float4 o0 = r0 < ncl ? __ldcg(reinterpret_cast<const float4*>(go + (r0 * ROWS + frow) * 64 + 4 * qq)) : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 o1 = r1 < ncl ? __ldcg(reinterpret_cast<const float4*>(go + (r1 * ROWS + frow) * 64 + 4 * qq)) : make_float4(0.f, 0.f, 0.f, 0.f);
      fold(M, Ls, acc, m0.x, m0.y, o0);
      fold(M, Ls, acc, m1.x, m1.y, o1);
    }
    group_fold(CLs, M, Ls, acc);
  }
  if (src == 0) {
    xs[slot * 6] = M; xs[slot * 6 + 1] = Ls;
    xs[slot * 6 + 2] = acc.x; xs[slot * 6 + 3] = acc.y; xs[slot * 6 + 4] = acc.z; xs[slot * 6 + 5] = acc.w;
  }
  __syncthreads();
  stamp(p, 5);
  // output threads: tid < 16 x rows per rank -> (row orow, piece q) = slot tid
  if (orow >= ROWS || (ncl > 1 && !s_last[orow])) return;
  M = xs[tid * 6]; Ls = xs[tid * 6 + 1];
  acc = make_float4(xs[tid * 6 + 2], xs[tid * 6 + 3], xs[tid * 6 + 4], xs[tid * 6 + 5]);
  if (llmode && ncl > 1) {
    if (cid == 0 && crank == 0 && tid == 0) {  // every cluster's row-0 partial arrived: every CTA of the problem read gen
      asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(llbuf + prob), "l"(gen + 1) : "memory");
    }
  } else if (ncl > 1 && q == 0) {
    p.cnt[prob * ROWS + orow] = 0;  // every owner of this row has arrived: no later access this launch
  }

  // --- output of head orow: O / l * v_scale -> rmsnorm over its 64 dims (the 16 q threads) -> * sigmoid(gate)
  const float inv = (Ls > 0.f ? 1.f / Ls : 0.f) / p.inv_v_scale;
  const float d[4] = {acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv};
  float ss = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
  const unsigned grpmask = 0xffffu << (lane & 16);  // the 16 q threads of this row share a half-warp
#pragma unroll
  for (int o2 = 8; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(grpmask, ss, o2);
  const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
  __nv_bfloat16* outp = p.out + (prob * ROWS + orow) * 64;
  *reinterpret_cast<__nv_bfloat162*>(outp + d0) = __floats2bfloat162_rn(d[0] * rn * sgate[0], d[2] * rn * sgate[1]);
  *reinterpret_cast<__nv_bfloat162*>(outp + d0 + 8) = __floats2bfloat162_rn(d[1] * rn * sgate[2], d[3] * rn * sgate[3]);
  stamp(p, 6);
}

// ===================================== q_len 8 (speculative verify block) =========================================
// Rows R = token j * 16 + head (head = kv head h * 8 + group head). Warp w = (kv head h = w & 1, m-tile mt = w >> 1): MMA
// rows g -> (token 2 mt, head h * 8 + g) and g + 8 -> (token 2 mt + 1, same head). Every warp consumes every tile of
// the CTA, so each 2-tile stage is dequantized ONCE per CTA (raw fp8 rows -> f16 rows padded to 272 B) and the warps
// take their fragments with ldmatrix; masks only on edge tiles; O is rescaled only when a row's max moved. Each warp's
// rows are its own: no intra-CTA reduce.
constexpr int LV = 8, RV = LV * ROWS;      // 128 rows
constexpr int FROW = 272;                  // f16 row stride (bytes): 128 dims + 16 B pad (conflict-free ldmatrix)
constexpr int FT = 16 * FROW;              // one f16 tile of K (or V): 16 keys
constexpr int FSTAGE = 2 * 2 * FT;         // a stage: 2 tiles x (K, V)
struct Acc8 {
  float o[8][4];
  float m[2], l[2];
};
__device__ __forceinline__ void ldsm4(uint32_t r[4], const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldsm4t(uint32_t r[4], const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
// one 16-key tile of kv head h (f16 K / V tiles, rows = keys) for rows g / g + 8. Edge tiles: key kk at absolute position
// pos0 + kk is valid for row r iff kk < nvalid and lo[r] <= pos <= hi[r].
__device__ __forceinline__ void consume8(Acc8& A, const uint32_t qa[4][4], const unsigned char* kt, const unsigned char* vt, int h,
                                         bool edge, int pos0, int nvalid, const int lo[2], const int hi[2]) {
  const int lane = threadIdx.x & 31, t = lane & 3, li = lane >> 3, lr = lane & 7;
  // every fragment of the tile first (8 ldmatrix in flight), then two interleaved QK chains, softmax, PV
  uint32_t kb[2][2][4], vb[4][4];
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int sp = 0; sp < 2; ++sp)  // k-steps 2 sp, 2 sp + 1: matrices (st, half) = (2 sp + (li >> 1), li & 1)
      ldsm4(kb[u][sp], kt + (8 * u + lr) * FROW + (64 * h + 16 * (2 * sp + (li >> 1)) + 8 * (li & 1)) * 2);
#pragma unroll
  for (int jp = 0; jp < 4; ++jp)  // n-tiles 2 jp, 2 jp + 1: matrices (n-tile, key half) = (2 jp + (li >> 1), li & 1)
    ldsm4t(vb[jp], vt + (8 * (li & 1) + lr) * FROW + (64 * h + 8 * (2 * jp + (li >> 1))) * 2);
  float sacc[2][4];
#pragma unroll
  for (int u = 0; u < 2; ++u) sacc[u][0] = sacc[u][1] = sacc[u][2] = sacc[u][3] = 0.f;
#pragma unroll
  for (int sp = 0; sp < 2; ++sp)
#pragma unroll
    for (int hs = 0; hs < 2; ++hs)
#pragma unroll
      for (int u = 0; u < 2; ++u) mma_f16(sacc[u], qa[2 * sp + hs], kb[u][sp][2 * hs], kb[u][sp][2 * hs + 1]);
  float tmax[2] = {-INFINITY, -INFINITY};
  if (edge) {
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int kk = 8 * u + 2 * t + (c & 1), r = c >> 1, pos = pos0 + kk;
        const float v = (kk < nvalid && pos >= lo[r] && pos <= hi[r]) ? sacc[u][c] : -INFINITY;  // q carries log2 e
        sacc[u][c] = v;
        tmax[r] = fmaxf(tmax[r], v);
      }
  } else {
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
      for (int c = 0; c < 4; ++c) tmax[c >> 1] = fmaxf(tmax[c >> 1], sacc[u][c]);
  }
  float mref[2], sc[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
    tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
    const float mn = fmaxf(A.m[r], tmax[r]);
    mref[r] = (mn == -INFINITY) ? 0.f : mn;  // a row with no valid key yet: every term below is ex2(-inf) = 0
    sc[r] = ex2(A.m[r] - mref[r]);
    A.m[r] = mn;
    A.l[r] *= sc[r];
  }
  if (__any_sync(0xffffffffu, sc[0] != 1.f || sc[1] != 1.f)) {
#pragma unroll
    for (int j = 0; j < 8; ++j) { A.o[j][0] *= sc[0]; A.o[j][1] *= sc[0]; A.o[j][2] *= sc[1]; A.o[j][3] *= sc[1]; }
  }
  float pv[2][4];
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int r = c >> 1;
      const float e = ex2(sacc[u][c] - mref[r]);
      pv[u][c] = e;
      A.l[r] += e;
    }
  const uint32_t pa[4] = {pack_f16x2(pv[0][0], pv[0][1]), pack_f16x2(pv[0][2], pv[0][3]), pack_f16x2(pv[1][0], pv[1][1]),
                          pack_f16x2(pv[1][2], pv[1][3])};
#pragma unroll
  for (int jp = 0; jp < 4; ++jp) {
    mma_f16(A.o[2 * jp], pa, vb[jp][0], vb[jp][1]);
    mma_f16(A.o[2 * jp + 1], pa, vb[jp][2], vb[jp][3]);
  }
}

// raw fp8 tile pair (K rows [16][128] then V rows [16][128] per tile, 2 tiles) -> f16 rows at FROW stride
__device__ __forceinline__ void dequant_stage(const unsigned char* raw, unsigned char* f16s) {
  const int tid = threadIdx.x;
  const int tile = tid >> 7, tensor = (tid >> 6) & 1, key = (tid >> 2) & 15, qtr = tid & 3;  // 32 raw bytes each
  const unsigned char* src = raw + tile * TILE + tensor * (TILE / 2) + key * ROWB + 32 * qtr;
  unsigned char* dst = f16s + (2 * tile + tensor) * FT + key * FROW + 64 * qtr;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint4 w = *reinterpret_cast<const uint4*>(src + 16 * i);
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    uint4 o0, o1;
    o0.x = e4m3x2_to_f16x2(ws[0]); o0.y = e4m3x2_to_f16x2(ws[0] >> 16); o0.z = e4m3x2_to_f16x2(ws[1]); o0.w = e4m3x2_to_f16x2(ws[1] >> 16);
    o1.x = e4m3x2_to_f16x2(ws[2]); o1.y = e4m3x2_to_f16x2(ws[2] >> 16); o1.z = e4m3x2_to_f16x2(ws[3]); o1.w = e4m3x2_to_f16x2(ws[3] >> 16);
    *reinterpret_cast<uint4*>(dst + 32 * i) = o0;
    *reinterpret_cast<uint4*>(dst + 32 * i + 16) = o1;
  }
}

// smem: ids [MAXCHUNK] | qs [128][64] f16 | newraw [TILE] | recv ([CL][128 / CL] (m, l), then [CL][128 / CL][64] O)
// | f16 stages [2][FSTAGE] | raw ring [RD][2 TILE] (after the loop: cross-cluster scratch)
constexpr int RECV8_BYTES = RV * 66 * 4;
template <int RD>
__global__ void __launch_bounds__(NW * 32, 1) b1_verify_kernel(const P p) {
  extern __shared__ __align__(16) unsigned char smem[];
  int* ids = reinterpret_cast<int*>(smem);
  __half* qs = reinterpret_cast<__half*>(smem + MAXCHUNK * 4);
  unsigned char* newraw = smem + MAXCHUNK * 4 + RV * 64 * 2;
  float* recv = reinterpret_cast<float*>(newraw + TILE);
  unsigned char* f16s = newraw + TILE + RECV8_BYTES;
  unsigned char* ring = f16s + 2 * FSTAGE;
  __shared__ int s_last[16];
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int h = warp & 1, mt = warp >> 1;
  const int rb = blockIdx.x / p.Stok, s = blockIdx.x - rb * p.Stok;  // request rb, its split s
  const bool tail = s == p.Stok - 1;
  const int QS = (2 * p.Hq + 2 * HKV) * 64;  // qkv row stride (per token)
  const __nv_bfloat16* qkvb = p.qkv + static_cast<size_t>(rb) * LV * QS;  // the request's 8 token rows
  const int32_t* ptab = p.page_table + static_cast<size_t>(rb) * p.max_pages;
  cluster_arrive_relaxed();
  stamp(p, 0);

  // --- first touches: length, this thread's query half-row (row tid >> 1, dims 32 (tid & 1) ..), gate, new tokens
  const int seq = p.seq_lens[rb];
  const int qr = tid >> 1, qh = tid & 1;  // query prep: row qr = token (qr >> 4) * 16 + head (qr & 15)
  uint4 qin[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) qin[i] = vld16(qkvb + (qr >> 4) * QS + (qr & 15) * 64 + 32 * qh + 8 * i);
  float4 ifq[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) ifq[i] = __ldg(reinterpret_cast<const float4*>(p.inv_freq) + 4 * qh + i);
  const int CLs = static_cast<int>(cluster.num_blocks()), crank = static_cast<int>(cluster.block_rank());
  const int RPCv = RV / CLs;                        // rows per cluster rank: 16 x RPCv (row, piece) slots
  const int npass = (RPCv * 16 + NW * 32 - 1) / (NW * 32);  // fold / output passes of 256 slots (1 at CL >= 8)
  const int q = tid & 15;
  // stored dims 4q .. 4q + 3 (= (t * 8 + j) * 2 + c of the PV fragments) are natural d0, d0 + 1, d0 + 8, d0 + 9
  const int d0 = 16 * (q & 3) + 2 * (q >> 2);
  const int qkv_gate = (p.Hq + 2 * HKV) * 64;
  // new tokens (tail CTA): thread -> (token tid >> 5, kv head (tid >> 4) & 1, piece q)
  const int ni = tid >> 5, nh = (tid >> 4) & 1;
  uint2 kin = make_uint2(0, 0), vin = kin;
  long long loc = 0;
  float2 ifr = make_float2(0.f, 0.f);
  if (tail) {
    kin = vld8(qkvb + ni * QS + p.Hq * 64 + nh * 64 + 4 * q);
    vin = vld8(qkvb + ni * QS + p.Hq * 64 + HKV * 64 + nh * 64 + 4 * q);
    loc = vld64(p.cache_loc + rb * LV + ni);
    const uint2 fr = vld8(p.inv_freq + 2 * q);
    ifr = make_float2(__uint_as_float(fr.x), __uint_as_float(fr.y));
  }

  // --- page ids -> smem (full layers: without waiting for the length); the window of token 0 starts the range
  const int kcap = p.max_pages << p.page_shift, pmask = (1 << p.page_shift) - 1;
  int k0 = s * p.chunk;
  if (p.window) k0 += max(0, seq - p.window + 1);
  if (PROBE_MODE(1 << 30)) {  // (probe mode bit 30: the serial loop, one DRAM trip per NW * 32 ids)
#pragma unroll 1
    for (int i = tid; i < p.chunk; i += NW * 32) {
      const int key = k0 + i;
      ids[i] = key < kcap ? (ptab[key >> p.page_shift] << p.page_shift) | (key & pmask) : 0;
    }
  } else {
    load_ids<NW * 32>(ids, ptab, k0, p.chunk, kcap, p.page_shift, tid);
  }
  const int k1 = min(seq, k0 + p.chunk);
  const int ncached = max(0, k1 - k0), ntiles = (ncached + 15) >> 4, nst = (ntiles + 1) >> 1;

  // --- query rows -> smem: rms * qk_scale (RoPE at position seq + token on sliding) -> bf16 -> * sm_scale * k_scale -> f16
  {
    float v[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&qin[i]);
#pragma unroll
      for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b[k]); v[8 * i + 2 * k] = f.x; v[8 * i + 2 * k + 1] = f.y; }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) ss += v[i] * v[i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f) * p.qk_scale;
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] *= inv;
    if (p.rope) {
      const float pos = static_cast<float>(seq + (qr >> 4));
      const float fq[16] = {ifq[0].x, ifq[0].y, ifq[0].z, ifq[0].w, ifq[1].x, ifq[1].y, ifq[1].z, ifq[1].w,
                            ifq[2].x, ifq[2].y, ifq[2].z, ifq[2].w, ifq[3].x, ifq[3].y, ifq[3].z, ifq[3].w};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float sn, cs;
        rope_sincos(pos * fq[i], &sn, &cs);
        const float e = v[2 * i], o = v[2 * i + 1];
        v[2 * i] = e * cs - o * sn;
        v[2 * i + 1] = e * sn + o * cs;
      }
    }
    uint4* dst = reinterpret_cast<uint4*>(qs + qr * 64 + 32 * qh);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      dst[i] = make_uint4(pack_f16x2(bf16_round(v[8 * i]) * p.q_mult, bf16_round(v[8 * i + 1]) * p.q_mult),
                          pack_f16x2(bf16_round(v[8 * i + 2]) * p.q_mult, bf16_round(v[8 * i + 3]) * p.q_mult),
                          pack_f16x2(bf16_round(v[8 * i + 4]) * p.q_mult, bf16_round(v[8 * i + 5]) * p.q_mult),
                          pack_f16x2(bf16_round(v[8 * i + 6]) * p.q_mult, bf16_round(v[8 * i + 7]) * p.q_mult));
  }
  __syncthreads();  // ids and query rows are visible
  stamp(p, 1);

  // --- raw ring: stage k = tiles 2k, 2k + 1 (plain rows) -> slot k % RD; thread -> one 16-byte copy per tile
  auto issue = [&](int k) {
    if (k < nst && !PROBE_MODE(2)) {
      unsigned char* slot = ring + (k % RD) * 2 * TILE;
      const int tensor = tid >> 7, key = (tid >> 3) & 15, ch = tid & 7;
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
        const int idx = 16 * (2 * k + tt) + key;
        const bool valid = idx < ncached;
        const size_t row = valid ? static_cast<size_t>(ids[idx]) : 0;
        cp_async16(slot + tt * TILE + tensor * (TILE / 2) + key * ROWB + ch * 16, (tensor ? p.v_cache : p.k_cache) + row * ROWB + ch * 16,
                   valid ? 16 : 0);
      }
    }
    cp_async_commit();
  };
#pragma unroll 1
  for (int k = 0; k < RD; ++k) issue(k);
  stamp(p, 2);

  // --- the 8 new tokens (tail CTA, while the copies fly): keys normalized (RoPE at seq + i on sliding) and quantized,
  //     values quantized; written to their cache rows (single writer; no CTA reads them) and as raw keys 0..7 of the
  //     new-token tile (keys 8..15 zero), attended last with the causal mask
  if (tail) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kin.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kin.y));
    float k4[4] = {a.x, a.y, b.x, b.y};
    norm_rope4(k4, 1.f, p.rope, static_cast<float>(seq + ni), ifr);
    const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vin.x));
    const float2 d = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vin.y));
    const float ik = p.inv_k_scale, iv = p.inv_v_scale;
    const uint32_t kq = fp8x4(k4[0] * ik, k4[1] * ik, k4[2] * ik, k4[3] * ik);
    const uint32_t vq = fp8x4(c.x * iv, c.y * iv, d.x * iv, d.y * iv);
    const int off = nh * 64 + 4 * q;  // byte offset in the row
    *reinterpret_cast<uint32_t*>(p.k_cache + static_cast<size_t>(loc) * ROWB + off) = kq;
    *reinterpret_cast<uint32_t*>(p.v_cache + static_cast<size_t>(loc) * ROWB + off) = vq;
    *reinterpret_cast<uint32_t*>(newraw + ni * ROWB + off) = kq;
    *reinterpret_cast<uint32_t*>(newraw + TILE / 2 + ni * ROWB + off) = vq;
    const int zr = 8 + (tid >> 5), zt = (tid >> 4) & 1;  // keys 8..15 (K rows then V rows): zeros
    *reinterpret_cast<uint2*>(newraw + zt * (TILE / 2) + zr * ROWB + (tid & 15) * 8) = make_uint2(0, 0);
  }

  // --- this warp's query A fragments (ldmatrix): rows lo = (token 2 mt, heads h * 8 ..), hi = (token 2 mt + 1, ..)
  uint32_t qa[4][4];
  {
    const int li = lane >> 3, lr = lane & 7;
    const int qrow = (2 * mt + (li & 1)) * 16 + h * 8 + lr;  // matrices: (lo, k 0-7), (hi, k 0-7), (lo, k 8-15), (hi, k 8-15)
#pragma unroll
    for (int st = 0; st < 4; ++st) ldsm4(qa[st], qs + qrow * 64 + 16 * st + 8 * (li >> 1));
  }
  const int lo[2] = {p.window ? seq + 2 * mt - p.window + 1 : 0, p.window ? seq + 2 * mt + 2 - p.window : 0};
  const int hi[2] = {seq + 2 * mt, seq + 2 * mt + 1};
  const int lo_max = p.window ? seq + 7 - p.window + 1 : 0;  // a tile is an edge tile if it starts below some row's bound
  Acc8 A;
#pragma unroll
  for (int j = 0; j < 8; ++j) A.o[j][0] = A.o[j][1] = A.o[j][2] = A.o[j][3] = 0.f;
  A.m[0] = A.m[1] = -INFINITY;
  A.l[0] = A.l[1] = 0.f;

  // --- main loop. Before it: stage 0 dequantized. Iteration k: [raw stage k + 1 landed] barrier (f16 stage k visible,
  //     everyone done with stage k - 1) -> refill the raw slot of stage k -> dequantize stage k + 1 -> consume stage k
  if (nst > 0) {
    cp_async_wait<RD - 1>();
    __syncthreads();
    dequant_stage(ring, f16s);
  }
#pragma unroll 1
  for (int k = 0; k < nst; ++k) {
    cp_async_wait<RD - 2>();
    __syncthreads();
    issue(k + RD);
    if (k + 1 < nst && !PROBE_MODE(32)) dequant_stage(ring + ((k + 1) % RD) * 2 * TILE, f16s + ((k + 1) & 1) * FSTAGE);
    const unsigned char* fs = f16s + (k & 1) * FSTAGE;
#pragma unroll 1
    for (int tt = 0; tt < 2; ++tt) {
      const int tile = 2 * k + tt;
      if (tile >= ntiles || PROBE_MODE(4)) break;
      const int pos0 = k0 + 16 * tile, nvalid = min(16, ncached - 16 * tile);
      consume8(A, qa, fs + 2 * tt * FT, fs + (2 * tt + 1) * FT, h, nvalid < 16 || pos0 < lo_max, pos0, nvalid, lo, hi);
    }
  }
  cp_async_wait<0>();
  __syncthreads();  // everyone is done with the f16 stages; the raw new-token tile is visible
  if (tail) {
    // dequantize the new-token tile (raw keys 0..15, K then V) into f16 stage 0, tile 0
    if (tid < 128) {
      const int tensor = (tid >> 6) & 1, key = (tid >> 2) & 15, qtr = tid & 3;
      const unsigned char* src = newraw + tensor * (TILE / 2) + key * ROWB + 32 * qtr;
      unsigned char* dst = f16s + tensor * FT + key * FROW + 64 * qtr;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 w = *reinterpret_cast<const uint4*>(src + 16 * i);
        *reinterpret_cast<uint4*>(dst + 32 * i) = make_uint4(e4m3x2_to_f16x2(w.x), e4m3x2_to_f16x2(w.x >> 16), e4m3x2_to_f16x2(w.y), e4m3x2_to_f16x2(w.y >> 16));
        *reinterpret_cast<uint4*>(dst + 32 * i + 16) = make_uint4(e4m3x2_to_f16x2(w.z), e4m3x2_to_f16x2(w.z >> 16), e4m3x2_to_f16x2(w.w), e4m3x2_to_f16x2(w.w >> 16));
      }
    }
    __syncthreads();
    consume8(A, qa, f16s, f16s + FT, h, true, seq, LV, lo, hi);
  }
  stamp(p, 3);

  // --- push this warp's rows into their owner ranks (row R -> rank R % CL, local row R / CL, source = this rank)
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    A.l[r] += __shfl_xor_sync(0xffffffffu, A.l[r], 1);
    A.l[r] += __shfl_xor_sync(0xffffffffu, A.l[r], 2);
  }
  cluster_wait();  // every rank has started
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int R = (2 * mt + r) * 16 + h * 8 + g;
    float* dst = cluster.map_shared_rank(recv, R % CLs);
    const int sl = crank * RPCv + R / CLs;
    if (t == 0) *reinterpret_cast<float2*>(dst + 2 * sl) = make_float2(A.m[r], A.l[r]);
    float* od = dst + 2 * RV + sl * 64;
#pragma unroll
    for (int j = 0; j < 8; j += 2)  // stored dims (t * 8 + j) * 2 + c: n-tiles j, j + 1 are 4 consecutive floats
      *reinterpret_cast<float4*>(od + (t * 8 + j) * 2) = make_float4(A.o[j][2 * r], A.o[j][2 * r + 1], A.o[j + 1][2 * r], A.o[j + 1][2 * r + 1]);
  }
  cluster_arrive_release();
  cluster_wait_acquire();
  stamp(p, 4);

  // --- fold: slot (local row orl, piece q) over the CL sources; then across clusters per row (last-arriving owner);
  //     pass ps covers local rows 16 ps .. 16 ps + 15 (thread -> local row 16 ps + (tid >> 4)); uniform barriers only
  const int ncl = p.Stok / CLs, cid = s / CLs;
#pragma unroll 1
  for (int ps = 0; ps < npass; ++ps) {
    const int orl = 16 * ps + (tid >> 4), orow = crank + CLs * orl;
    const bool oact = orl < RPCv;
    const int otok = (oact ? orow : crank) >> 4, ohead = (oact ? orow : crank) & 15;
    const uint32_t gate0 = vld4(qkvb + otok * QS + qkv_gate + ohead * 64 + d0);
    const uint32_t gate1 = vld4(qkvb + otok * QS + qkv_gate + ohead * 64 + d0 + 8);
    float M = -INFINITY, Ls = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (oact) {
      float2 ml[GS];
      float4 o4[GS];
#pragma unroll
      for (int r = 0; r < GS; ++r) {
        const int sl = r * RPCv + orl;
        ml[r] = r < CLs ? *reinterpret_cast<const float2*>(recv + 2 * sl) : make_float2(-INFINITY, 0.f);
        o4[r] = r < CLs ? *reinterpret_cast<const float4*>(recv + 2 * RV + sl * 64 + 4 * q) : make_float4(0.f, 0.f, 0.f, 0.f);
      }
#pragma unroll
      for (int r = 0; r < GS; ++r) fold(M, Ls, acc, ml[r].x, ml[r].y, o4[r]);
    }
    bool emit = oact;
    if (ncl > 1) {
      float* gml = p.grp_ml + static_cast<size_t>(rb) * ncl * RV * 2;
      float* go = p.grp_o + static_cast<size_t>(rb) * ncl * RV * 64;
      if (oact) {
        if (q == 0) *reinterpret_cast<float2*>(gml + (cid * RV + orow) * 2) = make_float2(M, Ls);
        *reinterpret_cast<float4*>(go + (cid * RV + orow) * 64 + 4 * q) = acc;
      }
      __syncthreads();
      if (oact && q == 0) s_last[tid >> 4] = atom_add_acq_rel(&p.cnt[rb * RV + orow], 1) == ncl - 1;
      __syncthreads();
      emit = oact && s_last[tid >> 4];
      if (emit) {
        float2 ml[GS];
        float4 o4[GS];
#pragma unroll
        for (int r = 0; r < GS; ++r) {
          ml[r] = r < ncl ? __ldcg(reinterpret_cast<const float2*>(gml + (r * RV + orow) * 2)) : make_float2(-INFINITY, 0.f);
          o4[r] = r < ncl ? __ldcg(reinterpret_cast<const float4*>(go + (r * RV + orow) * 64 + 4 * q)) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        M = -INFINITY; Ls = 0.f;
        acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int r = 0; r < GS; ++r) fold(M, Ls, acc, ml[r].x, ml[r].y, o4[r]);
        if (q == 0) p.cnt[rb * RV + orow] = 0;  // every owner of this row has arrived: no later access this launch
      }
      __syncthreads();  // s_last is reused by the next pass
    }
    if (emit) {
      // output of row orow (token otok, head ohead): O / l * v_scale -> rmsnorm over 64 dims -> * sigmoid(gate)
      const float inv = (Ls > 0.f ? 1.f / Ls : 0.f) / p.inv_v_scale;
      const float d[4] = {acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv};
      float ss = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
      const unsigned grpmask = 0xffffu << (lane & 16);
#pragma unroll
      for (int o2 = 8; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(grpmask, ss, o2);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate0));
      const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate1));
      __nv_bfloat16* outp = p.out + (static_cast<size_t>(rb) * RV + orow) * 64;
      *reinterpret_cast<__nv_bfloat162*>(outp + d0) =
          __floats2bfloat162_rn(d[0] * rn / (1.f + expf(-ga.x)), d[1] * rn / (1.f + expf(-ga.y)));
      *reinterpret_cast<__nv_bfloat162*>(outp + d0 + 8) =
          __floats2bfloat162_rn(d[2] * rn / (1.f + expf(-gb.x)), d[3] * rn / (1.f + expf(-gb.y)));
    }
  }
  stamp(p, 6);
}

// ===================================== q_len 8 via warpgroup MMA (wgmma) ==========================================
// wgmma.m64n64k16 f32 += f16 (A: registers, the m16n8k16 A fragment of this warp's 16 rows) x f16 (B: smem descriptor,
// K-major, no swizzle: 8 x 16-B core matrices, LBO = the K-direction stride, SBO = the N-direction stride)
__device__ __forceinline__ void wgmma64(float c[32], const uint32_t a[4], uint64_t db, int sd) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
  __trap();  // Hopper-only (sm_90a wgmma): other GPUs build this module for its mma.sync kernels
#else
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
#endif
}
// keeps the compiler from touching accumulators between an async wgmma and its wait
__device__ __forceinline__ void fence_acc(float c[32]) {
  asm volatile("" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31]));
}
__device__ __forceinline__ uint64_t wg_desc(const void* smem, uint32_t lbo, uint32_t sbo) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)((lbo & 0x3FFFF) >> 4) << 16) | ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32);
}
// warpgroup register re-allocation (sm_90a only; other GPUs build this module for its mma.sync kernels); each call site
// must be reached by every warp of its warpgroup at one PC
template <int INC, int N> __device__ __forceinline__ void ws_setmaxnreg() {
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900)
  if (INC) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N) : "memory");
  else asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N) : "memory");
#endif
}
// setmaxnreg.inc draws only on what this CTA's setmaxnreg.dec released (not the SM's unallocated registers): the
// consumers may take at most 128 x (168 - 64) / 256 = 52 more (216: multiple of 8); a larger inc blocks forever
constexpr int WS_NREG_MID = 168, WS_NREG_PROD = 64, WS_NREG_CONS = 216;
static_assert(128 * (WS_NREG_MID - WS_NREG_PROD) >= 256 * (WS_NREG_CONS - WS_NREG_MID), "setmaxnreg pool");
__device__ __forceinline__ void wg_fence() {
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900)
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit() {
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900)
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_wait0() {
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900)
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#endif
}
__device__ __forceinline__ float4 unpack4h(uint32_t a, uint32_t b, float l) {  // 4 f16 (O / l) -> O
  const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&a)), y = __half22float2(*reinterpret_cast<const __half2*>(&b));
  return make_float4(x.x * l, x.y * l, y.x * l, y.y * l);
}
__device__ __forceinline__ void wg_wait1() {  // all but the most recent wgmma group complete
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900)
  asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
#endif
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// A stage = 64 keys: raw rows (K [64][128] then V [64][128], 16 KB) -> per kv head h: K tile f16 [64 keys x 64 dims] and
// V^T tile f16 [64 dims x 64 keys], both in core-matrix layout: element (row n, col k) at ((k >> 3) * 8 + (n >> 3)) * 128 +
// (n & 7) * 16 + (k & 7) * 2 (K tile: n = key, k = dim; V^T tile: n = dim, k = key) -> descriptor LBO 1024, SBO 128.
constexpr int WTK = 64;                    // keys per stage
constexpr int WRAW = WTK * ROWB * 2;       // raw bytes per stage (K and V)
constexpr int WF = 64 * 64 * 2;            // one f16 tile (8 KB)
constexpr int WFST = 4 * WF;               // f16 stage: K h0, K h1, V^T h0, V^T h1 (32 KB)
__device__ __forceinline__ int core_off(int n, int k) { return ((k >> 3) * 8 + (n >> 3)) * 128 + (n & 7) * 16 + (k & 7) * 2; }
// raw stage layout (written by cp.async): K row key, 16-byte chunk c at key * 128 + ((c ^ (key & 7)) << 4); V row key at
// WTK * 128 + key * 128 + ((c ^ ((key >> 3) & 7)) << 4) - so both conversion read patterns below are bank-conflict free
__device__ __forceinline__ int raw_k_off(int key, int c) { return key * ROWB + ((c ^ (key & 7)) << 4); }
__device__ __forceinline__ int raw_v_off(int key, int c) { return WTK * ROWB + key * ROWB + ((c ^ ((key >> 3) & 7)) << 4); }
__device__ __forceinline__ void wg_dequant(const unsigned char* raw, unsigned char* f16s) {
  const int tid = threadIdx.x;
  {  // K: thread -> (key tid & 63, 32 bytes = chunks 2 b, 2 b + 1 with b = tid >> 6: kv head b >> 1); a phase of 8 lanes
     //    = 8 consecutive keys: the reads hit 8 swizzled chunks, the writes 8 core-matrix rows
    const int key = tid & 63, bb = tid >> 6, hq = bb >> 1;
    unsigned char* dst = f16s + hq * WF;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = 2 * bb + i;  // 16 dims: head hq, dims 16 (c & 3) ..
      const uint4 w = *reinterpret_cast<const uint4*>(raw + raw_k_off(key, c));
      const int dd = 16 * (c & 3);
      *reinterpret_cast<uint4*>(dst + core_off(key, dd)) =
          make_uint4(e4m3x2_to_f16x2(w.x), e4m3x2_to_f16x2(w.x >> 16), e4m3x2_to_f16x2(w.y), e4m3x2_to_f16x2(w.y >> 16));
      *reinterpret_cast<uint4*>(dst + core_off(key, dd + 8)) =
          make_uint4(e4m3x2_to_f16x2(w.z), e4m3x2_to_f16x2(w.z >> 16), e4m3x2_to_f16x2(w.w), e4m3x2_to_f16x2(w.w >> 16));
    }
  }
  {  // V^T: thread -> (key chunk kc = (tid & 3) + 4 (tid >> 7), word w = 2 ((tid >> 3) & 15) + ((tid >> 2) & 1) = dims
     //    4w .. 4w + 3): 8 rows x 4 bytes in, 4 x 16 B out (8 keys of one dim each); store i writes dim 4w + ((i + lane) & 3)
     //    so a phase's 8 lanes write 8 distinct dims mod 8 (conflict free)
    const int kc = (tid & 3) + 4 * (tid >> 7), w = 2 * ((tid >> 3) & 15) + ((tid >> 2) & 1);
    uint32_t v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = *reinterpret_cast<const uint32_t*>(raw + raw_v_off(8 * kc + k, w >> 2) + 4 * (w & 3));
    const int hq = w >> 4, dn = 4 * (w & 15);
    unsigned char* dst = f16s + (2 + hq) * WF;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = (i + tid) & 3;
      const uint32_t sel = c | ((4 + c) << 4);
      *reinterpret_cast<uint4*>(dst + core_off(dn + c, 8 * kc)) =
          make_uint4(e4m3x2_to_f16x2(__byte_perm(v[0], v[1], sel)), e4m3x2_to_f16x2(__byte_perm(v[2], v[3], sel)),
                     e4m3x2_to_f16x2(__byte_perm(v[4], v[5], sel)), e4m3x2_to_f16x2(__byte_perm(v[6], v[7], sel)));
    }
  }
}

// smem: ids [chunk] | qs [128][64] f16 | newraw [WRAW] | recv ([CL][128 / CL] (m, l), [CL][128 / CL][64] O) |
// f16 stages [2][WFST] | raw ring [RD][WRAW]. Warpgroup h (warps 4h..4h+3) = kv head h; warp w4 = warp & 3 holds MMA
// rows w4 * 16 + g / + 8 = (token 2 w4 / 2 w4 + 1, head h * 8 + g).
template <int RD>
__global__ void __launch_bounds__(NW * 32, 1) b1_wgv_kernel(const P p) {
  extern __shared__ __align__(128) unsigned char smem[];
  __half* qs = reinterpret_cast<__half*>(smem);
  unsigned char* newraw = smem + RV * 64 * 2;
  float* recv = reinterpret_cast<float*>(newraw + WRAW);
  unsigned char* f16s = newraw + WRAW + RECV8_BYTES;  // 1 KB aligned (RV*128 + 16 KB + 33 KB: see the host check)
  unsigned char* ring = f16s + 2 * WFST;
  int* ids = reinterpret_cast<int*>(ring + RD * WRAW);
  __shared__ int s_last[16];
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int h = warp >> 2, w4 = warp & 3;
  const int rb = blockIdx.x / p.Stok, s = blockIdx.x - rb * p.Stok;
  const bool tail = s == p.Stok - 1;
  const int QS = (2 * p.Hq + 2 * HKV) * 64;
  const __nv_bfloat16* qkvb = p.qkv + static_cast<size_t>(rb) * LV * QS;
  const int32_t* ptab = p.page_table + static_cast<size_t>(rb) * p.max_pages;
  cluster_arrive_relaxed();
  stamp(p, 0);

  const int seq = p.seq_lens[rb];
  const int qr = tid >> 1, qh = tid & 1;  // query prep: row qr = token (qr >> 4) * 16 + head (qr & 15)
  uint4 qin[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) qin[i] = vld16(qkvb + (qr >> 4) * QS + (qr & 15) * 64 + 32 * qh + 8 * i);
  float4 ifq[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) ifq[i] = __ldg(reinterpret_cast<const float4*>(p.inv_freq) + 4 * qh + i);
  const int CLs = static_cast<int>(cluster.num_blocks()), crank = static_cast<int>(cluster.block_rank());
  const int RPCv = RV / CLs;
  const int npass = (RPCv * 16 + NW * 32 - 1) / (NW * 32);
  const int q = tid & 15;
  const int d0 = 16 * (q & 3) + 2 * (q >> 2);  // stored dims 4q .. 4q + 3 = natural d0, d0 + 1, d0 + 8, d0 + 9
  const int qkv_gate = (p.Hq + 2 * HKV) * 64;
  const int ni = tid >> 5, nh = (tid >> 4) & 1;
  uint2 kin = make_uint2(0, 0), vin = kin;
  long long loc = 0;
  float2 ifr = make_float2(0.f, 0.f);
  if (tail) {
    kin = vld8(qkvb + ni * QS + p.Hq * 64 + nh * 64 + 4 * q);
    vin = vld8(qkvb + ni * QS + p.Hq * 64 + HKV * 64 + nh * 64 + 4 * q);
    loc = vld64(p.cache_loc + rb * LV + ni);
    const uint2 fr = vld8(p.inv_freq + 2 * q);
    ifr = make_float2(__uint_as_float(fr.x), __uint_as_float(fr.y));
  }

  const int kcap = p.max_pages << p.page_shift, pmask = (1 << p.page_shift) - 1;
  int k0 = s * p.chunk;
  if (p.window) k0 += max(0, seq - p.window + 1);
  if (PROBE_MODE(1 << 30)) {  // (probe mode bit 30: the serial loop, one DRAM trip per NW * 32 ids)
#pragma unroll 1
    for (int i = tid; i < p.chunk; i += NW * 32) {
      const int key = k0 + i;
      ids[i] = key < kcap ? (ptab[key >> p.page_shift] << p.page_shift) | (key & pmask) : 0;
    }
  } else {
    load_ids<NW * 32>(ids, ptab, k0, p.chunk, kcap, p.page_shift, tid);
  }
  const int k1 = min(seq, k0 + p.chunk);
  const int ncached = max(0, k1 - k0), nst = (ncached + WTK - 1) / WTK;

  {  // query rows -> smem: rms * qk_scale (RoPE at position seq + token) -> bf16 -> * q_mult (incl. log2 e) -> f16
    float v[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&qin[i]);
#pragma unroll
      for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b[k]); v[8 * i + 2 * k] = f.x; v[8 * i + 2 * k + 1] = f.y; }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) ss += v[i] * v[i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f) * p.qk_scale;
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] *= inv;
    if (p.rope) {
      const float pos = static_cast<float>(seq + (qr >> 4));
      const float fq[16] = {ifq[0].x, ifq[0].y, ifq[0].z, ifq[0].w, ifq[1].x, ifq[1].y, ifq[1].z, ifq[1].w,
                            ifq[2].x, ifq[2].y, ifq[2].z, ifq[2].w, ifq[3].x, ifq[3].y, ifq[3].z, ifq[3].w};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float sn, cs;
        rope_sincos(pos * fq[i], &sn, &cs);
        const float e = v[2 * i], o = v[2 * i + 1];
        v[2 * i] = e * cs - o * sn;
        v[2 * i + 1] = e * sn + o * cs;
      }
    }
    uint4* dst = reinterpret_cast<uint4*>(qs + qr * 64 + 32 * qh);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      dst[i] = make_uint4(pack_f16x2(bf16_round(v[8 * i]) * p.q_mult, bf16_round(v[8 * i + 1]) * p.q_mult),
                          pack_f16x2(bf16_round(v[8 * i + 2]) * p.q_mult, bf16_round(v[8 * i + 3]) * p.q_mult),
                          pack_f16x2(bf16_round(v[8 * i + 4]) * p.q_mult, bf16_round(v[8 * i + 5]) * p.q_mult),
                          pack_f16x2(bf16_round(v[8 * i + 6]) * p.q_mult, bf16_round(v[8 * i + 7]) * p.q_mult));
  }
  __syncthreads();  // ids and query rows are visible
  stamp(p, 1);

  // --- raw ring: stage k = keys 64k .. -> slot k % RD; thread -> 4 copies of 16 B (K and V rows, 8 chunks each)
  auto issue = [&](int k) {
    if (k < nst && !PROBE_MODE(2)) {
      unsigned char* slot = ring + (k % RD) * WRAW;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int it = r * NW * 32 + tid;  // 1024 copies: tensor (it >> 9), key ((it >> 3) & 63), chunk (it & 7)
        const int tensor = it >> 9, key = (it >> 3) & 63, ch = it & 7;
        const int idx = WTK * k + key;
        const bool valid = idx < ncached;
        const size_t row = valid ? static_cast<size_t>(ids[idx]) : 0;
        cp_async16(slot + (tensor ? raw_v_off(key, ch) : raw_k_off(key, ch)), (tensor ? p.v_cache : p.k_cache) + row * ROWB + ch * 16,
                   valid ? 16 : 0);
      }
    }
    cp_async_commit();
  };
#pragma unroll 1
  for (int k = 0; k < RD; ++k) issue(k);

  // --- the 8 new tokens (tail CTA): keys normalized (RoPE at seq + i) and quantized, values quantized; cache rows written
  //     here (single writer); raw keys 0..7 of the new-token stage (keys 8..63 zero), attended last with the causal mask
  if (tail) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kin.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kin.y));
    float k4[4] = {a.x, a.y, b.x, b.y};
    norm_rope4(k4, 1.f, p.rope, static_cast<float>(seq + ni), ifr);
    const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vin.x));
    const float2 d = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vin.y));
    const float ik = p.inv_k_scale, iv = p.inv_v_scale;
    const uint32_t kq = fp8x4(k4[0] * ik, k4[1] * ik, k4[2] * ik, k4[3] * ik);
    const uint32_t vq = fp8x4(c.x * iv, c.y * iv, d.x * iv, d.y * iv);
    const int off = nh * 64 + 4 * q;
    *reinterpret_cast<uint32_t*>(p.k_cache + static_cast<size_t>(loc) * ROWB + off) = kq;
    *reinterpret_cast<uint32_t*>(p.v_cache + static_cast<size_t>(loc) * ROWB + off) = vq;
#pragma unroll 1
    for (int i = tid; i < WRAW / 16; i += NW * 32) reinterpret_cast<uint4*>(newraw)[i] = make_uint4(0, 0, 0, 0);
    __syncthreads();  // tail is uniform across the CTA
    *reinterpret_cast<uint32_t*>(newraw + raw_k_off(ni, off >> 4) + (off & 15)) = kq;
    *reinterpret_cast<uint32_t*>(newraw + raw_v_off(ni, off >> 4) + (off & 15)) = vq;
  }

  // --- this warp's query A fragments (ldmatrix): rows lo = (token 2 w4, heads h * 8 ..), hi = (token 2 w4 + 1, ..)
  uint32_t qa[4][4];
  {
    const int li = lane >> 3, lr = lane & 7;
    const int qrow = (2 * w4 + (li & 1)) * 16 + h * 8 + lr;
#pragma unroll
    for (int st = 0; st < 4; ++st) ldsm4(qa[st], qs + qrow * 64 + 16 * st + 8 * (li >> 1));
  }
  const int lo[2] = {p.window ? seq + 2 * w4 - p.window + 1 : 0, p.window ? seq + 2 * w4 + 2 - p.window : 0};
  const int hi[2] = {seq + 2 * w4, seq + 2 * w4 + 1};
  const int lo_max = p.window ? seq + 7 - p.window + 1 : 0;
  float O[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) O[i] = 0.f;
  float M[2] = {-INFINITY, -INFINITY}, Lr[2] = {0.f, 0.f};

  // one 64-key stage from an f16 stage buffer: S = Q K^T (4 wgmma), softmax (masks on edge stages), O += P V (4 wgmma)
  auto consume = [&](const unsigned char* fs, int pos0, int nvalid, bool edge, auto&& overlap) {
    float S[32];
    const unsigned char* kt = fs + h * WF;
    const unsigned char* vt = fs + (2 + h) * WF;
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) wgmma64(S, qa[st], wg_desc(kt + st * 2048, 1024, 128), st);
    wg_commit();
    overlap();  // runs while the tensor cores compute S (it must not touch S, O or this stage's tiles)
    wg_wait0();
    fence_acc(S);
    float tmax[2] = {-INFINITY, -INFINITY};
    if (edge) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * t + (c & 1), r = c >> 1, pos = pos0 + kk;
          const float v = (kk < nvalid && pos >= lo[r] && pos <= hi[r]) ? S[4 * j + c] : -INFINITY;
          S[4 * j + c] = v;
          tmax[r] = fmaxf(tmax[r], v);
        }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) tmax[c >> 1] = fmaxf(tmax[c >> 1], S[4 * j + c]);
    }
    float mref[2], sc[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
      tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
      const float mn = fmaxf(M[r], tmax[r]);
      mref[r] = (mn == -INFINITY) ? 0.f : mn;
      sc[r] = ex2(M[r] - mref[r]);
      M[r] = mn;
      Lr[r] *= sc[r];
    }
    if (__any_sync(0xffffffffu, sc[0] != 1.f || sc[1] != 1.f)) {
#pragma unroll
      for (int j = 0; j < 8; ++j) { O[4 * j] *= sc[0]; O[4 * j + 1] *= sc[0]; O[4 * j + 2] *= sc[1]; O[4 * j + 3] *= sc[1]; }
    }
    uint32_t pa[4][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float e0 = ex2(S[4 * j] - mref[0]), e1 = ex2(S[4 * j + 1] - mref[0]);
      const float e2 = ex2(S[4 * j + 2] - mref[1]), e3 = ex2(S[4 * j + 3] - mref[1]);
      Lr[0] += e0 + e1;
      Lr[1] += e2 + e3;
      pa[j >> 1][(j & 1) * 2] = pack_f16x2(e0, e1);      // rows lo: keys 8j + 2t, +1  (k-step j >> 1, half j & 1)
      pa[j >> 1][(j & 1) * 2 + 1] = pack_f16x2(e2, e3);  // rows hi
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) wgmma64(O, pa[st], wg_desc(vt + st * 2048, 1024, 128), 1);
    wg_commit();
    wg_wait0();
    fence_acc(O);
  };

  // --- main loop. Before it: stage 0 dequantized. Iteration k: [raw stage k + 1 landed] barrier (f16 stage k visible,
  //     everyone done with stage k - 1) -> refill the raw slot of stage k -> dequantize stage k + 1 -> consume stage k
  if (nst > 0) {
    cp_async_wait<RD - 1>();
    __syncthreads();
    wg_dequant(ring, f16s);
    fence_proxy_async();  // generic-proxy smem writes -> visible to wgmma (async proxy) after the next barrier
  }
  stamp(p, 2);
#pragma unroll 1
  for (int k = 0; k < nst; ++k) {
    cp_async_wait<RD - 2>();
    __syncthreads();
    issue(k + RD);
    // the next stage's dequant overlaps this stage's QK wgmma (its buffer was last read by stage k - 1, complete)
    auto next_dequant = [&]() {
      if (k + 1 < nst && !PROBE_MODE(32)) {
        wg_dequant(ring + ((k + 1) % RD) * WRAW, f16s + ((k + 1) & 1) * WFST);
        fence_proxy_async();
      }
    };
    const int pos0 = k0 + WTK * k, nvalid = min(WTK, ncached - WTK * k);
    if (!PROBE_MODE(4)) consume(f16s + (k & 1) * WFST, pos0, nvalid, nvalid < WTK || pos0 < lo_max, next_dequant);
    else next_dequant();
    if (PROBE_MODE(4) && PROBE_MODE(8)) {  // QK only (no softmax / PV)
      float S[32];
      wg_fence();
#pragma unroll
      for (int st = 0; st < 4; ++st) wgmma64(S, qa[st], wg_desc(f16s + (k & 1) * WFST + h * WF + st * 2048, 1024, 128), st);
      wg_commit();
      wg_wait0();
      fence_acc(S);
      O[0] += S[0];
    }
  }
  cp_async_wait<0>();
  stamp(p, 3);
  __syncthreads();  // everyone is done with the f16 stages; the raw new-token stage is visible
  if (tail) {
    wg_dequant(newraw, f16s);
    fence_proxy_async();
    __syncthreads();
    consume(f16s, seq, LV, true, [] {});
  }

  // --- push this warp's rows into their owner ranks (row R -> rank R % CL, local row R / CL, source = this rank).
  //     O[4j + c] (c < 2: row lo, else row hi) is dim 8 j + 2 t + (c & 1): stored dims (t * 8 + j) * 2 + (c & 1)
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    Lr[r] += __shfl_xor_sync(0xffffffffu, Lr[r], 1);
    Lr[r] += __shfl_xor_sync(0xffffffffu, Lr[r], 2);
  }
  cluster_wait();
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int R = (2 * w4 + r) * 16 + h * 8 + g;
    float* dst = cluster.map_shared_rank(recv, R % CLs);
    const int sl = crank * RPCv + R / CLs;
    if (t == 0) *reinterpret_cast<float2*>(dst + 2 * sl) = make_float2(M[r], Lr[r]);
    float* od = dst + 2 * RV + sl * 64;
#pragma unroll
    for (int j = 0; j < 8; j += 2)
      *reinterpret_cast<float4*>(od + (t * 8 + j) * 2) = make_float4(O[4 * j + 2 * r], O[4 * j + 2 * r + 1], O[4 * j + 4 + 2 * r], O[4 * j + 5 + 2 * r]);
  }
  cluster_arrive_release();
  cluster_wait_acquire();
  stamp(p, 4);
  // --- fold: slot (local row orl, piece q) over the CL sources; then across clusters per row (last-arriving owner);
  //     pass ps covers local rows 16 ps .. 16 ps + 15 (thread -> local row 16 ps + (tid >> 4)); uniform barriers only
  const int ncl = p.Stok / CLs, cid = s / CLs;
#pragma unroll 1
  for (int ps = 0; ps < npass; ++ps) {
    const int orl = 16 * ps + (tid >> 4), orow = crank + CLs * orl;
    const bool oact = orl < RPCv;
    const int otok = (oact ? orow : crank) >> 4, ohead = (oact ? orow : crank) & 15;
    const uint32_t gate0 = vld4(qkvb + otok * QS + qkv_gate + ohead * 64 + d0);
    const uint32_t gate1 = vld4(qkvb + otok * QS + qkv_gate + ohead * 64 + d0 + 8);
    float M = -INFINITY, Ls = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (oact) {
      float2 ml[GS];
      float4 o4[GS];
#pragma unroll
      for (int r = 0; r < GS; ++r) {
        const int sl = r * RPCv + orl;
        ml[r] = r < CLs ? *reinterpret_cast<const float2*>(recv + 2 * sl) : make_float2(-INFINITY, 0.f);
        o4[r] = r < CLs ? *reinterpret_cast<const float4*>(recv + 2 * RV + sl * 64 + 4 * q) : make_float4(0.f, 0.f, 0.f, 0.f);
      }
#pragma unroll
      for (int r = 0; r < GS; ++r) fold(M, Ls, acc, ml[r].x, ml[r].y, o4[r]);
    }
    bool emit = oact;
    if (ncl > 1) {
      float* gml = p.grp_ml + static_cast<size_t>(rb) * ncl * RV * 2;
      float* go = p.grp_o + static_cast<size_t>(rb) * ncl * RV * 64;
      if (oact) {
        if (q == 0) *reinterpret_cast<float2*>(gml + (cid * RV + orow) * 2) = make_float2(M, Ls);
        *reinterpret_cast<float4*>(go + (cid * RV + orow) * 64 + 4 * q) = acc;
      }
      __syncthreads();
      if (oact && q == 0) s_last[tid >> 4] = atom_add_acq_rel(&p.cnt[rb * RV + orow], 1) == ncl - 1;
      __syncthreads();
      emit = oact && s_last[tid >> 4];
      if (emit) {
        float2 ml[GS];
        float4 o4[GS];
#pragma unroll
        for (int r = 0; r < GS; ++r) {
          ml[r] = r < ncl ? __ldcg(reinterpret_cast<const float2*>(gml + (r * RV + orow) * 2)) : make_float2(-INFINITY, 0.f);
          o4[r] = r < ncl ? __ldcg(reinterpret_cast<const float4*>(go + (r * RV + orow) * 64 + 4 * q)) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        M = -INFINITY; Ls = 0.f;
        acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int r = 0; r < GS; ++r) fold(M, Ls, acc, ml[r].x, ml[r].y, o4[r]);
        if (q == 0) p.cnt[rb * RV + orow] = 0;  // every owner of this row has arrived: no later access this launch
      }
      __syncthreads();  // s_last is reused by the next pass
    }
    if (emit) {
      // output of row orow (token otok, head ohead): O / l * v_scale -> rmsnorm over 64 dims -> * sigmoid(gate)
      const float inv = (Ls > 0.f ? 1.f / Ls : 0.f) / p.inv_v_scale;
      const float d[4] = {acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv};
      float ss = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
      const unsigned grpmask = 0xffffu << (lane & 16);
#pragma unroll
      for (int o2 = 8; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(grpmask, ss, o2);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate0));
      const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate1));
      __nv_bfloat16* outp = p.out + (static_cast<size_t>(rb) * RV + orow) * 64;
      *reinterpret_cast<__nv_bfloat162*>(outp + d0) =
          __floats2bfloat162_rn(d[0] * rn / (1.f + expf(-ga.x)), d[1] * rn / (1.f + expf(-ga.y)));
      *reinterpret_cast<__nv_bfloat162*>(outp + d0 + 8) =
          __floats2bfloat162_rn(d[2] * rn / (1.f + expf(-gb.x)), d[3] * rn / (1.f + expf(-gb.y)));
    }
  }
  stamp(p, 6);
}
// ============================ q_len 8, warp-specialized, FP8 tensor cores (wgmma e4m3) ==============================
// The accuracy tolerance admits e4m3 queries (the reference rounds queries to e4m3 too). Scores S = q8 . k8 with q8 =
// e4m3(rope(rms(q))) (the reference's e4m3 query) and the cached e4m3 keys read by wgmma STRAIGHT from the cp.async
// ring: each 128-B key row lands 128-B swizzled, which is the K-major SW128 operand layout (no K dequantization).
// P = 2^(s - m + 8) in e4m3 without MUFU: the e4m3 code of 2^x is ~ 8 x + 56 (linear between powers of two; one FFMA
// with a rounding magic number, a clamp and byte packing) - the 2^8 lifts P's tail above the e4m3 floor and cancels in
// O / l; l = sum of the SAME quantized P, from the MMA (V^T carries an all-ones row: n = 72). This adds about 2-4%
// relative error over the e4m3-query error. O += P V with V^T built per stage by a byte transpose (e4m3 wgmma wants
// K-major B). Roles: warpgroup 0 = producers - warps 0 / 1 load (keys 32 w .. of every 64-key stage: page ids
// cp.async'd RD stages ahead, cp.async of the K / V rows into an RD-deep raw ring), warps 2 / 3 transpose (key groups
// of 16: V -> V^T ring; they issue no cp.async, so their proxy fences are cheap); warpgroups 1 / 2 = consumers of kv
// head 0 / 1 (64 rows = 8 tokens x 8 query heads: one wgmma M tile), two stages per iteration. mbarriers: raw full
// (cp.async completion), raw empty (consumers' QK done + V read), V^T full / empty. On the tail CTA the transposers
// first prepare the 8 new tokens (cache rows + an extra stage the consumers attend first, while the gathers fly). Merge
// as in the verify kernels above.
constexpr int WS_NT = 384;
constexpr int WS_MAXP = 6;         // fold passes: 128 rows of a CL-1 CTA over WS_NT / 16 rows per pass
constexpr int WS_GP = 3;           // fold passes whose gates are requested early (CL >= 2 needs at most 3)
#ifdef OKC_B1_PROBE
#define WS_EARLY_GATES ((p.mode & (1 << 22)) != 0)  // probe build: early gate loads switched by mode bit 22
#else
#define WS_EARLY_GATES false  // off: the extra code costs more than the earlier gate loads save
#endif
constexpr int WSK = WTK * ROWB;     // raw K stage: 64 keys x 128 B, 128-B swizzled
constexpr int WSRAW = 2 * WSK;      // raw stage: K, then V (plain rows)
constexpr int WSVH = 9 * 4 * 128;   // V^T of one kv head: [72 rows (64 dims, ones, 7 zero)][64 key slots] e4m3
constexpr int WSVT = 2 * WSVH;      // V^T stage
constexpr int WS_VDMAX = 8;         // V^T stages (template VD; the consumers take two per iteration)
// V^T element (row n, key slot k): no-swizzle K-major core matrices (8 rows x 16 slots = 128 B): LBO 1152, SBO 128.
// Slots follow the e4m3 A fragment of P: in each 32-key block slot 16 hb + 4 t + e <-> key 16 hb + 8 (e >> 1) + 2 t +
// (e & 1), the S accumulator columns lane t holds, so P goes from accumulator to A fragment without shuffles.
__device__ __forceinline__ int vt_off(int n, int k) { return ((k >> 4) * 9 + (n >> 3)) * 128 + (n & 7) * 16 + (k & 15); }
// a V^T stage before its first use: zero, with row 64 (both heads) all ones (e4m3 1.0 = 0x38)
__device__ __forceinline__ void vt_init(unsigned char* vt, int t, int nt) {
  for (int i = t; i < WSVT / 16; i += nt) {
    const int o = 16 * i, blk = (o % WSVH) >> 7;
    const uint32_t v = (blk % 9 == 8 && (o & 127) < 16) ? 0x38383838u : 0u;
    reinterpret_cast<uint4*>(vt)[i] = make_uint4(v, v, v, v);
  }
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  const uint32_t a = smem_u32(b);
  uint32_t done;
  do {
    asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                 : "=r"(done) : "r"(a), "r"(parity) : "memory");
  } while (!done);
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* b) {  // arrives once this thread's prior cp.async landed
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void wgmma64_e4m3(float c[32], const uint32_t a[4], uint64_t db, int sd) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
  __trap();  // Hopper-only (sm_90a wgmma): other GPUs build this module for its mma.sync kernels
#else
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
#endif
}
__device__ __forceinline__ void wgmma64_e4m3_h(uint32_t c[16], const uint32_t a[4], uint64_t db, int sd) {  // f16 accumulators
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
  __trap();
#else
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]), "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]), "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]), "+r"(c[14]), "+r"(c[15])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
#endif
}
__device__ __forceinline__ void fence16(uint32_t c[16]) {
  asm volatile("" : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]), "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]), "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]), "+r"(c[14]), "+r"(c[15]));
}
__device__ __forceinline__ void wgmma64_e4m3_ss(float c[32], uint64_t da, uint64_t db, int sd) {  // A and B from smem
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
  __trap();
#else
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
               : "l"(da), "l"(db), "r"(sd));
#endif
}
__device__ __forceinline__ void wgmma72_e4m3(float c[36], const uint32_t a[4], uint64_t db) {  // accumulate
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
  __trap();  // Hopper-only (sm_90a wgmma): other GPUs build this module for its mma.sync kernels
#else
  asm volatile("wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, 1, 1, 1;\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31]), "+f"(c[32]), "+f"(c[33]), "+f"(c[34]), "+f"(c[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
#endif
}
__device__ __forceinline__ void fence_acc36(float c[36]) {
  asm volatile("" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31]), "+f"(c[32]), "+f"(c[33]), "+f"(c[34]), "+f"(c[35]));
}
__device__ __forceinline__ uint64_t wg_desc_sw128(const void* smem) {  // K-major, 128-B swizzle: SBO 1024, LBO unused
  return (uint64_t)((smem_u32(smem) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
// QK of one 64-key stage into S: hi-only queries (NQ 2) from per-thread A fragment slots (qsm[st * QSTR + slot]); hi + lo
// (NQ 4) from the head's swizzled query tile (A by descriptor)
template <int NQ, int QSTR>
__device__ __forceinline__ void qk_issue(float S[32], const uint4* qsm, int slot, const unsigned char* qt, int h,
                                         const unsigned char* kb) {
  if constexpr (NQ == 2) {
    uint32_t qa[2][4];
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const uint4 u = qsm[st * QSTR + slot];
      qa[st][0] = u.x; qa[st][1] = u.y; qa[st][2] = u.z; qa[st][3] = u.w;
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 2; ++st) wgmma64_e4m3(S, qa[st], wg_desc_sw128(kb + 64 * h + 32 * st), st);
  } else {
    wg_fence();
#pragma unroll
    for (int j = 0; j < NQ; ++j) wgmma64_e4m3_ss(S, wg_desc_sw128(qt + 8192 * h + 32 * j), wg_desc_sw128(kb + 64 * h + 32 * (j & 1)), j);
  }
}
template <int QSTR>
__device__ __forceinline__ void qk_issue_h(uint32_t S[16], const uint4* qsm, int slot, int h, const unsigned char* kb) {  // NQ 2
  uint32_t qa[2][4];
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const uint4 u = qsm[st * QSTR + slot];
    qa[st][0] = u.x; qa[st][1] = u.y; qa[st][2] = u.z; qa[st][3] = u.w;
  }
  wg_fence();
#pragma unroll
  for (int st = 0; st < 2; ++st) wgmma64_e4m3_h(S, qa[st], wg_desc_sw128(kb + 64 * h + 32 * st), st);
}
__device__ __forceinline__ float2 bf2f(uint32_t u) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u)); }

// smem (1 KB aligned): raw ring [RD][WSRAW] (after the loops: recv ([CL][128 / CL] (m, l), [CL][128 / CL][64] O)) |
// V^T ring [WS_VD][WSVT] | new-token K stage [WSK] | new-token V^T [WSVT] | queries [2 heads][64 rows][128 B: hi | lo]
// (128-B swizzled) | page ids
// [2 warps][16][32]
constexpr int WS_IDR = 16;  // page-id ring slots per producer warp (stages)
constexpr int WS_SMEM = 1024 + WSK + WSVT + 4 * 256 * 16 + 2 * WS_IDR * 32 * 4 + RV * 128;  // + RD * WSRAW + VD * WSVT
static_assert(4 * WSRAW >= RECV8_BYTES, "recv fits the raw ring");
template <int RD, int WS_VD, bool PIPE, bool W16, int NQ = 2, bool PP = false>  // PP: ping-pong. NQ: QK k32 steps (2: hi queries; 4: hi + lo). PIPE: the pipelined consumer (one loop per kernel: two make ptxas
// serialize). W16: 16 warps - 4 loaders (16 keys of a stage each), 4 transposers (one key group each), consumers on
// warps 8-15, producers' registers lent to the consumers (balanced setmaxnreg)
// PP: ping-pong - 2 loaders, 2 transposers; consumer warpgroup (h, par) = (kv head, stage parity) on the last
// 16 warps (640 threads); the two parities of a head alternate stages
__global__ void __launch_bounds__(PP ? (W16 ? 768 : 640) : W16 ? 512 : WS_NT, 1) b1_ws_kernel(const P p) {
  constexpr bool SMR = W16 || PP;  // balanced setmaxnreg
  // PP && W16 (B1_WS_PPW): ping-pong consumers with the W16 producers (4 loaders + 4 transposers), 768 threads
  constexpr int NT = PP ? (W16 ? 768 : 640) : W16 ? 512 : WS_NT, NL = W16 ? 4 : 2, NX = W16 ? 4 : 2;
  constexpr int CB = NL + NX, CT0 = 32 * CB;
  constexpr int KPW = WTK / NL, RPW = KPW / 4;
  constexpr int QSTR = PP ? 512 : 256;  // hi-only query A fragment slots per k step (one per consumer thread)  // keys per loader warp per stage, its copy instructions per tensor
  constexpr int REG_MID = PP ? (W16 ? 80 : 96) : W16 ? 128 : 168, REG_PROD = PP ? (W16 ? 48 : 56) : 72;
  constexpr int REG_CONS = PP ? (W16 ? 96 : 104) : 184;
  static_assert(!SMR || 32 * CB * (REG_MID - REG_PROD) >= (NT - 32 * CB) * (REG_CONS - REG_MID), "setmaxnreg pool");
  static_assert(!PP || CB % 4 == 0, "producers fill whole warpgroups (setmaxnreg is per warpgroup)");
  extern __shared__ __align__(1024) unsigned char smem_ws[];
  unsigned char* ring = smem_ws + ((1024 - (smem_u32(smem_ws) & 1023)) & 1023);  // SW128 atoms: 1 KB aligned
  unsigned char* vts = ring + RD * WSRAW;
  unsigned char* newk = vts + WS_VD * WSVT;
  unsigned char* newvt = newk + WSK;
  float* recv = reinterpret_cast<float*>(ring);  // merge receive buffer: the raw ring, once every rank of the cluster is done
  uint4* qsm = reinterpret_cast<uint4*>(newvt + WSVT);
  int* idr = reinterpret_cast<int*>(qsm + 4 * 256);
  const unsigned char* qt = reinterpret_cast<const unsigned char*>(qsm);  // query tiles (wgmma A descriptors)
  unsigned char* gsm = reinterpret_cast<unsigned char*>(idr + 2 * WS_IDR * 32);  // [RV / CL local rows][64] bf16 gates
  // compile-time switches (runtime-gated alternatives would leave dead paths in the code):
  constexpr bool gsmode = true;   // gates staged in smem by the loaders
  constexpr bool f16push = true;  // the cluster merge moves O / l as f16
  __shared__ __align__(8) uint64_t bars[2 * RD + 2 * WS_VD + 3];
  static_assert(WS_VD <= WS_VDMAX && WS_VD % 2 == 0, "V^T stages");
  __shared__ int s_last[NT / 4];  // fold tickets: per local row (quarter layout: up to NT / 4 rows)
  uint64_t* raw_full = bars;
  uint64_t* raw_empty = bars + RD;
  uint64_t* vt_full = bars + 2 * RD;
  uint64_t* vt_empty = vt_full + WS_VD;
  uint64_t* new_full = vt_empty + WS_VD;
  uint64_t* done_bar = new_full + 1;  // phase 0 completes at setup: a wait on parity 0 always passes
  uint64_t* push_bar = done_bar + 1;  // st.async merge (mode bit 17): every peer's pushed bytes into this rank
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rb = blockIdx.x / p.Stok, s = blockIdx.x - rb * p.Stok;
  const bool tail = s == p.Stok - 1;
  const int QS = (2 * p.Hq + 2 * HKV) * 64;
  const __nv_bfloat16* qkvb = p.qkv + static_cast<size_t>(rb) * p.L * QS;  // p.L tokens (verify 8; batch decode 1)
  const int32_t* ptab = p.page_table + static_cast<size_t>(rb) * p.max_pages;
  if ((p.mode & (1 << 10)) && p.window) {  // (B1_PTPF_B) this request's page-table row -> L2, split over its CTAs
    const int rowb = p.max_pages * 4, per = (rowb + p.Stok - 1) / p.Stok;
    const int beg = s * per, end = min(rowb, beg + per);
    for (int off = beg + 128 * tid; off < end; off += 128 * static_cast<int>(blockDim.x)) l2pf(reinterpret_cast<const unsigned char*>(ptab) + off, false);
  }
  if (tid == 0) {
#pragma unroll 1
    for (int i = 0; i < RD; ++i) {
      mbar_init(raw_full + i, 32 * NL);       // loader threads (cp.async completion)
      mbar_init(raw_empty + i, 32 * NX + 8);  // transposer threads + consumer warps
    }
    for (int i = 0; i < WS_VD; ++i) {
      mbar_init(vt_full + i, 32 * NX);
      mbar_init(vt_empty + i, 8);
    }
    mbar_init(new_full, 64);
    mbar_init(done_bar, 1);
    mbar_arrive(done_bar);
    if (PP && NQ == 2) {  // st.async merge: only the ping-pong kernel with hi-only queries (it pays off on batched full layers)
      mbar_init(push_bar, 1);
      const int pb = RV * (8 + (f16push ? 128 : 256));  // RV row partials (CL sources x RV / CL rows)
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(push_bar)), "r"(pb) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  stamp(p, 0);
  // consumers: the RoPE frequencies (a constant input) and, not PDL-chained, their query rows requested now, in flight
  // across the CTA barrier and the register rebalance (under PDL the rows load after griddepcontrol.wait)
  uint2 qpre[2][4];
  float2 fpre[4];
  if (warp >= CB) {
    const int qcw = PP ? ((warp - CB) & 7) : warp - CB, qh = qcw >> 2, qw4 = qcw & 3, qg = lane >> 2, qt = lane & 3;
#pragma unroll
    for (int c = 0; c < 4; ++c) fpre[c] = __ldg(reinterpret_cast<const float2*>(p.inv_freq + 8 * c + 2 * qt));
    if (!p.pdl) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const __nv_bfloat16* qr = qkvb + (2 * qw4 + r < p.L ? 2 * qw4 + r : 0) * QS + (8 * qh + qg) * 64 + 4 * qt;
#pragma unroll
        for (int c = 0; c < 4; ++c) qpre[r][c] = cld8(qr + 16 * c);
      }
    }
  }
  const int seq = p.seq_lens[rb];
  int k0 = s * p.chunk;
  if (p.window) k0 += max(0, seq - p.window + 1);
  const int ncached = max(0, min(seq, k0 + p.chunk) - k0), nst = (ncached + WTK - 1) / WTK;
  const float sc = p.qk_scale * p.q_mult;  // raw e4m3 scores -> log2 units
  const int roff = PROBE_MODE(8192) ? 1 : 0;
  float O[36], M[2] = {-INFINITY, -INFINITY};  // O[32 + 2 r] (lane t = 0): l of row r (the ones row of V^T)
#pragma unroll
  for (int i = 0; i < 36; ++i) O[i] = 0.f;
  __syncthreads();  // barrier init and the V^T stages' constant rows visible
  // PDL (decoder chains): dependents may launch now; reads of the previous kernel's output (qkv) and every global write
  // wait for it (griddepcontrol.wait: consumers before the query, transposers before the new tokens, all before the
  // fold); page ids and cache rows (inputs of the step) are gathered meanwhile
  gdc_launch();
  if (PROBE_MODE(1 << 20)) gdc_wait();

  // (probe modes: 1 << 16 consumers alone - producers idle, consumers never wait nor release; 1 << 14 producers alone -
  //  consumers idle, producers never wait for free slots. Every thread still meets the merge's cluster barriers)
  const bool consonly = PROBE_MODE(1 << 16), prodonly = PROBE_MODE(1 << 14);
  if (warp < CB) {
  if constexpr (SMR) ws_setmaxnreg<0, REG_PROD>();  // (every producer warpgroup at this one PC)
  if (!consonly) {
    // producer warpgroup(s) (loaders, transposers). (12 warps: no setmaxnreg - the pipelined consumer fits the launch
    // share; an unbalanced inc - more than this CTA's dec released - blocks forever)
    if (warp < NL) {
    // ================= loader warp lw: keys 32 lw .. 32 lw + 31 of every stage -> the raw ring (cp.async)
    const int lw = warp;
    const int kcap = p.max_pages << p.page_shift, pmask = (1 << p.page_shift) - 1;
    // page-table entries of this warp's keys, cp.async'd RD stages ahead into a per-warp ring (stage k -> slot k % 16):
    // a register prefetch would stall at the loop-carried move on the load. Lean layout: this warp copies keys
    // n = q + 4 (lw + NL j) (q < 4, j < JN; lane quarter q = lane >> 3 issues them); fetch lane l = (j, q) = (l >> 2, l & 3)
    // stores key n's id at q JN + j, so each quarter reads its ids with JN / 4 16-B loads
    constexpr int JN = KPW / 4;
    static_assert(NL % 2 == 0 && JN % 4 == 0, "lean loader: (lw + NL j) & 1 = lw & 1, ids in 16-B groups");
    int* myidr = idr + lw * WS_IDR * KPW;
    auto fetch_ids = [&](int k) {
      if (lane < KPW) {
        const int key = k0 + WTK * k + (lane & 3) + 4 * (lw + NL * (lane >> 2));
        const bool ok = key < kcap;
        const uint32_t d = smem_u32(myidr + (k % WS_IDR) * KPW + (lane & 3) * JN + (lane >> 2));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(d), "l"(ptab + (ok ? key >> p.page_shift : 0)), "r"(ok ? 4 : 0));
      }
    };
    // (full layers: the first RD stages' ids without waiting for the length - in-bounds table reads; sliding layers
    //  need it for the window start anyway)
    const int npre = p.window ? min(nst, RD) : RD;
#pragma unroll 1
    for (int k = 0; k < npre; ++k) fetch_ids(k);
    cp_async_commit();
    cp_async_wait<0>();
    __syncwarp();
#pragma unroll 1
    for (int k = 0; k < nst; ++k) {
      if (tid == 0) tstamp(p, k, 0);
      if (k >= RD) {
        if (!prodonly) mbar_wait(raw_empty + k % RD, ((k / RD) - 1) & 1);
        cp_async_wait<RD - 1>();  // the ids of stage k (fetched with the group of iteration k - RD) landed
        __syncwarp();
      }
      if (tid == 0) tstamp(p, k, 1);
      {
        // key n = q + 4 m (m = lw + NL j): K chunk c at n * 128 + ((c ^ (n & 7)) << 4), n & 7 = q + 4 (lw & 1);
        // V chunk c at WSK + n * 128 + 16 c; source row (id << shift) | (position & page mask), offset row * 128
        const int q = lane >> 3, c = lane & 7;
        const uint32_t sb = smem_u32(ring + (k % RD) * WSRAW) + q * 128;
        const uint32_t kd = sb + ((c ^ (q + 4 * (lw & 1))) << 4), vd = sb + WSK + 16 * c;
        const unsigned char* kc = p.k_cache + 16 * c;
        const unsigned char* vc = p.v_cache + 16 * c;
        const uint64_t pol = l2_evict_first();  // (streamed once)
        const int4* ip = reinterpret_cast<const int4*>(myidr + (k % WS_IDR) * KPW + q * JN);
        int idv[JN];
#pragma unroll
        for (int v = 0; v < JN / 4; ++v) {
          const int4 u = ip[v];
          idv[4 * v] = u.x; idv[4 * v + 1] = u.y; idv[4 * v + 2] = u.z; idv[4 * v + 3] = u.w;
        }
        const int pos0 = k0 + WTK * k + q + 4 * lw;  // position of this lane's key j = 0
        if (WTK * (k + 1) <= ncached) {
#pragma unroll
          for (int j = 0; j < JN; ++j) {
            const size_t off = static_cast<size_t>((static_cast<uint32_t>(idv[j]) << p.page_shift) | ((pos0 + 4 * NL * j) & pmask)) << 7;
            cp_async16s(kd + (lw + NL * j) * 512, kc + off, pol);
            cp_async16s(vd + (lw + NL * j) * 512, vc + off, pol);
          }
        } else {  // the last stage: keys past the cached ones zero-filled (row 0, no bytes read)
#pragma unroll
          for (int j = 0; j < JN; ++j) {
            const bool valid = pos0 + 4 * NL * j - k0 < ncached;
            const size_t off = valid ? static_cast<size_t>((static_cast<uint32_t>(idv[j]) << p.page_shift) | ((pos0 + 4 * NL * j) & pmask)) << 7 : 0;
            cp_async16sz(kd + (lw + NL * j) * 512, kc + off, valid ? 16 : 0, pol);
            cp_async16sz(vd + (lw + NL * j) * 512, vc + off, valid ? 16 : 0, pol);
          }
        }
      }
      cp_async_mbar_arrive(raw_full + k % RD);
      if (tid == 0) tstamp(p, k, 12);
      if (k + RD < nst) fetch_ids(k + RD);  // after the arrive: raw full does not wait for it
      cp_async_commit();
    }
    if (gsmode) {
      // this rank's output gates -> smem while the consumers still work (the merge then reads no global memory);
      // qkv is the previous kernel's output: after the wait. Visible to the fold through the cluster barriers
      if (p.pdl) gdc_wait();
      const int CLg = static_cast<int>(cluster.num_blocks()), crg = static_cast<int>(cluster.block_rank());
      const int qkv_gate_ = (p.Hq + 2 * HKV) * 64;
#pragma unroll 1
      for (int i = tid; i < (RV / CLg) * 8; i += 32 * NL) {
        const int orl = i >> 3, ch = i & 7, orow = crg + CLg * orl;
        if ((orow >> 4) < p.L) cp_async16(gsm + orl * 128 + ch * 16, qkvb + (orow >> 4) * QS + qkv_gate_ + (orow & 15) * 64 + ch * 8, 16);
      }
      cp_async_commit();
      cp_async_wait<0>();
    }
    } else {
    // ================= transposer warp tw (no cp.async of its own: its proxy fences stay cheap): key groups 2 tw,
    // 2 tw + 1 of every stage -> V^T; first, on the tail CTA, the 8 new tokens
    const int tw = warp - NL, tt = tid - 32 * NL;
    for (int i = 0; i < WS_VD; ++i) vt_init(vts + i * WSVT, tt, 32 * NX);
    if (tail && tt < 64) {  // (the first two transposer warps: the named barrier below counts 64)
      if (p.pdl) gdc_wait();
      // thread -> (token tt >> 3, kv head (tt >> 2) & 1, dims 16 (tt & 3) ..): cache rows written here (single writer);
      // their e4m3 K rows / V^T columns form the extra stage (keys 8.. of it: K garbage, masked; V^T 0)
      const int ni = tt >> 3, nh = (tt >> 2) & 1, q4 = tt & 3;
      vt_init(newvt, tt, 64);
      const bool nok = ni < p.L;  // (a block of p.L < 8 tokens: the other lanes only take part in the barriers)
      const __nv_bfloat16* kr = qkvb + (nok ? ni : 0) * QS + p.Hq * 64 + nh * 64 + 16 * q4;
      const uint4 kin[2] = {cld16(kr), cld16(kr + 8)};
      const uint4 vin[2] = {cld16(kr + HKV * 64), cld16(kr + HKV * 64 + 8)};
      const long long loc = nok ? vld64(p.cache_loc + rb * p.L + ni) : 0;
      const float4 fr0 = __ldg(reinterpret_cast<const float4*>(p.inv_freq) + 2 * q4);
      const float4 fr1 = __ldg(reinterpret_cast<const float4*>(p.inv_freq) + 2 * q4 + 1);
      const float f8[8] = {fr0.x, fr0.y, fr0.z, fr0.w, fr1.x, fr1.y, fr1.z, fr1.w};
      float kf[16], vf[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t kw = (i < 4) ? (&kin[0].x)[i] : (&kin[1].x)[i - 4], vw = (i < 4) ? (&vin[0].x)[i] : (&vin[1].x)[i - 4];
        const float2 a2 = bf2f(kw), b2 = bf2f(vw);
        kf[2 * i] = a2.x; kf[2 * i + 1] = a2.y; vf[2 * i] = b2.x; vf[2 * i + 1] = b2.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) ss += kf[i] * kf[i];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int i = 0; i < 16; ++i) kf[i] *= inv;
      if (p.rope) {
        const float pos = static_cast<float>(seq + ni);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float sn, cs;
          rope_sincos(pos * f8[i], &sn, &cs);
          const float e = kf[2 * i], o = kf[2 * i + 1];
          kf[2 * i] = e * cs - o * sn;
          kf[2 * i + 1] = e * sn + o * cs;
        }
      }
      const float ik = p.inv_k_scale, iv = p.inv_v_scale;
      uint32_t kq[4], vq[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        kq[i] = fp8x4(kf[4 * i] * ik, kf[4 * i + 1] * ik, kf[4 * i + 2] * ik, kf[4 * i + 3] * ik);
        vq[i] = fp8x4(vf[4 * i] * iv, vf[4 * i + 1] * iv, vf[4 * i + 2] * iv, vf[4 * i + 3] * iv);
      }
      const uint4 k16 = make_uint4(kq[0], kq[1], kq[2], kq[3]), v16 = make_uint4(vq[0], vq[1], vq[2], vq[3]);
      if (nok) {
        *reinterpret_cast<uint4*>(p.k_cache + static_cast<size_t>(loc) * ROWB + nh * 64 + 16 * q4) = k16;
        *reinterpret_cast<uint4*>(p.v_cache + static_cast<size_t>(loc) * ROWB + nh * 64 + 16 * q4) = v16;
      }
      *reinterpret_cast<uint4*>(newk + raw_k_off(ni, 4 * nh + q4)) = k16;
      asm volatile("bar.sync 1, 64;" ::: "memory");  // the zeroed new V^T stage is complete
      unsigned char* vt = newvt + nh * WSVH;
      const int sl = 4 * (ni >> 1) + (ni & 1);
      if (nok) {
#pragma unroll
        for (int e = 0; e < 16; ++e) vt[vt_off(16 * q4 + e, sl)] = static_cast<unsigned char>(vq[e >> 2] >> (8 * (e & 3)));
      }
      fence_proxy_async();
      mbar_arrive(new_full);
      stamp_t(p, 1, 32 * NL);
    }
    const int rot = (lane >> 1) & 3;
    // The selectors rotate the chosen source byte, folding all per-word SHF into the first PRMT pass.
    const uint32_t sel_a = 0x4040u | ((0x1100u + 0x1111u * rot) & 0x3333u);
    const uint32_t sel_b = 0x4040u | ((0x3322u + 0x1111u * rot) & 0x3333u);
#pragma unroll 1
    for (int j = 0; j < nst; ++j) {
      // V^T of stage j, key groups kg (slot chunk kg): lane -> dims 4 (lane & 15) .. + 3 of kv head lane >> 4; 4 x 4 byte
      // transposes of keys (2 t, 2 t + 1, 2 t + 8, 2 t + 9); store i holds dim (i + rot) & 3 (conflict-free STS.128)
      if (tid == (PROBE_MODE(8192) ? 32 * NL + 32 : 32 * NL)) tstamp(p, j, 2);
      if (!prodonly) mbar_wait(raw_full + j % RD, (j / RD) & 1);  // (producers alone: no waits at all - a lapped
      //                                                              // parity wait could block forever)
      if (tid == (PROBE_MODE(8192) ? 32 * NL + 32 : 32 * NL)) tstamp(p, j, 3);
      if (j >= WS_VD && !prodonly) mbar_wait(vt_empty + j % WS_VD, ((j / WS_VD) - 1) & 1);
      __syncwarp();
      if (tid == (PROBE_MODE(8192) ? 32 * NL + 32 : 32 * NL)) tstamp(p, j, 4);
      if (!PROBE_MODE(32)) {
#pragma unroll
        for (int g2 = 0; g2 < 4 / NX; ++g2) {
          const int kg = (4 / NX) * tw + g2;
          const unsigned char* src = ring + (j % RD) * WSRAW + WSK + 16 * kg * ROWB;
          uint32_t w[16];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            // Each matrix is ONE 128-B V row, viewed as 8 x 8 b16. Lanes 8m+r provide its eight 16-B chunks.
            // Normal ldmatrix returns byte [4*lane .. 4*lane+3], the same per-lane bytes as a 4-byte scalar load.
            const int m = lane >> 3;
            const int kk = 2 * t + (m & 1) + 8 * (m >> 1);
            ldsm4(w + 4 * t, src + kk * ROWB + 16 * (lane & 7));
          }
          uint32_t o[4][4];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const uint32_t a = __byte_perm(w[4 * t], w[4 * t + 1], sel_a), b = __byte_perm(w[4 * t], w[4 * t + 1], sel_b);
            const uint32_t c = __byte_perm(w[4 * t + 2], w[4 * t + 3], sel_a), d = __byte_perm(w[4 * t + 2], w[4 * t + 3], sel_b);
            o[0][t] = __byte_perm(a, c, 0x5410);
            o[1][t] = __byte_perm(a, c, 0x7632);
            o[2][t] = __byte_perm(b, d, 0x5410);
            o[3][t] = __byte_perm(b, d, 0x7632);
          }
          unsigned char* dst = vts + (j % WS_VD) * WSVT + (lane >> 4) * WSVH;
#pragma unroll
          for (int i = 0; i < 4; ++i)
            *reinterpret_cast<uint4*>(dst + vt_off(4 * (lane & 15) + ((i + rot) & 3), 16 * kg)) = make_uint4(o[i][0], o[i][1], o[i][2], o[i][3]);
        }
      }
      fence_proxy_async();  // generic V^T stores -> wgmma (async proxy)
      mbar_arrive(vt_full + j % WS_VD);
      mbar_arrive(raw_empty + j % RD);
      if (tid == (PROBE_MODE(8192) ? 32 * NL + 32 : 32 * NL)) tstamp(p, j, 5);
    }
    stamp(p, 4);
    }
  }
  if constexpr (SMR) ws_setmaxnreg<1, REG_MID>();  // (back to the launch share for the merge)
  } else {
    if constexpr (SMR) ws_setmaxnreg<1, REG_CONS>();
    // ================= consumer warpgroup h: rows lo = (token 2 w4, head 8 h + g), hi = (token 2 w4 + 1, same head)
    const int cw = PP ? ((warp - CB) & 7) : warp - CB, h = cw >> 2, w4 = cw & 3, g = lane >> 2, t = lane & 3;
    const int par = PP ? (warp - CB) >> 3 : 0;  // (PP: stage parity of this warpgroup)
    if (p.pdl) gdc_wait();
#ifdef OKC_B1_PROBE
    if (PROBE_MODE(1 << 21) && tid == CT0 && p.dbg) {
      unsigned long long t;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
      p.dbg[60000 + blockIdx.x * 4] = cld4(qkvb);
      p.dbg[60000 + blockIdx.x * 4 + 1] = t;
      p.dbg[60000 + blockIdx.x * 4 + 2] = p.pdl;
    }
#endif
    // queries -> e4m3 hi + lo (lo = e4m3(q - hi): ~7 significant bits) in this head's swizzled tile, A row
    // 16 w4 + 8 r + g; QK reads them by descriptor (4 wgmmas per stage, no A registers)
    {
      float v[2][16];  // [row][4 c + e] = dim 16 c + 4 t + e
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const bool qok = 2 * w4 + r < p.L;  // (rows of missing tokens: zero queries, never written out)
        const __nv_bfloat16* qr = qkvb + (qok ? 2 * w4 + r : 0) * QS + (8 * h + g) * 64 + 4 * t;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (p.pdl) qpre[r][c] = cld8(qr + 16 * c);
          const uint2 u = qpre[r][c];
          const float2 a = bf2f(u.x), b = bf2f(u.y);
          v[r][4 * c] = a.x; v[r][4 * c + 1] = a.y; v[r][4 * c + 2] = b.x; v[r][4 * c + 3] = b.y;
        }
        if (!qok) {
#pragma unroll
          for (int i = 0; i < 16; ++i) v[r][i] = 0.f;
        }
      }
      float2 fr[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) fr[c] = fpre[c];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += v[r][i] * v[r][i];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int i = 0; i < 16; ++i) v[r][i] *= inv;
        if (p.rope) {
          const float pos = static_cast<float>(seq + 2 * w4 + r);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float sn, cs;
            rope_sincos(pos * ((i & 1) ? fr[i >> 1].y : fr[i >> 1].x), &sn, &cs);
            const float e = v[r][2 * i], o = v[r][2 * i + 1];
            v[r][2 * i] = e * cs - o * sn;
            v[r][2 * i + 1] = e * sn + o * cs;
          }
        }
      }
      if constexpr (NQ == 2) {
        // hi only: e4m3 A fragments (k step st = dims 32 st ..; regs (lo, hi) x (dims + 4 t, + 16 + 4 t)) in this thread's
        // smem slots, reloaded per stage (ptxas treats register A operands of wgmma as dead after the instruction)
#pragma unroll
        for (int st = 0; st < 2; ++st)
          qsm[st * QSTR + tid - CT0] = make_uint4(fp8x4(v[0][8 * st], v[0][8 * st + 1], v[0][8 * st + 2], v[0][8 * st + 3]),
                                                  fp8x4(v[1][8 * st], v[1][8 * st + 1], v[1][8 * st + 2], v[1][8 * st + 3]),
                                                  fp8x4(v[0][8 * st + 4], v[0][8 * st + 5], v[0][8 * st + 6], v[0][8 * st + 7]),
                                                  fp8x4(v[1][8 * st + 4], v[1][8 * st + 5], v[1][8 * st + 6], v[1][8 * st + 7]));
      } else {
        unsigned char* qh = reinterpret_cast<unsigned char*>(qsm) + 8192 * h;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int m = 16 * w4 + 8 * r + g;
#pragma unroll
          for (int c = 0; c < 4 && par == 0; ++c) {
            const float* x = v[r] + 4 * c;
            const uint32_t hi = fp8x4(x[0], x[1], x[2], x[3]);
            __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(hi & 0xffffu), __NV_E4M3);
            __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(hi >> 16), __NV_E4M3);
            const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&h01));
            const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&h23));
            const uint32_t lo = fp8x4(x[0] - a.x, x[1] - a.y, x[2] - b.x, x[3] - b.y);
            *reinterpret_cast<uint32_t*>(qh + m * 128 + ((c ^ (m & 7)) << 4) + 4 * t) = hi;
            *reinterpret_cast<uint32_t*>(qh + m * 128 + (((c + 4) ^ (m & 7)) << 4) + 4 * t) = lo;
          }
        }
        fence_proxy_async();  // generic query stores -> wgmma (async proxy)
        asm volatile("bar.sync %0, %1;" ::"r"(2 + h), "r"(PP ? 256 : 128) : "memory");  // this head's tile complete
      }
    }
    stamp_t(p, 2, CT0);
    const int lo[2] = {p.window ? seq + 2 * w4 - p.window + 1 : 0, p.window ? seq + 2 * w4 + 2 - p.window : 0};
    const int hi[2] = {seq + 2 * w4, seq + 2 * w4 + 1};
    const int lo_max = p.window ? seq + 7 - p.window + 1 : 0;
    // stages k, k + 1 per iteration (k = -1: the new tokens, tail CTA, alone): S = Q K^T (wgmma per 32 dims), softmax
    // over the (up to) 128 keys (masks on edge stages), O += P V. The PV wgmmas complete under the next iteration's
    // QK wait (its V^T stages are released there)
    const int sb = tail ? -1 : 0;  // first stage (-1: the new tokens, tail CTA)
    if constexpr (PP) {
    if (nst > sb && !prodonly) {
      // ping-pong: this warpgroup takes stages sb + par, sb + par + 2, ..; the other parity's warpgroup of the same head
      // runs its QK / PV while this one does its softmax. Each step: QK, wait (QK + own previous PV), release the raw
      // stage (K read) and the previous V^T stage, softmax, rescale, wait for this stage's V^T, PV
      int prevs = -1;  // this warpgroup's previous ring stage (its PV retires at this stage's wait)
#pragma unroll 1
      for (int sx = sb + par; sx < nst; sx += 2) {
        const bool nw = sx < 0;
        mbar_wait(nw ? new_full : raw_full + sx % RD, nw ? 0 : (sx / RD) & 1);
        __syncwarp();
        float S[32];
#ifdef B1_H2A
        uint32_t Sh[16];  // (NQ 2) f16 scores: Sh[2 j] row g, Sh[2 j + 1] row g + 8, keys 8 j + 2 t, + 1
#endif
        {
          const unsigned char* kb = nw ? newk : ring + (sx % RD) * WSRAW;
#ifdef B1_H2A
          if constexpr (NQ == 2) qk_issue_h<QSTR>(Sh, qsm, tid - CT0, h, kb);
          else
#endif
          qk_issue<NQ, QSTR>(S, qsm, tid - CT0, qt, h, kb);
          wg_commit();
        }
        wg_wait0();  // QK of this stage and this warpgroup's previous PV
#ifdef B1_H2A
        if constexpr (NQ == 2) fence16(Sh);
        else
#endif
        fence_acc(S);
        fence_acc36(O);
        if (lane == 0) {
          if (!nw) mbar_arrive(raw_empty + sx % RD);  // K read (QK done); the transposers release V
          if (prevs >= 0) mbar_arrive(vt_empty + prevs % WS_VD);  // the previous stage's PV is done
        }
        const int pos0 = nw ? seq : k0 + WTK * sx;
        const int nvalid = nw ? p.L : min(WTK, ncached - WTK * sx);
        float mref[2], scl[2];
        const float sc8 = 8.f * sc;
        uint32_t pa[2][4];
#ifdef B1_H2
        if (!(nvalid < WTK || pos0 < lo_max)) {  // full stage: packed-half row max and e4m3 codes
          (void)mref;
          uint32_t h[2][8];  // row r, n-tile j: f16x2 of keys 8 j + 2 t, + 1
#ifdef B1_H2A
          if constexpr (NQ == 2) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { h[0][j] = Sh[2 * j]; h[1][j] = Sh[2 * j + 1]; }
          } else
#endif
          {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const __half2 x0 = __floats2half2_rn(S[4 * j], S[4 * j + 1]), x1 = __floats2half2_rn(S[4 * j + 2], S[4 * j + 3]);
            h[0][j] = *reinterpret_cast<const uint32_t*>(&x0);
            h[1][j] = *reinterpret_cast<const uint32_t*>(&x1);
          }
          }
          const __half2 sc8h = __float2half2_rn(sc8), zc = __float2half2_rn(1024.f);
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            __half2 x[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) x[j] = *reinterpret_cast<const __half2*>(&h[r][j]);
            const __half2 mx = __hmax2(__hmax2(__hmax2(x[0], x[4]), __hmax2(x[1], x[5])), __hmax2(__hmax2(x[2], x[6]), __hmax2(x[3], x[7])));
            float tmax = fmaxf(__low2float(mx), __high2float(mx));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            const float mn = fmaxf(M[r], tmax);
            scl[r] = (M[r] == -INFINITY) ? 1.f : ex2((M[r] - mn) * sc);
            M[r] = mn;
            const __half2 mr = __float2half2_rn(1144.f - sc8 * mn);  // code = v - 1024 = 120 + 8 sc (s - mn)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const __half2 v = __hmax2(__hfma2(x[j], sc8h, mr), zc);
              h[r][j] = *reinterpret_cast<const uint32_t*>(&v);
            }
          }
#pragma unroll
          for (int bk = 0; bk < 2; ++bk) {  // the f16 low bytes (codes) of keys 2 t, 2 t + 1 of two n-tiles per register
            pa[bk][0] = __byte_perm(h[0][4 * bk], h[0][4 * bk + 1], 0x6420);
            pa[bk][1] = __byte_perm(h[1][4 * bk], h[1][4 * bk + 1], 0x6420);
            pa[bk][2] = __byte_perm(h[0][4 * bk + 2], h[0][4 * bk + 3], 0x6420);
            pa[bk][3] = __byte_perm(h[1][4 * bk + 2], h[1][4 * bk + 3], 0x6420);
          }
        } else
#endif
        {
#ifdef B1_H2A
          if constexpr (NQ == 2) {  // masked stage: the f16 scores as f32
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&Sh[2 * j]));
              const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&Sh[2 * j + 1]));
              S[4 * j] = a0.x; S[4 * j + 1] = a0.y; S[4 * j + 2] = a1.x; S[4 * j + 3] = a1.y;
            }
          }
#endif
          float tm[2][4] = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY}, {-INFINITY, -INFINITY, -INFINITY, -INFINITY}};
          if (nvalid < WTK || pos0 < lo_max) {
  #pragma unroll
            for (int j = 0; j < 8; ++j)
  #pragma unroll
              for (int c = 0; c < 4; ++c) {
                const int kk = 8 * j + 2 * t + (c & 1), r = c >> 1, pos = pos0 + kk;
                const float v = (kk < nvalid && pos >= lo[r] && pos <= hi[r]) ? S[4 * j + c] : -INFINITY;
                S[4 * j + c] = v;
                tm[r][(j & 1) * 2 + (c & 1)] = fmaxf(tm[r][(j & 1) * 2 + (c & 1)], v);
              }
          } else {
  #pragma unroll
            for (int j = 0; j < 8; ++j)
  #pragma unroll
              for (int c = 0; c < 4; ++c) tm[c >> 1][(j & 1) * 2 + (c & 1)] = fmaxf(tm[c >> 1][(j & 1) * 2 + (c & 1)], S[4 * j + c]);
          }
          float tmax[2] = {fmaxf(fmaxf(tm[0][0], tm[0][1]), fmaxf(tm[0][2], tm[0][3])), fmaxf(fmaxf(tm[1][0], tm[1][1]), fmaxf(tm[1][2], tm[1][3]))};
  #pragma unroll
          for (int r = 0; r < 2; ++r) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
            const float mn = fmaxf(M[r], tmax[r]);
            scl[r] = (mn == -INFINITY) ? 1.f : ex2((M[r] - mn) * sc);
            mref[r] = (mn == -INFINITY) ? 12582912.f : 12582912.f + 120.f - 8.f * sc * mn;
            M[r] = mn;
          }
  #pragma unroll
          for (int bk = 0; bk < 2; ++bk) {
            uint32_t e[4][4];
  #pragma unroll
            for (int jj = 0; jj < 4; ++jj)
  #pragma unroll
              for (int c = 0; c < 4; ++c)
                e[jj][c] = __float_as_uint(fmaxf(fmaf(S[4 * (4 * bk + jj) + c], sc8, mref[c >> 1]), 12582912.f));
            pa[bk][0] = __byte_perm(__byte_perm(e[0][0], e[0][1], 0x0040), __byte_perm(e[1][0], e[1][1], 0x0040), 0x5410);
            pa[bk][1] = __byte_perm(__byte_perm(e[0][2], e[0][3], 0x0040), __byte_perm(e[1][2], e[1][3], 0x0040), 0x5410);
            pa[bk][2] = __byte_perm(__byte_perm(e[2][0], e[2][1], 0x0040), __byte_perm(e[3][0], e[3][1], 0x0040), 0x5410);
            pa[bk][3] = __byte_perm(__byte_perm(e[2][2], e[2][3], 0x0040), __byte_perm(e[3][2], e[3][3], 0x0040), 0x5410);
          }
        }
#ifdef B1_RSKIP
        if (__any_sync(0xffffffffu, scl[0] != 1.f || scl[1] != 1.f)) {  // some row's running max moved
#else
        {
#endif
#pragma unroll
          for (int j = 0; j < 9; ++j) { O[4 * j] *= scl[0]; O[4 * j + 1] *= scl[0]; O[4 * j + 2] *= scl[1]; O[4 * j + 3] *= scl[1]; }
        }
        mbar_wait(nw ? new_full : vt_full + sx % WS_VD, nw ? 0 : (sx / WS_VD) & 1);
        __syncwarp();
        {
          const unsigned char* vb = (nw ? newvt : vts + (sx % WS_VD) * WSVT) + WSVH * h;
          wg_fence();
#pragma unroll
          for (int st = 0; st < 2; ++st) wgmma72_e4m3(O, pa[st], wg_desc(vb + 2 * 9 * 128 * st, 9 * 128, 128));
          wg_commit();
        }
        prevs = nw ? -1 : sx;
      }
      wg_wait0();
      fence_acc36(O);
    }
    // the two parities of each head: parity 1 -> parity 0 (row max and O / l rescaled to the joint max) through the
    // raw ring (every consumer loop is done, so every stage was loaded and consumed)
    asm volatile("bar.sync 6, 512;" ::: "memory");
    {
      float* xb = reinterpret_cast<float*>(ring) + static_cast<size_t>(cw * 32 + lane) * 40;
      if (par == 1) {
        xb[0] = M[0]; xb[1] = M[1];
#pragma unroll
        for (int i = 0; i < 36; ++i) xb[2 + i] = O[i];
      }
      asm volatile("bar.sync 6, 512;" ::: "memory");
      if (par == 0) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float m1 = xb[r], mn = fmaxf(M[r], m1);
          const float a = (M[r] == -INFINITY) ? 0.f : ex2((M[r] - mn) * sc), b = (m1 == -INFINITY) ? 0.f : ex2((m1 - mn) * sc);
#pragma unroll
          for (int j = 0; j < 9; ++j) {
            O[4 * j + 2 * r] = a * O[4 * j + 2 * r] + b * xb[2 + 4 * j + 2 * r];
            O[4 * j + 2 * r + 1] = a * O[4 * j + 2 * r + 1] + b * xb[2 + 4 * j + 2 * r + 1];
          }
          M[r] = mn;
        }
      }
    }
    } else
    if constexpr (PIPE) {
    if (nst > sb && !prodonly) {
      // pipelined: one stage of 64 keys per step; the next stage's QK is issued before this stage's softmax and the
      // previous stage's PV is retired only before the O rescale (both by one wait_group 0), so the tensor cores run
      // under the softmax. Past the last stage a step re-reads the last stage fully masked (P = 0): waits and wgmmas stay
      // unconditional (done_bar always passes)
      // stage order over steps si = 0 .. ntot - 1: the tail CTA's new tokens (stage -1) first, or last (mode bit 15)
      const bool nlast = tail && (p.mode & (1 << 15));
      const int ntot = nst - sb;
      auto stg = [&](int si) { return nlast ? (si == nst ? -1 : si) : si + sb; };
      float SA[32], SB[32];
      int prevs = -1;  // the stage whose PV is in flight (-1: none / not a ring stage)
      {
        const int s0 = stg(0);
        mbar_wait(consonly ? done_bar : s0 < 0 ? new_full : raw_full, 0);
        __syncwarp();
        const unsigned char* kb = s0 < 0 ? newk : ring;
        qk_issue<NQ, QSTR>(SA, qsm, tid - CT0, qt, h, kb);
        wg_commit();
        wg_wait0();
        fence_acc(SA);
        if (s0 >= 0 && lane == 0 && !consonly) mbar_arrive(raw_empty);
      }
      auto step = [&](float (&Sc)[32], float (&Sn)[32], const int si) {
        const int sx = stg(si), sn = stg(si + 1);  // this step's stage, the next one's (-1: the new tokens)
        const bool rc = si < ntot, rn = si + 1 < ntot;
        // 1. QK of the next stage (or a masked re-read of the last one) -> Sn
        if (tid == CT0) tstamp(p, sx, 6);
        const bool nowait = consonly;
        mbar_wait(rn && !nowait ? (sn < 0 ? new_full : raw_full + sn % RD) : done_bar, rn && !nowait && sn >= 0 ? (sn / RD) & 1 : 0);
        if (tid == CT0) tstamp(p, sx, 7);
        __syncwarp();
        {
          const int kn = rn ? sn : nst - 1;
          const unsigned char* kb = kn < 0 ? newk : ring + (kn % RD) * WSRAW;
          qk_issue<NQ, QSTR>(Sn, qsm, tid - CT0, qt, h, kb);
          wg_commit();
        }
        if (tid == CT0) tstamp(p, sx, 8);
        // 2. softmax of this stage (masked keys -> -inf; a re-read stage is entirely masked)
        const int pos0 = sx < 0 ? seq : k0 + WTK * sx;
        const int nvalid = !rc ? 0 : sx < 0 ? p.L : min(WTK, ncached - WTK * sx);
        float mref[2], scl[2];
        const float sc8 = 8.f * sc;
        uint32_t pa[2][4];
#ifdef B1_H2
        if (!(nvalid < WTK || pos0 < lo_max)) {  // full stage: packed-half row max and e4m3 codes
          (void)mref;
          uint32_t h[2][8];  // row r, n-tile j: f16x2 of keys 8 j + 2 t, + 1
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const __half2 x0 = __floats2half2_rn(Sc[4 * j], Sc[4 * j + 1]), x1 = __floats2half2_rn(Sc[4 * j + 2], Sc[4 * j + 3]);
            h[0][j] = *reinterpret_cast<const uint32_t*>(&x0);
            h[1][j] = *reinterpret_cast<const uint32_t*>(&x1);
          }
          const __half2 sc8h = __float2half2_rn(sc8), zc = __float2half2_rn(1024.f);
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            __half2 x[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) x[j] = *reinterpret_cast<const __half2*>(&h[r][j]);
            const __half2 mx = __hmax2(__hmax2(__hmax2(x[0], x[4]), __hmax2(x[1], x[5])), __hmax2(__hmax2(x[2], x[6]), __hmax2(x[3], x[7])));
            float tmax = fmaxf(__low2float(mx), __high2float(mx));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            const float mn = fmaxf(M[r], tmax);
            scl[r] = (M[r] == -INFINITY) ? 1.f : ex2((M[r] - mn) * sc);
            M[r] = mn;
            const __half2 mr = __float2half2_rn(1144.f - sc8 * mn);  // code = v - 1024 = 120 + 8 sc (s - mn)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const __half2 v = __hmax2(__hfma2(x[j], sc8h, mr), zc);
              h[r][j] = *reinterpret_cast<const uint32_t*>(&v);
            }
          }
#pragma unroll
          for (int bk = 0; bk < 2; ++bk) {  // the f16 low bytes (codes) of keys 2 t, 2 t + 1 of two n-tiles per register
            pa[bk][0] = __byte_perm(h[0][4 * bk], h[0][4 * bk + 1], 0x6420);
            pa[bk][1] = __byte_perm(h[1][4 * bk], h[1][4 * bk + 1], 0x6420);
            pa[bk][2] = __byte_perm(h[0][4 * bk + 2], h[0][4 * bk + 3], 0x6420);
            pa[bk][3] = __byte_perm(h[1][4 * bk + 2], h[1][4 * bk + 3], 0x6420);
          }
        } else
#endif
        {
          float tm[2][4] = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY}, {-INFINITY, -INFINITY, -INFINITY, -INFINITY}};
          if (nvalid < WTK || pos0 < lo_max) {
  #pragma unroll
            for (int j = 0; j < 8; ++j)
  #pragma unroll
              for (int c = 0; c < 4; ++c) {
                const int kk = 8 * j + 2 * t + (c & 1), r = c >> 1, pos = pos0 + kk;
                const float v = (kk < nvalid && pos >= lo[r] && pos <= hi[r]) ? Sc[4 * j + c] : -INFINITY;
                Sc[4 * j + c] = v;
                tm[r][(j & 1) * 2 + (c & 1)] = fmaxf(tm[r][(j & 1) * 2 + (c & 1)], v);
              }
          } else {
  #pragma unroll
            for (int j = 0; j < 8; ++j)
  #pragma unroll
              for (int c = 0; c < 4; ++c) tm[c >> 1][(j & 1) * 2 + (c & 1)] = fmaxf(tm[c >> 1][(j & 1) * 2 + (c & 1)], Sc[4 * j + c]);
          }
          float tmax[2] = {fmaxf(fmaxf(tm[0][0], tm[0][1]), fmaxf(tm[0][2], tm[0][3])), fmaxf(fmaxf(tm[1][0], tm[1][1]), fmaxf(tm[1][2], tm[1][3]))};
  #pragma unroll
          for (int r = 0; r < 2; ++r) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
            const float mn = fmaxf(M[r], tmax[r]);
            scl[r] = (mn == -INFINITY) ? 1.f : ex2((M[r] - mn) * sc);
            mref[r] = (mn == -INFINITY) ? 12582912.f : 12582912.f + 120.f - 8.f * sc * mn;
            M[r] = mn;
          }
  #pragma unroll
          for (int bk = 0; bk < 2; ++bk) {
            uint32_t e[4][4];
  #pragma unroll
            for (int jj = 0; jj < 4; ++jj)
  #pragma unroll
              for (int c = 0; c < 4; ++c)
                e[jj][c] = __float_as_uint(fmaxf(fmaf(Sc[4 * (4 * bk + jj) + c], sc8, mref[c >> 1]), 12582912.f));
            pa[bk][0] = __byte_perm(__byte_perm(e[0][0], e[0][1], 0x0040), __byte_perm(e[1][0], e[1][1], 0x0040), 0x5410);
            pa[bk][1] = __byte_perm(__byte_perm(e[0][2], e[0][3], 0x0040), __byte_perm(e[1][2], e[1][3], 0x0040), 0x5410);
            pa[bk][2] = __byte_perm(__byte_perm(e[2][0], e[2][1], 0x0040), __byte_perm(e[3][0], e[3][1], 0x0040), 0x5410);
            pa[bk][3] = __byte_perm(__byte_perm(e[2][2], e[2][3], 0x0040), __byte_perm(e[3][2], e[3][3], 0x0040), 0x5410);
          }
        }
        // 3. the next stage's QK and the previous stage's PV are done: their raw / V^T slots are free; rescale O
        //    (unconditional)
        if (tid == CT0) tstamp(p, sx, 9);
        wg_wait0();
        fence_acc(Sn);
        fence_acc36(O);
        if (lane == 0 && !consonly) {
          if (prevs >= 0) mbar_arrive(vt_empty + prevs % WS_VD);
          if (rn && sn >= 0) mbar_arrive(raw_empty + sn % RD);
        }
#pragma unroll
        for (int j = 0; j < 9; ++j) { O[4 * j] *= scl[0]; O[4 * j + 1] *= scl[0]; O[4 * j + 2] *= scl[1]; O[4 * j + 3] *= scl[1]; }
        // 4. O += P V of this stage
        if (tid == CT0) tstamp(p, sx, 10);
        mbar_wait(!rc || nowait ? done_bar : sx < 0 ? new_full : vt_full + sx % WS_VD, (!rc || nowait || sx < 0) ? 0 : (sx / WS_VD) & 1);
        if (tid == CT0) tstamp(p, sx, 11);
        __syncwarp();
        {
          const int vx = rc ? sx : nst - 1;
          const unsigned char* vb = vx < 0 ? newvt : vts + (vx % WS_VD) * WSVT;
          wg_fence();
#pragma unroll
          for (int st = 0; st < 2; ++st) wgmma72_e4m3(O, pa[st], wg_desc(vb + WSVH * h + 2 * 9 * 128 * st, 9 * 128, 128));
          wg_commit();
        }
        prevs = (rc && sx >= 0) ? sx : -1;  // (its PV completes under the next step's QK and softmax)
      };
#pragma unroll 1
      for (int si = 0; si < ntot; si += 2) {
        step(SA, SB, si);
        step(SB, SA, si + 1);
      }
      wg_wait0();
      fence_acc36(O);
    }
    } else {
    int pva = -1, pvb = -1;  // V^T slots of the PV in flight (-1: none)
#pragma unroll 1
    for (int k = tail ? -1 : 0; k < nst; k += (k < 0 ? 1 : 2)) {
      const bool nw = k < 0, two = !nw && k + 1 < nst;
      const int ka = nw ? 0 : k, kb = nw ? 0 : k + 1;
      const unsigned char* ksa = nw ? newk : ring + (ka % RD) * WSRAW;
      // (no second stage: the pair's b half re-reads stage a, fully masked: P = 0; the wgmmas are never conditional)
      const unsigned char* ksb = two ? ring + (kb % RD) * WSRAW : ksa;
      const unsigned char* vsa = nw ? newvt : vts + (ka % WS_VD) * WSVT;
      const unsigned char* vsb = two ? vts + (kb % WS_VD) * WSVT : vsa;
      if (tid == CT0) tstamp(p, k, 6);
      // (branch-free waits: a conditional spin before the wgmmas makes ptxas serialize them; a single stage waits twice)
      mbar_wait(nw ? new_full : raw_full + ka % RD, nw ? 0 : (ka / RD) & 1);
      mbar_wait(two ? raw_full + kb % RD : nw ? new_full : raw_full + ka % RD, two ? (kb / RD) & 1 : nw ? 0 : (ka / RD) & 1);
      __syncwarp();  // lanes leave the spin at different times: wgmma (.aligned) needs the warp converged
      if (tid == CT0) tstamp(p, k, 7);
      float S[64];
      qk_issue<NQ, QSTR>(S, qsm, tid - CT0, qt, h, ksa);
      qk_issue<NQ, QSTR>(S + 32, qsm, tid - CT0, qt, h, ksb);
      wg_commit();
      wg_wait0();  // (also the previous iteration's PV)
      fence_acc(S);
      fence_acc(S + 32);
      fence_acc36(O);
      if (pva >= 0 && lane == 0) {
        mbar_arrive(vt_empty + pva);
        if (pvb >= 0) mbar_arrive(vt_empty + pvb);
      }
      if (tid == CT0) tstamp(p, k, 8);
      if (!nw && lane == 0) {
        mbar_arrive(raw_empty + ka % RD);
        if (two) mbar_arrive(raw_empty + kb % RD);
      }
      float tm[2][4] = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY}, {-INFINITY, -INFINITY, -INFINITY, -INFINITY}};  // 4 chains / row
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pos0 = nw ? seq : k0 + WTK * (k + hf), nvalid = nw ? (hf ? 0 : p.L) : (hf && !two) ? 0 : min(WTK, ncached - WTK * (k + hf));
        if (nvalid < WTK || pos0 < lo_max) {
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int kk = 8 * j + 2 * t + (c & 1), r = c >> 1, pos = pos0 + kk;
              const float v = (kk < nvalid && pos >= lo[r] && pos <= hi[r]) ? S[32 * hf + 4 * j + c] : -INFINITY;
              S[32 * hf + 4 * j + c] = v;
              tm[r][(j & 1) * 2 + (c & 1)] = fmaxf(tm[r][(j & 1) * 2 + (c & 1)], v);
            }
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) tm[c >> 1][(j & 1) * 2 + (c & 1)] = fmaxf(tm[c >> 1][(j & 1) * 2 + (c & 1)], S[32 * hf + 4 * j + c]);
        }
      }
      float tmax[2] = {fmaxf(fmaxf(tm[0][0], tm[0][1]), fmaxf(tm[0][2], tm[0][3])), fmaxf(fmaxf(tm[1][0], tm[1][1]), fmaxf(tm[1][2], tm[1][3]))};
      float mref[2], scl[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
        tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
        const float mn = fmaxf(M[r], tmax[r]);
        scl[r] = (mn == -INFINITY) ? 1.f : ex2((M[r] - mn) * sc);
        // e4m3 code of P = 2^(s sc - mn sc + 8) ~ 8 (s sc - mn sc + 8) + 56, rounded by the 1.5 * 2^23 magic number
        mref[r] = (mn == -INFINITY) ? 12582912.f : 12582912.f + 120.f - 8.f * sc * mn;
        M[r] = mn;
      }
      // (unconditional: a per-warp branch touching the accumulators makes ptxas serialize the warpgroup's wgmmas)
#pragma unroll
      for (int j = 0; j < 9; ++j) { O[4 * j] *= scl[0]; O[4 * j + 1] *= scl[0]; O[4 * j + 2] *= scl[1]; O[4 * j + 3] *= scl[1]; }
      const float sc8 = 8.f * sc;
      uint32_t pa[4][4];  // k blocks: stage a dims-of-keys 0..31, 32..63, then stage b
#pragma unroll
      for (int bk = 0; bk < 4; ++bk) {
        uint32_t e[4][4];  // [j - 4 (bk & 1)][c]: the code in the low byte (masked keys: -inf, clamped to code 0)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
#pragma unroll
          for (int c = 0; c < 4; ++c)
            e[jj][c] = __float_as_uint(fmaxf(fmaf(S[32 * (bk >> 1) + 4 * (4 * (bk & 1) + jj) + c], sc8, mref[c >> 1]), 12582912.f));
        pa[bk][0] = __byte_perm(__byte_perm(e[0][0], e[0][1], 0x0040), __byte_perm(e[1][0], e[1][1], 0x0040), 0x5410);
        pa[bk][1] = __byte_perm(__byte_perm(e[0][2], e[0][3], 0x0040), __byte_perm(e[1][2], e[1][3], 0x0040), 0x5410);
        pa[bk][2] = __byte_perm(__byte_perm(e[2][0], e[2][1], 0x0040), __byte_perm(e[3][0], e[3][1], 0x0040), 0x5410);
        pa[bk][3] = __byte_perm(__byte_perm(e[2][2], e[2][3], 0x0040), __byte_perm(e[3][2], e[3][3], 0x0040), 0x5410);
      }
      if (tid == CT0) tstamp(p, k, 9);
      mbar_wait(nw ? new_full : vt_full + ka % WS_VD, nw ? 0 : (ka / WS_VD) & 1);
      mbar_wait(two ? vt_full + kb % WS_VD : nw ? new_full : vt_full + ka % WS_VD, two ? (kb / WS_VD) & 1 : nw ? 0 : (ka / WS_VD) & 1);
      __syncwarp();
      if (tid == CT0) tstamp(p, k, 10);
      wg_fence();
#pragma unroll
      for (int st = 0; st < 2; ++st) wgmma72_e4m3(O, pa[st], wg_desc(vsa + WSVH * h + 2 * 9 * 128 * st, 9 * 128, 128));
#pragma unroll
      for (int st = 0; st < 2; ++st) wgmma72_e4m3(O, pa[2 + st], wg_desc(vsb + WSVH * h + 2 * 9 * 128 * st, 9 * 128, 128));
      wg_commit();
      if (tid == CT0) tstamp(p, k, 11);
      pva = nw ? -1 : ka % WS_VD;
      pvb = two ? kb % WS_VD : -1;
    }
    wg_wait0();
    fence_acc36(O);
    }
    stamp_t(p, 3, CT0);
    if constexpr (SMR) ws_setmaxnreg<0, REG_MID>();
  }

  // --- push the consumer rows into their owner ranks (row R -> rank R % CL, local row R / CL, source = this rank).
  //     O[4j + c] (c < 2: row lo, else row hi) is dim 8 j + 2 t + (c & 1): stored dims (t * 8 + j) * 2 + (c & 1)
  const int CLs = static_cast<int>(cluster.num_blocks()), crank = static_cast<int>(cluster.block_rank());
  const int RPCv = RV / CLs;
  const int npass = (RPCv * 16 + NT - 1) / NT;
  const int q = tid & 15;
  const int d0 = 16 * (q & 3) + 2 * (q >> 2);  // stored dims 4q .. 4q + 3 = natural d0, d0 + 1, d0 + 8, d0 + 9
  const int qkv_gate = (p.Hq + 2 * HKV) * 64;
  // the gates of this rank's output rows, requested before the cluster barriers (their latency hides behind the merge);
  // passes beyond WS_GP load theirs in the fold
  uint32_t ge[WS_GP][2];
  const bool early = WS_EARLY_GATES;
  if (early) {
    if (p.pdl) gdc_wait();  // (producer threads: before the gates and the output)
#pragma unroll
    for (int ps = 0; ps < WS_GP; ++ps) {
      const int orl = (NT / 16) * ps + (tid >> 4), orow = crank + CLs * orl;
      const int r0 = orl < RPCv ? orow : crank;
      const int r1 = (r0 >> 4) < p.L ? r0 : (r0 & 15);
      ge[ps][0] = ps < npass ? cld4(qkvb + (r1 >> 4) * QS + qkv_gate + (r1 & 15) * 64 + d0) : 0u;
      ge[ps][1] = ps < npass ? cld4(qkvb + (r1 >> 4) * QS + qkv_gate + (r1 & 15) * 64 + d0 + 8) : 0u;
    }
  }
  cluster_arrive_release();  // this rank is done with its ring (the peers push into it after the wait)
  cluster_wait_acquire();
  constexpr bool wsa = PP && NQ == 2;
  if (warp >= CB && (!PP || warp < CB + 8)) {  // (PP: the parity-0 warpgroups hold the merged rows)
    const int cw = warp - CB, h = cw >> 2, w4 = cw & 3, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float Lq = __shfl_sync(0xffffffffu, O[32 + 2 * r], lane & ~3);  // l of the row: column 64 at lane t = 0
      const int R = (2 * w4 + r) * 16 + h * 8 + g;
      float* dst = cluster.map_shared_rank(recv, R % CLs);
      const int sl = crank * RPCv + R / CLs;
      const float Mr = M[r] == -INFINITY ? -INFINITY : M[r] * sc;
      if (wsa) {
        // st.async: the bytes count on the owner's push_bar (no cluster-wide barrier after the push)
        const uint32_t rr = mapa_u32(smem_u32(recv), R % CLs), rbar = mapa_u32(smem_u32(push_bar), R % CLs);
        if (t == 0) st_async2(rr + 8 * sl, Mr, Lq, rbar);
        const uint32_t ro = rr + (2 * RV + sl * 64) * 4;
        if (f16push) {
          const float il = Lq > 0.f ? 1.f / Lq : 0.f;
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            const int j = 4 * jj;
            st_async4u(ro + (t * 16 + 8 * jj) * 2,
                       make_uint4(pack_f16x2(O[4 * j + 2 * r] * il, O[4 * j + 2 * r + 1] * il),
                                  pack_f16x2(O[4 * j + 4 + 2 * r] * il, O[4 * j + 5 + 2 * r] * il),
                                  pack_f16x2(O[4 * j + 8 + 2 * r] * il, O[4 * j + 9 + 2 * r] * il),
                                  pack_f16x2(O[4 * j + 12 + 2 * r] * il, O[4 * j + 13 + 2 * r] * il)), rbar);
          }
        } else {
#pragma unroll
          for (int j = 0; j < 8; j += 2)
            st_async4(ro + (t * 8 + j) * 2 * 4, make_float4(O[4 * j + 2 * r], O[4 * j + 2 * r + 1], O[4 * j + 4 + 2 * r], O[4 * j + 5 + 2 * r]), rbar);
        }
        continue;
      }
      if (t == 0) *reinterpret_cast<float2*>(dst + 2 * sl) = make_float2(Mr, Lq);
      float* od = dst + 2 * RV + sl * 64;
      if (f16push) {
        // O / l as f16 (stored dims 16 t .. 16 t + 15 in two 16-B stores: half the remote requests); the fold scales
        // by l again
        const float il = Lq > 0.f ? 1.f / Lq : 0.f;
        uint4 hv[2];
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int j = 4 * jj;
          hv[jj] = make_uint4(pack_f16x2(O[4 * j + 2 * r] * il, O[4 * j + 2 * r + 1] * il),
                              pack_f16x2(O[4 * j + 4 + 2 * r] * il, O[4 * j + 5 + 2 * r] * il),
                              pack_f16x2(O[4 * j + 8 + 2 * r] * il, O[4 * j + 9 + 2 * r] * il),
                              pack_f16x2(O[4 * j + 12 + 2 * r] * il, O[4 * j + 13 + 2 * r] * il));
        }
        uint4* hd = reinterpret_cast<uint4*>(reinterpret_cast<__half*>(od) + t * 16);
        hd[0] = hv[0];
        hd[1] = hv[1];
      } else {
#pragma unroll
        for (int j = 0; j < 8; j += 2)
          *reinterpret_cast<float4*>(od + (t * 8 + j) * 2) = make_float4(O[4 * j + 2 * r], O[4 * j + 2 * r + 1], O[4 * j + 4 + 2 * r], O[4 * j + 5 + 2 * r]);
      }
    }
  }
  if (wsa) {
    mbar_wait_cluster(smem_u32(push_bar), 0);  // every peer's rows landed in this rank
  } else {
    cluster_arrive_release();
    cluster_wait_acquire();
  }
  stamp(p, 5);
  // --- fold: slot (local row orl, piece q) over the CL sources; then across clusters per row (last-arriving owner);
  //     pass ps covers local rows 24 ps .. 24 ps + 23 (thread -> local row 24 ps + (tid >> 4)); uniform barriers only
  if (!early && p.pdl) gdc_wait();  // (producer threads: before the gates and the output)
  const int ncl = p.Stok / CLs, cid = s / CLs;
  if (!PROBE_MODE(1 << 23) && ncl > 1 && ncl <= 4 && RPCv * 4 <= NT) {  // (one fold pass for 2-4 clusters per request)
    // one pass: thread -> (local row orl = tid >> 2, quarter qq: stored dims 16 qq .. 16 qq + 15 = natural
    // 16 i + 2 qq + {0, 1, 8, 9}, i = 0..3); the cross-cluster round trips (partial, ticket, partials) happen once
    const int orl = tid >> 2, qq = tid & 3, orow = crank + CLs * orl;
    const bool oact = orl < RPCv;
    const int orw = oact ? orow : crank;
    uint32_t gw[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (gsmode) {
        const unsigned char* gr = gsm + (oact ? orl : 0) * 128 + (16 * i + 2 * qq) * 2;
        gw[i][0] = *reinterpret_cast<const uint32_t*>(gr);
        gw[i][1] = *reinterpret_cast<const uint32_t*>(gr + 16);
      } else {
        const int orw1 = (orw >> 4) < p.L ? orw : (orw & 15);
        const __nv_bfloat16* gr = qkvb + (orw1 >> 4) * QS + qkv_gate + (orw1 & 15) * 64 + 16 * i + 2 * qq;
        gw[i][0] = cld4(gr);
        gw[i][1] = cld4(gr + 8);
      }
    }
    float Mf = -INFINITY, Ls = 0.f;
    float4 acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (oact) {
#pragma unroll 1
      for (int r = 0; r < CLs; ++r) {
        const int sl = r * RPCv + orl;
        const float2 ml = *reinterpret_cast<const float2*>(recv + 2 * sl);
        float4 o4[4];
        if (f16push) {
          const uint4* hs = reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(recv + 2 * RV + sl * 64) + 16 * qq);
          const uint4 h0 = hs[0], h1 = hs[1];
          o4[0] = unpack4h(h0.x, h0.y, ml.y); o4[1] = unpack4h(h0.z, h0.w, ml.y);
          o4[2] = unpack4h(h1.x, h1.y, ml.y); o4[3] = unpack4h(h1.z, h1.w, ml.y);
        } else {
          const float4* src = reinterpret_cast<const float4*>(recv + 2 * RV + sl * 64 + 16 * qq);
          o4[0] = src[0]; o4[1] = src[1]; o4[2] = src[2]; o4[3] = src[3];
        }
        fold16(Mf, Ls, acc, ml.x, ml.y, o4);
      }
    }
    bool emit = oact;
    if (ncl > 1) {
      // ticket first (relaxed, low half of the row's word): the first ncl - 1 arrivers publish their partial and count
      // it in the high half (release); the last waits for that count (acquire) and pulls theirs - its own partial is
      // never written. It waits only for rows whose tickets are already taken, and publishing never waits, so no
      // co-residency is needed (rows of one CTA may get different roles: every CTA publishes before it waits)
      float* gml = p.grp_ml + static_cast<size_t>(rb) * ncl * RV * 2;
      float* go = p.grp_o + static_cast<size_t>(rb) * ncl * RV * 64;
      int* tk = p.cnt + rb * RV + orow;
      if (oact && qq == 0) s_last[orl] = (atom_add_relaxed(tk, 1) & 0xffff) == ncl - 1;
      __syncthreads();
      emit = oact && s_last[orl];
      if (oact && !emit) {
        if (qq == 0) *reinterpret_cast<float2*>(gml + (cid * RV + orow) * 2) = make_float2(Mf, Ls);
        float4* dst = reinterpret_cast<float4*>(go + (cid * RV + orow) * 64 + 16 * qq);
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = acc[i];
      }
      __syncthreads();
      if (oact && !emit && qq == 0) red_add_release(tk, 1 << 16);
      if (emit) {
        while ((ld_acquire(tk) >> 16) < ncl - 1) {
        }
        // fold in cluster order 0 .. ncl - 1 with this rank's own partial at its index: the same association whichever
        // cluster arrives last (bitwise-deterministic output)
        float2 ml[4];
        float4 o4[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const bool src = r < ncl && r != cid;
          ml[r] = src ? __ldcg(reinterpret_cast<const float2*>(gml + (r * RV + orow) * 2))
                      : r == cid ? make_float2(Mf, Ls) : make_float2(-INFINITY, 0.f);
          const float4* sp = reinterpret_cast<const float4*>(go + (r * RV + orow) * 64 + 16 * qq);
#pragma unroll
          for (int i = 0; i < 4; ++i) o4[r][i] = src ? __ldcg(sp + i) : r == cid ? acc[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        Mf = -INFINITY; Ls = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int r = 0; r < 4; ++r) fold16(Mf, Ls, acc, ml[r].x, ml[r].y, o4[r]);
        __syncwarp(0xfu << (lane & 28));  // the row's 4 threads are past their waits
        if (qq == 0) *tk = 0;  // every owner of this row has arrived and published: no later access this launch
      }
    }
    if (emit && (orow >> 4) < p.L) {  // (rows of missing tokens: merged for the tickets, never written)
      // output of row orow: O / l * v_scale -> rmsnorm over 64 dims (the 4 quarter threads) -> * sigmoid(gate)
      const float inv = (Ls > 0.f ? 1.f / Ls : 0.f) / p.inv_v_scale;
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[i].x *= inv; acc[i].y *= inv; acc[i].z *= inv; acc[i].w *= inv;
        ss += acc[i].x * acc[i].x + acc[i].y * acc[i].y + acc[i].z * acc[i].z + acc[i].w * acc[i].w;
      }
      const unsigned qmask = 0xfu << (lane & 28);  // the 4 quarter threads of this row
      ss += __shfl_xor_sync(qmask, ss, 1);
      ss += __shfl_xor_sync(qmask, ss, 2);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      __nv_bfloat16* outp = p.out + (static_cast<size_t>(rb) * p.L * 16 + orow) * 64 + 2 * qq;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[i][0]));
        const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[i][1]));
        *reinterpret_cast<__nv_bfloat162*>(outp + 16 * i) =
            __floats2bfloat162_rn(__fdividef(acc[i].x * rn, 1.f + __expf(-ga.x)), __fdividef(acc[i].y * rn, 1.f + __expf(-ga.y)));
        *reinterpret_cast<__nv_bfloat162*>(outp + 16 * i + 8) =
            __floats2bfloat162_rn(__fdividef(acc[i].z * rn, 1.f + __expf(-gb.x)), __fdividef(acc[i].w * rn, 1.f + __expf(-gb.y)));
      }
    }
    stamp(p, 6);
    return;
  }
#pragma unroll 1
  for (int ps = 0; ps < npass; ++ps) {
    const int orl = (NT / 16) * ps + (tid >> 4), orow = crank + CLs * orl;
    const bool oact = orl < RPCv;
    const int otok = (oact ? orow : crank) >> 4, ohead = (oact ? orow : crank) & 15;
    uint32_t gate0, gate1;
    if (gsmode) {
      const unsigned char* gr = gsm + (oact ? orl : 0) * 128 + d0 * 2;
      gate0 = *reinterpret_cast<const uint32_t*>(gr);
      gate1 = *reinterpret_cast<const uint32_t*>(gr + 16);
    } else if (early && ps < WS_GP) {
      gate0 = ge[0][0]; gate1 = ge[0][1];
#pragma unroll
      for (int k = 1; k < WS_GP; ++k)
        if (ps == k) { gate0 = ge[k][0]; gate1 = ge[k][1]; }
    } else {
      const int gtok = otok < p.L ? otok : 0;
      gate0 = cld4(qkvb + gtok * QS + qkv_gate + ohead * 64 + d0);
      gate1 = cld4(qkvb + gtok * QS + qkv_gate + ohead * 64 + d0 + 8);
    }
    float Mf = -INFINITY, Ls = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (oact) {
#pragma unroll 1
      for (int r = 0; r < CLs; ++r) {  // rolled: shared-memory sources (the fold's code is fetched cold)
        const int sl = r * RPCv + orl;
        const float2 ml = *reinterpret_cast<const float2*>(recv + 2 * sl);
        float4 o4;
        if (f16push) {
          const uint2 hq = *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(recv + 2 * RV + sl * 64) + 4 * q);
          o4 = unpack4h(hq.x, hq.y, ml.y);
        } else {
          o4 = *reinterpret_cast<const float4*>(recv + 2 * RV + sl * 64 + 4 * q);
        }
        fold(Mf, Ls, acc, ml.x, ml.y, o4);
      }
    }
    bool emit = oact;
    if (ncl > 1) {
      float* gml = p.grp_ml + static_cast<size_t>(rb) * ncl * RV * 2;
      float* go = p.grp_o + static_cast<size_t>(rb) * ncl * RV * 64;
      if (oact) {
        if (q == 0) *reinterpret_cast<float2*>(gml + (cid * RV + orow) * 2) = make_float2(Mf, Ls);
        *reinterpret_cast<float4*>(go + (cid * RV + orow) * 64 + 4 * q) = acc;
      }
      __syncthreads();
      if (oact && q == 0) s_last[tid >> 4] = atom_add_acq_rel(&p.cnt[rb * RV + orow], 1) == ncl - 1;
      __syncthreads();
      emit = oact && s_last[tid >> 4];
      if (emit) {
        float2 ml[GS];
        float4 o4[GS];
#pragma unroll
        for (int r = 0; r < GS; ++r) {
          ml[r] = r < ncl ? __ldcg(reinterpret_cast<const float2*>(gml + (r * RV + orow) * 2)) : make_float2(-INFINITY, 0.f);
          o4[r] = r < ncl ? __ldcg(reinterpret_cast<const float4*>(go + (r * RV + orow) * 64 + 4 * q)) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        Mf = -INFINITY; Ls = 0.f;
        acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int r = 0; r < GS; ++r) fold(Mf, Ls, acc, ml[r].x, ml[r].y, o4[r]);
        if (q == 0) p.cnt[rb * RV + orow] = 0;  // every owner of this row has arrived: no later access this launch
      }
      __syncthreads();  // s_last is reused by the next pass
    }
    if (emit && otok < p.L) {  // (rows of missing tokens: merged for the tickets, never written)
      // output of row orow (token otok, head ohead): O / l * v_scale -> rmsnorm over 64 dims -> * sigmoid(gate)
      const float inv = (Ls > 0.f ? 1.f / Ls : 0.f) / p.inv_v_scale;
      const float d[4] = {acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv};
      float ss = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
      const unsigned grpmask = 0xffffu << (lane & 16);
#pragma unroll
      for (int o2 = 8; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(grpmask, ss, o2);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate0));
      const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate1));
      __nv_bfloat16* outp = p.out + (static_cast<size_t>(rb) * p.L * 16 + orow) * 64;
      *reinterpret_cast<__nv_bfloat162*>(outp + d0) =
          __floats2bfloat162_rn(__fdividef(d[0] * rn, 1.f + __expf(-ga.x)), __fdividef(d[1] * rn, 1.f + __expf(-ga.y)));
      *reinterpret_cast<__nv_bfloat162*>(outp + d0 + 8) =
          __floats2bfloat162_rn(__fdividef(d[2] * rn, 1.f + __expf(-gb.x)), __fdividef(d[3] * rn, 1.f + __expf(-gb.y)));
    }
  }
  stamp(p, 6);
}

// ===================================== batch q_len 8 on private group rings ==========================================
// CTA = (request rb, split s) of Stok per request. 8 warps = 2 groups x 4 warps. Group grp streams the CTA's 16-key tiles
// grp, grp + 2, ... through its own cp.async ring (RD - 1 tiles in flight: the whole share on sliding layers); each tile
// is loaded by the group's 4 warps (a quarter each) and consumed by all 4 after one named barrier. Warp wi of a group
// owns kv head h = wi & 1 and tokens 4 (wi >> 1) .. + 3: two m-tiles (rows g / g + 8 = tokens 2 mt / 2 mt + 1, query
// head 8 h + g); a tile's K / V fragments are converted once and shared by both. The tail CTA's group 0 first attends the
// 8 new tokens as one more tile (causal mask), while the gathers fly. After the loop the groups' row partials are folded
// in smem; the request's CTAs form one cluster: every rank pushes each row partial to the row's owner (DSMEM), and the
// owner folds them and writes the output (rms norm, sigmoid gate prefetched into registers).
constexpr int VB_PSTR = 68;  // smem partial row (floats): m, l, 2 pad, 64 natural dims
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
// one 16-key tile, one m-tile: rows g (token 2 mt) / g + 8 (token 2 mt + 1) of kv head h's query head g
__device__ __forceinline__ void vb_consume(Acc8& A, const uint32_t qa[8], const uint32_t qb[8], const uint32_t kb[2][4][2],
                                           const uint32_t vb[8][2], bool edge, int pos0, int nvalid, const int lo[2], const int hi[2]) {
  const int t = threadIdx.x & 3;
  float sacc[2][4];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    sacc[u][0] = sacc[u][1] = sacc[u][2] = sacc[u][3] = 0.f;
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint32_t a[4] = {qa[2 * st], qb[2 * st], qa[2 * st + 1], qb[2 * st + 1]};
      mma_f16(sacc[u], a, kb[u][st][0], kb[u][st][1]);
    }
  }
  float tmax[2] = {-INFINITY, -INFINITY};
  if (edge) {
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int kk = 8 * u + 2 * t + (c & 1), r = c >> 1;
        const float v = (kk < nvalid && pos0 + kk >= lo[r] && pos0 + kk <= hi[r]) ? sacc[u][c] : -INFINITY;  // q has log2 e
        sacc[u][c] = v;
        tmax[r] = fmaxf(tmax[r], v);
      }
  } else {
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
      for (int c = 0; c < 4; ++c) tmax[c >> 1] = fmaxf(tmax[c >> 1], sacc[u][c]);
  }
  float mref[2], sc[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
    tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
    const float mn = fmaxf(A.m[r], tmax[r]);
    mref[r] = (mn == -INFINITY) ? 0.f : mn;  // a row with no valid key yet: every term below is ex2(-inf) = 0
    sc[r] = ex2(A.m[r] - mref[r]);
    A.m[r] = mn;
    A.l[r] *= sc[r];
  }
  if (__any_sync(0xffffffffu, sc[0] != 1.f || sc[1] != 1.f)) {
#pragma unroll
    for (int j = 0; j < 8; ++j) { A.o[j][0] *= sc[0]; A.o[j][1] *= sc[0]; A.o[j][2] *= sc[1]; A.o[j][3] *= sc[1]; }
  }
  float pv[2][4];
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float e = ex2(sacc[u][c] - mref[c >> 1]);
      pv[u][c] = e;
      A.l[c >> 1] += e;
    }
  const uint32_t pa[4] = {pack_f16x2(pv[0][0], pv[0][1]), pack_f16x2(pv[0][2], pv[0][3]), pack_f16x2(pv[1][0], pv[1][1]),
                          pack_f16x2(pv[1][2], pv[1][3])};
#pragma unroll
  for (int j = 0; j < 8; ++j) mma_f16(A.o[j], pa, vb[j][0], vb[j][1]);
}

template <int RD>
__global__ void __launch_bounds__(NW * 32, 1) b1_vb_kernel(const P p) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* qs = reinterpret_cast<__half*>(smem);   // [128][64] f16 queries (scaled, log2 units)
  unsigned char* ntile = smem + RV * 128;         // the 8 new tokens as one ring tile (keys 8..15 zero)
  unsigned char* ring = ntile + TILE;             // [2][RD][TILE]; after the loop: pt [128][VB_PSTR], recv [CL][128/CL][VB_PSTR]
  int* ids = reinterpret_cast<int*>(ring + (2 * RD * TILE > 2 * RV * VB_PSTR * 4 ? 2 * RD * TILE : 2 * RV * VB_PSTR * 4));
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int grp = warp >> 2, wi = warp & 3, h = wi & 1, th = wi >> 1;
  const int rb = blockIdx.x / p.Stok, s = blockIdx.x - rb * p.Stok;  // one cluster per request: rank s
  const int CLs = p.Stok, RPC = RV / CLs;  // rows owned per rank (output)
  const int QS = (2 * p.Hq + 2 * HKV) * 64, qkv_gate = (p.Hq + 2 * HKV) * 64;
  const __nv_bfloat16* qkv = p.qkv + static_cast<size_t>(rb) * LV * QS;
  const int32_t* ptab = p.page_table + static_cast<size_t>(rb) * p.max_pages;
  const bool tailcta = s == p.Stok - 1;
  stamp(p, 0);
  const int seq = p.seq_lens[rb];
  const int part = tid & 15, prow = tid >> 4;  // (row, 4-dim piece) layout: 16 rows x 16 pieces per pass
  const uint2 ifr_raw = vld8(p.inv_freq + 2 * part);
  const float2 ifr = make_float2(__uint_as_float(ifr_raw.x), __uint_as_float(ifr_raw.y));
  const int ntok = tid >> 5, nkv = (tid >> 4) & 1;  // new-token prep: (token, kv head, piece)
  long long loc = 0;
  if (tailcta) loc = vld64(p.cache_loc + rb * LV + ntok);
  // the step's qkv rows (queries; gates of the rows this rank outputs; new tokens): standalone requested first (ahead of
  // the K/V burst); with PDL they come from the previous kernel and are read (coherently) after griddepcontrol.wait
  uint2 qin[LV];
  uint32_t gq[LV][2];
  uint2 kin = make_uint2(0, 0), vin = kin;
  auto load_qkv = [&]() {
#pragma unroll
    for (int u = 0; u < LV; ++u) qin[u] = cld8(qkv + u * QS + prow * 64 + 4 * part);
#pragma unroll
    for (int ps = 0; ps < LV; ++ps) {
      const int R = s * RPC + ps * 16 + prow;  // owned row of output pass ps
      if (ps * 16 + prow < RPC) {
        const uint2 gw = cld8(qkv + (R >> 4) * QS + qkv_gate + (R & 15) * 64 + 4 * part);
        gq[ps][0] = gw.x; gq[ps][1] = gw.y;
      }
    }
    if (tailcta) {
      kin = cld8(qkv + ntok * QS + p.Hq * 64 + nkv * 64 + 4 * part);
      vin = cld8(qkv + ntok * QS + p.Hq * 64 + HKV * 64 + nkv * 64 + 4 * part);
    }
  };
  if (!p.pdl) load_qkv();

  // --- the CTA's keys: cached positions [k0, k1) of the union of the 8 windows; page ids -> smem
  const int kcap = p.max_pages << p.page_shift, pmask = (1 << p.page_shift) - 1;
  int k0 = s * p.chunk;
  if (p.window) k0 += max(0, seq - p.window + 1);
  if (PROBE_MODE(1 << 30)) {  // (probe mode bit 30: the serial loop, one DRAM trip per NW * 32 ids)
#pragma unroll 1
    for (int i = tid; i < p.chunk; i += NW * 32) {
      const int key = k0 + i;
      ids[i] = key < kcap ? (ptab[key >> p.page_shift] << p.page_shift) | (key & pmask) : 0;
    }
  } else {
    load_ids<NW * 32>(ids, ptab, k0, p.chunk, kcap, p.page_shift, tid);
  }
  const int k1 = min(seq, k0 + p.chunk);
  const int ncached = max(0, k1 - k0), ntiles = (ncached + 15) >> 4;
  const int nmine = ntiles > grp ? (ntiles - grp + 1) >> 1 : 0;  // this group's tiles: grp, grp + 2, ...
  __syncthreads();  // ids are visible
  unsigned char* gring = ring + grp * RD * TILE;
  // tile k of the group -> slot k % RD (16 whole 128-B rows of K, then of V; 16-B chunk swizzle K: ch ^ 4 (key & 1),
  // V: ch ^ 2 ((key >> 1) & 3)); warp wi issues copies 64 wi .. 64 wi + 63 of the tile's 256
  auto issue = [&](int k) {
    if (k < nmine && !PROBE_MODE(2)) {
      unsigned char* slot = gring + (k % RD) * TILE;
      const int base = 16 * (grp + 2 * k);
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int it = wi * 64 + r * 32 + lane;
        const int tensor = it >> 7, key = (it >> 3) & 15, ch = it & 7;
        const int idx = base + key;
        const bool valid = idx < ncached;
        const size_t row = valid ? static_cast<size_t>(ids[idx]) : 0;
        const int sw = tensor ? (((key >> 1) & 3) << 1) : ((key & 1) << 2);
        cp_async16(slot + tensor * (TILE / 2) + key * ROWB + ((ch ^ sw) << 4), (tensor ? p.v_cache : p.k_cache) + row * ROWB + ch * 16,
                   valid ? 16 : 0);
      }
    }
    cp_async_commit();
  };
#pragma unroll 1
  for (int k = 0; k < RD - 1; ++k) issue(k);
  stamp(p, 1);
  gdc_launch();
  gdc_wait();
  if (p.pdl) load_qkv();
  // --- queries -> smem: rms * qk_scale (RoPE at seq + token on sliding) -> bf16 -> * sm_scale * k_scale * log2 e -> f16
#pragma unroll
  for (int u = 0; u < LV; ++u) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qin[u].x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qin[u].y));
    float v[4] = {a.x, a.y, b.x, b.y};
    if (!PROBE_MODE(1)) norm_rope4(v, p.qk_scale, p.rope, static_cast<float>(seq + u), ifr);
    *reinterpret_cast<uint2*>(qs + (u * 16 + prow) * 64 + 4 * part) =
        make_uint2(pack_f16x2(bf16_round(v[0]) * p.q_mult, bf16_round(v[1]) * p.q_mult),
                   pack_f16x2(bf16_round(v[2]) * p.q_mult, bf16_round(v[3]) * p.q_mult));
  }
  // --- the new tokens (tail CTA): key normalized (RoPE at seq + i on sliding) and quantized, value quantized; rows to
  //     the cache (single writer; no CTA of this launch reads them) and into the new-token tile (ring layout)
  if (tailcta) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kin.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kin.y));
    float k4[4] = {a.x, a.y, b.x, b.y};
    if (!PROBE_MODE(1)) norm_rope4(k4, 1.f, p.rope, static_cast<float>(seq + ntok), ifr);
    const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vin.x));
    const float2 d = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vin.y));
    const float ik = p.inv_k_scale, iv = p.inv_v_scale;
    const uint32_t kq = fp8x4(k4[0] * ik, k4[1] * ik, k4[2] * ik, k4[3] * ik);
    const uint32_t vq = fp8x4(c.x * iv, c.y * iv, d.x * iv, d.y * iv);
    const int off = nkv * 64 + 4 * part, ch = off >> 4;  // byte offset in the row; its 16-B chunk
    *reinterpret_cast<uint32_t*>(p.k_cache + static_cast<size_t>(loc) * ROWB + off) = kq;
    *reinterpret_cast<uint32_t*>(p.v_cache + static_cast<size_t>(loc) * ROWB + off) = vq;
    *reinterpret_cast<uint32_t*>(ntile + ntok * ROWB + ((ch ^ ((ntok & 1) << 2)) << 4) + (off & 15)) = kq;
    *reinterpret_cast<uint32_t*>(ntile + TILE / 2 + ntok * ROWB + ((ch ^ (((ntok >> 1) & 3) << 1)) << 4) + (off & 15)) = vq;
    *reinterpret_cast<uint32_t*>(ntile + (8 + ntok) * ROWB + 4 * (tid & 31)) = 0u;  // keys 8..15: zero (finite V)
    *reinterpret_cast<uint32_t*>(ntile + TILE / 2 + (8 + ntok) * ROWB + 4 * (tid & 31)) = 0u;
  }
  __syncthreads();  // queries and the new-token tile are visible
  // this warp's query fragments: m-tile m -> tokens 2 (2 th + m) (qa) / + 1 (qb), head 8 h + g, dims 16 t .. 16 t + 15
  uint32_t qa[2][8], qb[2][8];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const __half* src = qs + ((2 * (2 * th + m) + r) * 16 + 8 * h + g) * 64 + 16 * t;
      const uint4 x = *reinterpret_cast<const uint4*>(src), y = *reinterpret_cast<const uint4*>(src + 8);
      uint32_t* d = r ? qb[m] : qa[m];
      d[0] = x.x; d[1] = x.y; d[2] = x.z; d[3] = x.w; d[4] = y.x; d[5] = y.y; d[6] = y.z; d[7] = y.w;
    }
  int lo[2][2], hi[2][2];  // per row: the token's window [lo, hi] (cached keys always satisfy hi; new keys: causal)
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int tok = 2 * (2 * th + m) + r;
      lo[m][r] = p.window ? seq + tok - p.window + 1 : INT_MIN;
      hi[m][r] = seq + tok;
    }
  const int lomax = lo[1][1];
  Acc8 A[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int j = 0; j < 8; ++j) A[m].o[j][0] = A[m].o[j][1] = A[m].o[j][2] = A[m].o[j][3] = 0.f;
    A[m].m[0] = A[m].m[1] = -INFINITY;
    A[m].l[0] = A[m].l[1] = 0.f;
  }
  // one ring-layout tile (16 keys at positions pos0 ..) for both m-tiles: fragments converted once
  auto tile = [&](const unsigned char* slot, int pos0, int nvalid, bool edge) {
    uint32_t kb[2][4][2], vb[8][2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const uint4 kr = *reinterpret_cast<const uint4*>(slot + (8 * u + g) * ROWB + (((4 * h + t) ^ ((g & 1) << 2)) << 4));
      const uint32_t kw[4] = {kr.x, kr.y, kr.z, kr.w};
#pragma unroll
      for (int st = 0; st < 4; ++st) { kb[u][st][0] = e4m3x2_to_f16x2(kw[st]); kb[u][st][1] = e4m3x2_to_f16x2(kw[st] >> 16); }
    }
    {
      uint2 vr[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int key = (i < 2) ? 2 * t + i : 2 * t + 8 + (i - 2);
        const int ch = (4 * h + (g >> 1)) ^ (((key >> 1) & 3) << 1);
        vr[i] = *reinterpret_cast<const uint2*>(slot + TILE / 2 + key * ROWB + (ch << 4) + (g & 1) * 8);
      }
      const uint32_t vw[4][2] = {{vr[0].x, vr[0].y}, {vr[1].x, vr[1].y}, {vr[2].x, vr[2].y}, {vr[3].x, vr[3].y}};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t sel = (j & 3) | ((4 + (j & 3)) << 4);  // byte j of each key's 8 bytes -> (key 2t(+8), key 2t+1(+9))
        vb[j][0] = e4m3x2_to_f16x2(__byte_perm(vw[0][j >> 2], vw[1][j >> 2], sel));
        vb[j][1] = e4m3x2_to_f16x2(__byte_perm(vw[2][j >> 2], vw[3][j >> 2], sel));
      }
    }
    if (!PROBE_MODE(4)) {
#pragma unroll
      for (int m = 0; m < 2; ++m) vb_consume(A[m], qa[m], qb[m], kb, vb, edge, pos0, nvalid, lo[m], hi[m]);
    }
  };
  if (tailcta && grp == 0) tile(ntile, seq, LV, true);  // the new tokens first, while the gathers fly

  // --- main loop: tile k landed (own copies, then the group's barrier), refill the slot of tile k - 1, consume
#pragma unroll 1
  for (int k = 0; k < nmine; ++k) {
    cp_async_wait<RD - 2>();
    named_bar(1 + grp, 128);  // every warp's copies of tile k landed; every warp is done with tile k - 1
    issue(k + RD - 1);        // -> slot (k - 1) % RD
    const int tk = grp + 2 * k, pos0 = k0 + 16 * tk;
    const int nvalid = min(16, ncached - 16 * tk);
    tile(gring + (k % RD) * TILE, pos0, nvalid, nvalid < 16 || pos0 < lomax);
  }
  cp_async_wait<0>();
  stamp(p, 2);
  __syncthreads();  // every group is done with its ring: the area takes the row partials
  // group 1's row partials -> smem; group 0 (same rows per warp index) folds them into its registers and writes the CTA
  // partial back in place: pt [128 rows][VB_PSTR] = m, l, 2 pad, 64 natural dims
  float* pt = reinterpret_cast<float*>(ring);
#pragma unroll 1
  for (int phase = 0; phase < 2; ++phase) {
    if (grp == 1 - phase) {
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          float l = A[m].l[r];
          l += __shfl_xor_sync(0xffffffffu, l, 1);
          l += __shfl_xor_sync(0xffffffffu, l, 2);
          float* row = pt + ((2 * (2 * th + m) + r) * 16 + 8 * h + g) * VB_PSTR;
          float* od = row + 4 + 16 * t;  // lane (g, t) holds dims 16 t + j (o[j][2 r]) and 16 t + 8 + j (o[j][2 r + 1])
          float mm = A[m].m[r];
          if (phase == 1) {  // fold group 1's partial of this row (scale both into the common max)
            const float m1 = row[0], l1 = row[1];
            const float mn = fmaxf(mm, m1);
            const float a = mm == -INFINITY ? 0.f : exp2f(mm - mn), b = m1 == -INFINITY ? 0.f : exp2f(m1 - mn);
            const float4 x0 = *reinterpret_cast<const float4*>(od), x1 = *reinterpret_cast<const float4*>(od + 4);
            const float4 x2 = *reinterpret_cast<const float4*>(od + 8), x3 = *reinterpret_cast<const float4*>(od + 12);
            const float xs[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              A[m].o[j][2 * r] = A[m].o[j][2 * r] * a + xs[j] * b;
              A[m].o[j][2 * r + 1] = A[m].o[j][2 * r + 1] * a + xs[8 + j] * b;
            }
            l = l * a + l1 * b;
            mm = mn;
            __syncwarp();  // every lane of the quad read row[0], row[1] before lane t 0 rewrites them
          }
          if (t == 0) { row[0] = mm; row[1] = l; }
          *reinterpret_cast<float4*>(od) = make_float4(A[m].o[0][2 * r], A[m].o[1][2 * r], A[m].o[2][2 * r], A[m].o[3][2 * r]);
          *reinterpret_cast<float4*>(od + 4) = make_float4(A[m].o[4][2 * r], A[m].o[5][2 * r], A[m].o[6][2 * r], A[m].o[7][2 * r]);
          *reinterpret_cast<float4*>(od + 8) =
              make_float4(A[m].o[0][2 * r + 1], A[m].o[1][2 * r + 1], A[m].o[2][2 * r + 1], A[m].o[3][2 * r + 1]);
          *reinterpret_cast<float4*>(od + 12) =
              make_float4(A[m].o[4][2 * r + 1], A[m].o[5][2 * r + 1], A[m].o[6][2 * r + 1], A[m].o[7][2 * r + 1]);
        }
    }
    __syncthreads();
  }
  stamp(p, 3);
  // --- across the request's cluster: every rank pushes row R's partial into rank R / RPC's receive buffer (DSMEM);
  //     the owner folds the CL partials of its rows and writes the output. recv [CL][RPC][VB_PSTR] follows pt.
  float* recv = pt + RV * VB_PSTR;
  if (CLs > 1) {
    cluster_arrive_release();
    cluster_wait_acquire();  // every rank is done with its ring area (recv aliases it)
#pragma unroll 1
    for (int ps = 0; ps < RV / 16; ++ps) {
      const int R = ps * 16 + prow, own = R / RPC;
      const float* src = pt + R * VB_PSTR;
      float* dst = cluster.map_shared_rank(recv, own) + (s * RPC + R - own * RPC) * VB_PSTR;
      if (part == 0) *reinterpret_cast<float2*>(dst) = *reinterpret_cast<const float2*>(src);
      *reinterpret_cast<float4*>(dst + 4 + 4 * part) = *reinterpret_cast<const float4*>(src + 4 + 4 * part);
    }
    cluster_arrive_release();
    cluster_wait_acquire();  // every rank's pushes into this rank are visible; nobody touches remote smem after this
  }
  stamp(p, 4);
  // output of row R (token R >> 4, head R & 15), piece part: O / l * v_scale -> rmsnorm over 64 dims -> * sigmoid(gate)
  const unsigned grpmask = 0xffffu << (lane & 16);  // the 16 piece threads of a row share a half-warp
#pragma unroll
  for (int ps = 0; ps < LV; ++ps) {
    if (ps * 16 >= RPC) break;
    const int rr = ps * 16 + prow, R = s * RPC + rr;
    if (rr >= RPC) continue;  // CL 16: 8 rows per rank (half-warp uniform)
    float M = -INFINITY, Ls = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
    for (int src = 0; src < CLs; ++src) {
      const float* row = CLs > 1 ? recv + (src * RPC + rr) * VB_PSTR : pt + R * VB_PSTR;
      fold(M, Ls, acc, row[0], row[1], *reinterpret_cast<const float4*>(row + 4 + 4 * part));
    }
    const float inv = (Ls > 0.f ? 1.f / Ls : 0.f) / p.inv_v_scale;
    const float d[4] = {acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv};
    float ss = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
#pragma unroll
    for (int o2 = 8; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(grpmask, ss, o2);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gq[ps][0]));
    const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gq[ps][1]));
    __nv_bfloat162 o0 = __floats2bfloat162_rn(d[0] * rn / (1.f + expf(-ga.x)), d[1] * rn / (1.f + expf(-ga.y)));
    __nv_bfloat162 o1 = __floats2bfloat162_rn(d[2] * rn / (1.f + expf(-gb.x)), d[3] * rn / (1.f + expf(-gb.y)));
    *reinterpret_cast<uint2*>(p.out + (static_cast<size_t>(rb) * RV + R) * 64 + 4 * part) =
        make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));
  }
  stamp(p, 5);
}
}  // namespace okcb1

using namespace okcb1;

static char g_b1_msg[256];
static const char* g_b1_err = nullptr;
#define B1_CHECK(cond, ...) do { if (!(cond)) { snprintf(g_b1_msg, sizeof(g_b1_msg), __VA_ARGS__); g_b1_err = g_b1_msg; return; } } while (0)

static bool g_b1_pdl = false;  // decoder chains: launch with programmatic stream serialization
void attn_b1_set_pdl(bool on) { g_b1_pdl = on; }

template <int RD>
static void launch_b1(const P& p, int cl, cudaStream_t st) {
  const int bytes = ROWS * 64 * 2 + TILE + RECV_BYTES + NW * RD * TILE + ((p.chunk * 4 + 15) & ~15) + 16;
  static int optin = 0;
  if (!optin) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    optin -= 1024;  // headroom for the static shared variables
    cudaFuncSetAttribute(b1_decode_kernel<RD>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    cudaFuncSetAttribute(b1_decode_kernel<RD>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  B1_CHECK(bytes <= optin, "attn_b1: shared memory %d > %d (chunk %d, RD %d)", bytes, optin, (int)p.chunk, RD);
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(p.S);
  c.blockDim = dim3(NW * 32);
  c.dynamicSmemBytes = bytes;
  c.stream = st;
  cudaLaunchAttribute a[2];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = cl; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  a[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a[1].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = a;
  c.numAttrs = g_b1_pdl ? 2 : 1;
  P q = p;
  q.pdl = g_b1_pdl ? 1 : 0;
  B1_CHECK(cudaLaunchKernelEx(&c, b1_decode_kernel<RD>, q) == cudaSuccess, "attn_b1: launch");
}

template <int RD>
static void launch_b1v(const P& p, int cl, cudaStream_t st) {
  const int bytes = MAXCHUNK * 4 + RV * 64 * 2 + TILE + RECV8_BYTES + 2 * FSTAGE + RD * 2 * TILE;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(b1_verify_kernel<RD>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    cudaFuncSetAttribute(b1_verify_kernel<RD>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr = true;
  }
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(p.S);
  c.blockDim = dim3(NW * 32);
  c.dynamicSmemBytes = bytes;
  c.stream = st;
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = cl; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  c.attrs = a;
  c.numAttrs = 1;
  B1_CHECK(cudaLaunchKernelEx(&c, b1_verify_kernel<RD>, p) == cudaSuccess, "attn_b1: verify launch");
}

template <int RD>
static void launch_wgv(const P& p, int cl, cudaStream_t st) {
  const int bytes = RV * 64 * 2 + WRAW + RECV8_BYTES + 2 * WFST + RD * WRAW + ((p.chunk * 4 + 15) & ~15);
  static int optin = 0;
  if (!optin) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    optin -= 1024;
    cudaFuncSetAttribute(b1_wgv_kernel<RD>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    cudaFuncSetAttribute(b1_wgv_kernel<RD>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  B1_CHECK(bytes <= optin, "attn_b1: wgv shared memory %d > %d", bytes, optin);
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(p.S);
  c.blockDim = dim3(NW * 32);
  c.dynamicSmemBytes = bytes;
  c.stream = st;
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = cl; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  c.attrs = a;
  c.numAttrs = 1;
  B1_CHECK(cudaLaunchKernelEx(&c, b1_wgv_kernel<RD>, p) == cudaSuccess, "attn_b1: wgv launch");
}

template <int RD>
static void launch_ws_pp(const P& p, int cl, cudaStream_t st) {
  const int bytes = WS_SMEM + RD * WSRAW + 4 * WSVT;
  const bool qlo = (p.mode & (1 << 24)) != 0, ppw = !qlo && (p.mode & (1 << 13)) != 0;  // PPW: W16 producers
  auto kf = qlo ? b1_ws_kernel<RD, 4, true, false, 4, true>
                : ppw ? b1_ws_kernel<RD, 4, true, true, 2, true> : b1_ws_kernel<RD, 4, true, false, 2, true>;
  const int ai = qlo ? 1 : ppw ? 2 : 0;
  static bool attr[3] = {false, false, false};
  if (!attr[ai]) {
    cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    cudaFuncSetAttribute(kf, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr[ai] = true;
  }
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(p.S);
  c.blockDim = dim3(ppw ? 768 : 640);
  c.dynamicSmemBytes = bytes;
  c.stream = st;
  cudaLaunchAttribute a[2];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = cl; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  a[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a[1].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = a;
  c.numAttrs = g_b1_pdl ? 2 : 1;
  P q = p;
  q.pdl = g_b1_pdl ? 1 : 0;
  B1_CHECK(cudaLaunchKernelEx(&c, kf, q) == cudaSuccess, "attn_b1: ws pp launch");
}

template <int RD, int VD>
static void launch_ws(const P& p, int cl, cudaStream_t st) {
  const int bytes = WS_SMEM + RD * WSRAW + VD * WSVT;
  const bool pipe = (p.mode & (1 << 25)) != 0, w16 = (p.mode & (1 << 12)) != 0;
  const bool qlo = pipe && w16 && (p.mode & (1 << 24)) != 0;  // hi + lo queries (B1_WS_QLO)
  auto kf = qlo ? b1_ws_kernel<RD, VD, true, true, 4>
                : pipe ? (w16 ? b1_ws_kernel<RD, VD, true, true> : b1_ws_kernel<RD, VD, true, false>)
                       : (w16 ? b1_ws_kernel<RD, VD, false, true> : b1_ws_kernel<RD, VD, false, false>);
  static bool attr[5] = {false, false, false, false, false};
  const int ai = qlo ? 4 : 2 * pipe + w16;
  if (!attr[ai]) {
    cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    cudaFuncSetAttribute(kf, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr[ai] = true;
  }
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(p.S);
  c.blockDim = dim3(w16 ? 512 : WS_NT);
  c.dynamicSmemBytes = bytes;
  c.stream = st;
  cudaLaunchAttribute a[2];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = cl; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  a[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a[1].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = a;
  c.numAttrs = g_b1_pdl ? 2 : 1;
  P q = p;
  q.pdl = g_b1_pdl ? 1 : 0;
  B1_CHECK(cudaLaunchKernelEx(&c, kf, q) == cudaSuccess, "attn_b1: ws launch");
}

static int vb_bytes(int rd, int chunk) {
  const int ringb = 2 * rd * TILE > 2 * RV * VB_PSTR * 4 ? 2 * rd * TILE : 2 * RV * VB_PSTR * 4;
  return RV * 128 + TILE + ringb + ((chunk * 4 + 15) & ~15);
}
static int vb_optin() {
  static int optin = 0;
  if (!optin) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    optin -= 1024;
  }
  return optin;
}
template <int RD>
static void launch_vb(const P& p, cudaStream_t st) {
  const int bytes = vb_bytes(RD, p.chunk);
  static int optin = 0;
  if (!optin) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    optin -= 1024;
    cudaFuncSetAttribute(b1_vb_kernel<RD>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    cudaFuncSetAttribute(b1_vb_kernel<RD>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  B1_CHECK(bytes <= optin, "attn_b1: vb shared memory %d > %d (chunk %d, RD %d)", bytes, optin, (int)p.chunk, RD);
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(p.S);
  c.blockDim = dim3(NW * 32);
  c.dynamicSmemBytes = bytes;
  c.stream = st;
  cudaLaunchAttribute a[2];
  a[0].id = cudaLaunchAttributeClusterDimension;  // one cluster per request
  a[0].val.clusterDim.x = p.Stok; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  a[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a[1].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = a;
  c.numAttrs = g_b1_pdl ? 2 : 1;
  P q = p;
  q.pdl = g_b1_pdl ? 1 : 0;
  B1_CHECK(cudaLaunchKernelEx(&c, b1_vb_kernel<RD>, q) == cudaSuccess, "attn_b1: vb launch");
}

// clusters of `cl` CTAs that can be resident at once: kind 0 = the decode kernel (ring depth 2, `chunk` page ids),
// kind 1 = the ws verify kernel on full layers (6-deep raw / 8-deep V^T ring, pipelined, 16 warps)
int64_t attn_b1_max_clusters(int64_t kind, int64_t cl, int64_t chunk) {
  cudaLaunchConfig_t c = {};
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = static_cast<unsigned>(cl); a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  c.attrs = a;
  c.numAttrs = 1;
  c.gridDim = dim3(static_cast<unsigned>(cl) * GS);
  int n = 0;
  if (kind == 0) {
    auto kf = b1_decode_kernel<2>;
    c.blockDim = dim3(NW * 32);
    c.dynamicSmemBytes = ROWS * 64 * 2 + TILE + RECV_BYTES + NW * 2 * TILE + ((static_cast<int>(chunk) * 4 + 15) & ~15) + 16;
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - 1024);
    cudaFuncSetAttribute(kf, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (cudaOccupancyMaxActiveClusters(&n, kf, &c) != cudaSuccess) n = 0;
  } else {
    auto kf = b1_ws_kernel<6, 8, true, true>;
    c.blockDim = dim3(512);
    c.dynamicSmemBytes = WS_SMEM + 6 * WSRAW + 8 * WSVT;
    cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(c.dynamicSmemBytes));
    cudaFuncSetAttribute(kf, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (cudaOccupancyMaxActiveClusters(&n, kf, &c) != cudaSuccess) n = 0;
  }
  cudaGetLastError();  // (a failed query leaves no sticky error: n = 0 means "unknown", the schedule stays)
  return n;
}

struct B1Args {  // (declared identically in the .cu and the .cpp)
  int64_t qkv, cache_loc, k_cache, v_cache, page_table, seq_lens, inv_freq, grp_ml, grp_o, cnt, out, dbg;  // device pointers
  int64_t dbg_numel, k_cache_numel, page_table_numel, max_pages, cache_loc_numel, seq_lens_numel, cnt_numel;
  int64_t Hq, page_shift, window, S, chunk, cl, depth, mode, stream;
  int64_t contiguous, rope, warm, mode_verify;
  double qk_scale, sm_scale, k_scale, v_scale;
};

static void attn_b1_impl(const B1Args& a) {
  const int64_t Hq = a.Hq, page_shift = a.page_shift, window = a.window, S = a.S, chunk = a.chunk, cl = a.cl, depth = a.depth;
  const int64_t mode = a.mode;
  const bool rope = a.rope, warm = a.warm, mode_verify = a.mode_verify;
  const double qk_scale = a.qk_scale, sm_scale = a.sm_scale, k_scale = a.k_scale, v_scale = a.v_scale;
  B1_CHECK(Hq == 16 && a.contiguous, "attn_b1: layout");
  B1_CHECK(cl >= 1 && cl <= GS && (cl & (cl - 1)) == 0 && S % cl == 0 && chunk <= 32768, "attn_b1: split");
  P p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<void*>(a.qkv));
  p.cache_loc = reinterpret_cast<int64_t*>(a.cache_loc);
  p.k_cache = reinterpret_cast<uint8_t*>(reinterpret_cast<void*>(a.k_cache));
  p.v_cache = reinterpret_cast<uint8_t*>(reinterpret_cast<void*>(a.v_cache));
  p.page_table = reinterpret_cast<int32_t*>(a.page_table);
  p.seq_lens = reinterpret_cast<int32_t*>(a.seq_lens);
  p.inv_freq = reinterpret_cast<float*>(a.inv_freq);
  p.grp_ml = reinterpret_cast<float*>(a.grp_ml); p.grp_o = reinterpret_cast<float*>(a.grp_o);
  p.cnt = reinterpret_cast<int*>(a.cnt);
  p.out = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<void*>(a.out));
  p.dbg = a.dbg_numel >= S * 16 ? reinterpret_cast<unsigned long long*>(reinterpret_cast<void*>(a.dbg)) : nullptr;
  p.cache_bytes = warm ? a.k_cache_numel : 0;
  p.table_bytes = warm ? a.page_table_numel * 4 : 0;
  p.mode = static_cast<int>(mode);
  p.pdl = 0;
  p.page_shift = page_shift; p.max_pages = a.max_pages; p.window = window; p.S = S; p.chunk = chunk;
  p.rope = rope ? 1 : 0; p.Hq = Hq;
  p.qk_scale = (float)qk_scale; p.q_mult = (float)(sm_scale * k_scale * 1.4426950408889634);  // scores in log2 units
  p.inv_k_scale = (float)(1.0 / k_scale); p.inv_v_scale = (float)(1.0 / v_scale);
  cudaStream_t st = reinterpret_cast<cudaStream_t>(a.stream);
  p.B = static_cast<int>(a.seq_lens_numel);
  p.L = static_cast<int>(a.cache_loc_numel) / p.B;
  p.Stok = S;
  if (mode_verify && (p.L == LV || (mode & 128))) {  // all 128 rows of a request per CTA (ws: p.L <= 8 tokens), keys split
    //                                                  // across S CTAs per request
    B1_CHECK(S / cl <= GS && a.cnt_numel >= p.B * RV && (chunk <= MAXCHUNK || (mode & 448)), "attn_b1: verify split");
    p.S = p.B * S;
    if (mode & 256) {  // batch: private group rings, one cluster per request (S a power of two <= 16)
      B1_CHECK(S <= GS && (S & (S - 1)) == 0, "attn_b1: vb split");
      const int tpg = ((chunk + 15) / 16 + 1) / 2;  // tiles per group (a shallower ring where shared memory is short)
      const int optin = vb_optin();
      if (tpg > 8 && vb_bytes(17, chunk) <= optin) launch_vb<17>(p, st);
      else if (tpg > 4 && vb_bytes(9, chunk) <= optin) launch_vb<9>(p, st);
      else launch_vb<5>(p, st);
      return;
    }
    if (mode & 128) {  // warp-specialized FP8 kernel
      if (false) {
      }
      // full layers: raw ring of 6 stages, V^T ring of 8 (219 KB of shared memory);
      // sliding layers: raw ring of 8, V^T ring of 4
      // ping-pong consumers (B1_WS_PP): batched full layers only (on batch-1 and sliding layers a CTA has too few
      // stages to amortize the parity merge)
      else if ((mode & (1 << 19)) && p.window == 0 && (p.B > 1 || (mode & (1 << 26)))) launch_ws_pp<8>(p, cl, st);
      else if (p.window == 0) launch_ws<6, 8>(p, cl, st);
      else launch_ws<8, 4>(p, cl, st);
      return;
    }
    if (mode & 64) {  // warpgroup-MMA consumer (a shallower raw ring when the page-id array is large)
      if (chunk > 4096) launch_wgv<2>(p, cl, st);
      else launch_wgv<4>(p, cl, st);
      return;
    }
    launch_b1v<12>(p, cl, st);  // raw ring 12 stages (96 KB in flight), smem ~210 KB: 1 CTA / SM
    return;
  }
  // one decode step per token: grid L x S, the token's S CTAs in its own clusters
  B1_CHECK(p.L >= 1 && p.L <= LMAX && S / cl <= 2 * cl && a.cnt_numel >= p.B * p.L * ROWS, "attn_b1: per-token split");
  p.S = p.B * p.L * S;
  if (depth <= 2) launch_b1<2>(p, cl, st);
  else if (depth <= 4) launch_b1<4>(p, cl, st);
  else launch_b1<6>(p, cl, st);
}

const char* attn_b1_launch(const B1Args* a) {
  g_b1_err = nullptr;
  attn_b1_impl(*a);
  return g_b1_err;
}
"""

ATTN_B1_CPP = r"""
#include <ATen/cuda/CUDAContext.h>
void attn_b1_set_pdl(bool on);
int64_t attn_b1_max_clusters(int64_t kind, int64_t cl, int64_t chunk);
struct B1Args {  // (declared identically in the .cu and the .cpp)
  int64_t qkv, cache_loc, k_cache, v_cache, page_table, seq_lens, inv_freq, grp_ml, grp_o, cnt, out, dbg;  // device pointers
  int64_t dbg_numel, k_cache_numel, page_table_numel, max_pages, cache_loc_numel, seq_lens_numel, cnt_numel;
  int64_t Hq, page_shift, window, S, chunk, cl, depth, mode, stream;
  int64_t contiguous, rope, warm, mode_verify;
  double qk_scale, sm_scale, k_scale, v_scale;
};
const char* attn_b1_launch(const B1Args* a);

void attn_b1(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
             torch::Tensor seq_lens, torch::Tensor inv_freq, torch::Tensor grp_ml, torch::Tensor grp_o, torch::Tensor cnt,
             torch::Tensor out, torch::Tensor dbg, int64_t Hq, int64_t page_shift, int64_t window, bool rope, double qk_scale,
             double sm_scale, double k_scale, double v_scale, int64_t S, int64_t chunk, int64_t cl, int64_t depth, bool warm,
             int64_t mode, bool mode_verify) {
  B1Args a;
  a.qkv = reinterpret_cast<int64_t>(qkv.data_ptr());
  a.cache_loc = reinterpret_cast<int64_t>(cache_loc.data_ptr<int64_t>());
  a.k_cache = reinterpret_cast<int64_t>(k_cache.data_ptr());
  a.v_cache = reinterpret_cast<int64_t>(v_cache.data_ptr());
  a.page_table = reinterpret_cast<int64_t>(page_table.data_ptr<int32_t>());
  a.seq_lens = reinterpret_cast<int64_t>(seq_lens.data_ptr<int32_t>());
  a.inv_freq = reinterpret_cast<int64_t>(inv_freq.data_ptr<float>());
  a.grp_ml = reinterpret_cast<int64_t>(grp_ml.data_ptr<float>());
  a.grp_o = reinterpret_cast<int64_t>(grp_o.data_ptr<float>());
  a.cnt = reinterpret_cast<int64_t>(cnt.data_ptr<int>());
  a.out = reinterpret_cast<int64_t>(out.data_ptr());
  a.dbg = reinterpret_cast<int64_t>(dbg.data_ptr());
  a.dbg_numel = dbg.numel(); a.k_cache_numel = k_cache.numel(); a.page_table_numel = page_table.numel();
  a.max_pages = page_table.size(1); a.cache_loc_numel = cache_loc.numel(); a.seq_lens_numel = seq_lens.numel();
  a.cnt_numel = cnt.numel();
  a.Hq = Hq; a.page_shift = page_shift; a.window = window; a.S = S; a.chunk = chunk; a.cl = cl; a.depth = depth; a.mode = mode;
  a.stream = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());
  a.contiguous = qkv.is_contiguous() && k_cache.is_contiguous() && page_table.is_contiguous();
  a.rope = rope; a.warm = warm; a.mode_verify = mode_verify;
  a.qk_scale = qk_scale; a.sm_scale = sm_scale; a.k_scale = k_scale; a.v_scale = v_scale;
  const char* e = attn_b1_launch(&a);
  TORCH_CHECK(e == nullptr, e ? e : "");
}
"""

_B1_EXT = {}
_B1_CNT = {}
_B1_LL = {}
_B1_INV_FREQ = {}
B1_SPLITS_SLIDING, B1_CLUSTER_SLIDING = 16, 16  # (CTAs, cluster size); H200: 7 co-resident clusters of 16, 15 of 8
B1_SPLITS_FULL, B1_CLUSTER_FULL = 120, 8
B1V_SPLITS_SLIDING, B1V_CLUSTER_SLIDING = 8, 8  # q_len 8 sliding: 8 tokens x 8 CTAs = 8 co-resident clusters of 8
B1V_SPLITS_FULL, B1V_CLUSTER_FULL = 112, 8  # q_len 8 full layers (verify kernel): 14 clusters of 8, all co-resident on the H200 (GPCs 7 x 18 + 6 SMs; a 15th cluster would run late)
B1_VERIFY_FULL = True  # batch-1 q_len 8 full layers: the warp-specialized FP8 kernel
B1_BATCH_SLIDING = True  # batch 2..32 sliding layers: one decode step per (request, token)
B1_BATCH_SCHED = {(16, 1): (2, 8), (32, 1): (2, 4)}  # (batch, q_len) -> (CL, S)
B1_BATCH_FULL = False  # batch 16 / 32 full layers (q_len 1)
B1_BATCH_SCHED_FULL = {(16, 1): (2, 8), (32, 1): (2, 4)}
B1_BATCH_VERIFY = True  # batch q_len 8 via the warp-specialized FP8 kernel on the shapes listed below
B1_WGV = True  # verify path: the warpgroup-MMA consumer (b1_wgv_kernel) instead of the mma.sync one
B1_WS = True  # verify path: the warp-specialized FP8 kernel (b1_ws_kernel; overrides B1_WGV)
B1_WS_B1 = False  # batch-1 q_len 8 sliding through the verify path (else one decode step per token)
B1_BATCH_VSCHED = {(16, 8): (4, 4), (32, 8): (2, 4)}  # (batch, q_len) -> (CL, S per request), sliding
B1_BATCH_VSCHED_FULL = {(16, 8): (2, 8), (32, 8): (2, 4)}  # full layers
B1_VB = False  # batch q_len 8 via b1_vb_kernel (off: its mma.sync loop is latency-bound)
B1_VB_SCHED = {(16, 8): 8, (32, 8): 4}  # (batch, q_len) -> CTAs per request, sliding
B1_VB_SCHED_FULL = {}  # full layers
B1_DEPTH = 0       # ring slots per warp; 0: enough for the warp's whole share (at most 6)
# (ws kernel: gates staged in smem and the f16 O / l cluster push are compile-time on; the st.async merge off)
B1_WS_DECFULL = False  # batch decode (q_len 1) full layers through the ws kernel (p.L = 1: 16 of its 128 rows)
B1_WS_DECFULL_SCHED = {(16, 1): (2, 8), (32, 1): (2, 4)}  # (batch, q_len) -> (CL, S per request)
B1_WS16 = True     # ws kernel: 16 warps (4 loaders, 4 transposers, consumers on warps 8-15; balanced setmaxnreg)
B1_WS_PP = os.environ.get("OKC_B1_WS_PP", "0") == "1"  # ws kernel ping-pong: 4 consumer warpgroups (kv head x stage parity)
B1_WS_PIPE = True   # ws kernel: pipelined consumer (next stage's QK under this stage's softmax)
B1_WS_QLO = os.environ.get("OKC_B1_WS_QLO", "0") == "1"  # ws kernel queries as e4m3 hi + lo (verify rel-L2 ~7e-2 -> ~1e-2; +7% time)
B1_SA = True       # decode kernel: cluster merge by st.async + per-rank mbarrier (no cluster-wide barrier after the push)
B1_LL = False      # decode kernel, several clusters per step: flag-in-data cross-cluster merge (off: polling slows
                   # the stragglers more than the tickets do)
B1_PTPF = True     # decode kernel, batch-1 sliding: the request's page-table row -> L2 while its length loads
B1_L2PF = 0        # decode kernel: the warp's tiles past the ring -> L2 up front (1: prefetch.global.L2, 2: bulk prefetch)
B1_WARM = True     # TLB warm-up loads of the cache pools and page table
B1_PROBE = False   # probe build: phase stamps (B1_DBG) and the ablation mask (B1_MODE)
B1_MODE = 0
B1_DBG = None      # int64 [S * 16]
_B1_NODBG = {}


B1_WS_PP1 = False  # ping-pong ws consumers on batch-1 full layers too (mode bit 26)
B1_WS_RSKIP = False  # ping-pong consumer: skip the O rescale when no row's max moved (a build: B1_RSKIP)
B1_TAIL_D = {}  # (batch, q_len, sliding) -> cached keys the tail CTA gives up (its new-token work)
B1_WS_PPW = False  # ping-pong ws kernel with the W16 producers: 4 loaders + 4 transposers, 768 threads (mode bit 13)
B1_PTPF_B = False  # page-table row L2 prefetch for batched sliding decode too (mode bit 10, per request)
B1_WS_NEWLAST = False  # PIPE consumer: the tail CTA's new-token stage last (mode bit 15)
B1_DEC_T = False  # transposed tile math in the batch-1 decode kernel (a separate build: B1_DT)
B1_WS_H2A = False  # + f16-accumulator QK in the ping-pong consumer (a separate build: B1_H2A)
B1_WS_H2 = False  # ping-pong ws consumer: packed-half softmax on full stages (a separate build: B1_H2)


def attn_b1_ext():
    key = (bool(B1_PROBE), bool(B1_WS_H2 or B1_WS_H2A), bool(B1_WS_H2A), bool(B1_DEC_T), bool(B1_WS_RSKIP))
    if key not in _B1_EXT:
        import os
        arch = os.environ.get("OKC_ARCH")
        if not arch:
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}{'a' if major == 9 else ''}"
        src = ("#define OKC_B1_PROBE 1\n" if key[0] else "") + ("#define B1_H2 1\n" if key[1] else "") + ("#define B1_H2A 1\n" if key[2] else "") + ("#define B1_DT 1\n" if key[3] else "") + ("#define B1_RSKIP 1\n" if key[4] else "") + ATTN_B1_SRC
        _B1_EXT[key] = lean_load_inline("okc_attn_b1_v110" + ("_probe" if key[0] else "") + ("_h2" if key[1] else "") + ("a" if key[2] else "") + ("_dt" if key[3] else "") + ("_rs" if key[4] else ""), cpp_sources=ATTN_B1_CPP, cuda_sources=src,
                                   functions=["attn_b1", "attn_b1_set_pdl", "attn_b1_max_clusters"], extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                                   )
    return _B1_EXT[key]


def _b1_keys(data) -> int:
    cfg = data["config"]
    max_keys = data["page_table"].shape[1] * cfg["page_size"]
    return min(max_keys, cfg["window"]) if cfg["window"] else max_keys


_B1_MAXCL = {}


def _b1_max_clusters(kind: int, cl: int, chunk: int) -> int:
    """Co-resident clusters of `cl` CTAs for kind 0 (decode kernel) / 1 (ws verify kernel), asked once per device;
    a large number when unknown (the query failed or it is not an H200-class part with 16 clusters per launch)."""
    key = (torch.cuda.current_device(), kind, cl, chunk // 1024)
    if key not in _B1_MAXCL:
        n = int(attn_b1_ext().attn_b1_max_clusters(kind, cl, chunk))
        _B1_MAXCL[key] = n if n > 0 else 1 << 20
    return _B1_MAXCL[key]


def b1_schedule(keys: int, window: int, splits=None, cluster=None, q_len=1, batch=1):
    """(S, chunk, cluster). Decode kernel (one decode step per (request, token)): S CTAs per step (a multiple of the
    cluster size, at most 2 CL clusters) of `chunk` keys (a multiple of 16) covering `keys` positions, trailing CTAs
    possibly empty. Verify kernel (batch 1, q_len 8, full layers): CL >= 8, at most 16 clusters."""
    vb = (B1_VB_SCHED if window else B1_VB_SCHED_FULL).get((batch, q_len)) if B1_VB else None
    if batch > 1 and q_len == 8 and vb:  # private-group-ring kernel: S CTAs per request, no clusters
        S, cl = splits or vb, 1
    elif batch > 1 and q_len == 1 and not window and B1_WS_DECFULL and B1_WS and (batch, q_len) in B1_WS_DECFULL_SCHED:
        cl, S = B1_WS_DECFULL_SCHED[(batch, q_len)]  # ws kernel with one token per request
        cl, S = cluster or cl, splits or S
        S = max(cl, min(S, 16 * cl) // cl * cl)
    elif batch > 1 and q_len == 8:  # verify kernel: S CTAs per request, clusters of CL, at most 16 clusters per request
        cl, S = (B1_BATCH_VSCHED if window else B1_BATCH_VSCHED_FULL).get((batch, q_len), (1, 1))
        cl, S = cluster or cl, splits or S
        S = max(cl, min(S, 16 * cl) // cl * cl)
    elif batch > 1:  # (S, CL) per step, chosen so every cluster of the grid is co-resident on the H200
        cl, S = (B1_BATCH_SCHED if window else B1_BATCH_SCHED_FULL).get((batch, q_len), (1, 1))
        cl, S = cluster or cl, splits or S
        S = max(cl, min(S, 2 * cl * cl) // cl * cl)
    elif q_len == 8 and not window:
        cl = cluster or B1V_CLUSTER_FULL
        S = splits or B1V_SPLITS_FULL
        S = max(cl, min(S, 16 * cl, cl * _b1_max_clusters(1, cl, 0)) // cl * cl)
    elif q_len == 8:
        cl = cluster or B1V_CLUSTER_SLIDING
        S = splits or B1V_SPLITS_SLIDING
        S = max(cl, min(S, 2 * cl * cl) // cl * cl)
    else:
        cl = cluster or (B1_CLUSTER_SLIDING if window else B1_CLUSTER_FULL)
        S = splits or (B1_SPLITS_SLIDING if window else B1_SPLITS_FULL)
        S = max(cl, min(S, 2 * cl * cl) // cl * cl)  # at most 2 CL clusters (each merge lane folds 2)
        if not window:
            S = max(cl, min(S, cl * _b1_max_clusters(0, cl, -(-keys // S))) // cl * cl)
    keys_eff = keys + B1_TAIL_D.get((batch, q_len, bool(window)), 0)  # the tail CTA's deficit (new tokens)
    chunk = -(-(-(-keys_eff // S)) // 16) * 16
    return S, chunk, cl


def attn_b1_supported(data) -> bool:
    """Batch 1: q_len 1 (any layer) and q_len 8 on sliding layers (q_len 8 full layers: B1_VERIFY_FULL). Batch 16 / 32:
    q_len 1 on sliding layers (B1_BATCH_SLIDING), one decode step per request."""
    cfg = data["config"]
    ps = cfg["page_size"]
    L, B = cfg["q_len"], cfg["batch"]
    if not (cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64 and ps > 0 and ps & (ps - 1) == 0):
        return False
    if B == 1:
        ok = L == 1 or (L == 8 and (cfg["window"] or B1_VERIFY_FULL))
    else:
        if L == 8:
            ok = (B1_VB and (B, L) in (B1_VB_SCHED if cfg["window"] else B1_VB_SCHED_FULL)) or (
                B1_BATCH_VERIFY and (B, L) in (B1_BATCH_VSCHED if cfg["window"] else B1_BATCH_VSCHED_FULL))
        elif cfg["window"]:
            ok = B1_BATCH_SLIDING and (B, L) in B1_BATCH_SCHED
        else:
            ok = (B1_BATCH_FULL and (B, L) in B1_BATCH_SCHED_FULL) or (
                B1_WS_DECFULL and B1_WS and (B, L) in B1_WS_DECFULL_SCHED)
    return bool(ok) and b1_schedule(_b1_keys(data), cfg["window"], q_len=L, batch=B)[1] <= (2048 if B == 1 and L == 1 else 24576)


def _inv_freq(theta: float, dev):
    key = (theta, str(dev))
    if key not in _B1_INV_FREQ:
        if theta:
            _B1_INV_FREQ[key] = (1.0 / (theta ** (torch.arange(0, 64, 2, device=dev).float() / 64))).contiguous()
        else:
            _B1_INV_FREQ[key] = torch.zeros(32, dtype=torch.float32, device=dev)
    return _B1_INV_FREQ[key]


def attn_b1_forward(data, splits=None, depth=None, cluster=None, cnt=None):
    """Batch-1 decode / verify step: [1, q_len, Hq * 64] bf16 (normalized, gated attention output); writes the new
    cache rows."""
    cfg = data["config"]
    qkv = data["qkv"]
    dev = qkv.device
    L = cfg["q_len"]
    B = cfg["batch"]
    S, chunk, cl = b1_schedule(_b1_keys(data), cfg["window"], splits, cluster, q_len=L, batch=B)
    verify = L == 8 and (B > 1 or not cfg["window"] or (B1_WS and B1_WS_B1))  # all-rows verify kernel (else per token)
    verify = verify or (L == 1 and B > 1 and not cfg["window"] and B1_WS_DECFULL and B1_WS and (B, L) in B1_WS_DECFULL_SCHED)
    vbr = B > 1 and L == 8 and B1_VB and (B, L) in (B1_VB_SCHED if cfg["window"] else B1_VB_SCHED_FULL)
    d = depth or B1_DEPTH or min(6 if chunk <= 2048 else 4, -(-chunk // 128))  # tiles per warp (ring vs the ids array)
    if not depth and not B1_DEPTH and L == 1 and (B == 1) != bool(cfg["window"]):
        d = 2  # batch-1 full / batch sliding decode: a shallower ring issues sooner
    if cnt is None:  # a call site may pass its own zeroed int32 tensor (two instances must never run concurrently on one)
        cnt = _B1_CNT.get(str(dev))
        if cnt is None:
            cnt = _B1_CNT[str(dev)] = torch.zeros(32 * 8 * 16, dtype=torch.int32, device=dev)  # self-resetting tickets
    ncl = S // cl
    rows, parts = (128, B * ncl) if verify else (16, B * L * ncl)  # (the ws kernel indexes 128 rows per request)
    grp_ml = torch.empty((parts, rows, 2), dtype=torch.float32, device=dev)
    grp_o = torch.empty((parts, rows, 64), dtype=torch.float32, device=dev)
    llbit = 0
    if B1_LL and not verify and ncl > 1:  # persistent zeroed words [gen per problem (256)][parts x 16 rows x 68]
        ll = _B1_LL.get(str(dev))
        if ll is None and not torch.cuda.is_current_stream_capturing():
            ll = _B1_LL[str(dev)] = torch.zeros(256 + 128 * 16 * 68, dtype=torch.int64, device=dev)
        if ll is not None and B * L <= 256 and ll.numel() >= 256 + parts * 16 * 68:
            grp_o, llbit = ll.view(torch.float32), 1 << 27
    out = torch.empty((B, L, cfg["num_q_heads"] * 64), dtype=torch.bfloat16, device=dev)
    theta = float(cfg["rope_theta"])
    dbg = B1_DBG if (B1_PROBE and B1_DBG is not None) else _B1_NODBG.setdefault(str(dev), torch.empty(0, dtype=torch.int64, device=dev))
    attn_b1_ext().attn_b1(qkv.contiguous(), data["cache_loc"], data["k_cache"], data["v_cache"], data["page_table"],
                          data["seq_lens"], _inv_freq(theta, dev), grp_ml, grp_o, cnt, out, dbg, cfg["num_q_heads"],
                          cfg["page_size"].bit_length() - 1, cfg["window"], theta != 0.0, float(cfg["qk_scale_factor"]),
                          float(cfg["sm_scale"]), float(cfg["kv_scale"]), float(cfg["kv_scale"]), S, chunk, cl, d, B1_WARM,
                          (B1_MODE if B1_PROBE else 0) | (256 if (verify and vbr) else 128 if (verify and B1_WS) else 64 if (verify and B1_WGV) else 0)
                          | ((B1_L2PF & 3) << 28) | llbit | ((1 << 18) if B1_SA else 0) | ((1 << 25) if B1_WS_PIPE else 0) | ((1 << 12) if B1_WS16 else 0) | ((1 << 10) if (cfg["window"] and ((B1_PTPF and B == 1) or (B1_PTPF_B and B > 1))) else 0) | ((1 << 24) if B1_WS_QLO else 0) | ((1 << 19) if B1_WS_PP else 0) | ((1 << 26) if B1_WS_PP1 else 0) | ((1 << 13) if B1_WS_PPW else 0) | ((1 << 15) if B1_WS_NEWLAST else 0),
                          verify)
    return out
B1_WS_PPW = True  # ping-pong ws kernel with 4 loaders + 4 transposers
B1_WS_RSKIP = True  # + skip the O rescale when no row moved its max
B1_TAIL_D = {(16, 8, True): 256}  # verify-16 sliding: the tail CTA (which also prepares and attends the new tokens) gives up 256 cached keys
B1_WS_H2 = True  # packed-half softmax in the ws ping-pong / PIPE consumers (B1_H2 build)
B1_WS_H2A = True  # + f16-accumulator QK in the ping-pong consumer (B1_H2A build)
B1_WS_PP = os.environ.get("OKC_B1_WS_PP", "1") == "1"  # ping-pong ws consumers on batched full layers (OKC_B1_WS_PP=0 turns them off)
# Batch decode on full-attention layers (q_len 1), warp-specialized: the split-KV decode kernel above
# (attn_decode_kernel<1>: same query prep, new-token handling, 16-key tile math, partials and merge kernel) with its
# gathers moved to NLW dedicated loader warps. Page-size-1 gathers need rows in flight from many issuing warps (H200:
# 32-64 warps/SM -> ~4 TB/s), while attn_decode_kernel's 16 warps/SM both gather and compute. Here 8 compute warps (4
# pairs, one warp per kv head) consume a RS-tile ring that up to 24 loader warps (1024 threads per block) fill with
# whole 128-B K / V rows (tile i: loader i % NLW, pair i % 4, slot i % RS; per-slot full / empty mbarriers, cp.async
# completion signalled with cp.async.mbarrier.arrive). Loaders fetch their tiles' page ids from global one tile ahead
# (no MAX_CHUNK id staging: chunks of tens of thousands of keys). On sm_90a the loaders drop to 32 registers and the
# compute warps take the rest (balanced setmaxnreg). wsd_source assembles the kernel from pieces of the decode
# kernel's source.
import torch

WSD_KERNEL = r"""
// Length-balanced split (mode bit 256) of the batch's B <= 32 requests over N CTAs, computed by one whole warp from
// the lengths (lane b = request b): request b gets n_b >= 1 CTAs [P_b, P_b + n_b): 1 + floor(len_b (N - B) / T), then
// one more for the requests with the most keys per CTA (ties: lower index) until all N are used. Every warp of every
// CTA and the merge compute the same split.
__device__ __forceinline__ void wsd_split(const Params& p, int N, int lane, int& len, int& n, int& pre) {
  len = lane < p.B ? p.seq_lens[lane] : 0;
  int T = len;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) T += __shfl_xor_sync(0xffffffffu, T, o);
  n = lane < p.B ? 1 + static_cast<int>(static_cast<long long>(len) * (N - p.B) / max(T, 1)) : 0;
  int used = n;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) used += __shfl_xor_sync(0xffffffffu, used, o);
  const float w = lane < p.B ? static_cast<float>(len) / static_cast<float>(n) : -1.f;
  int rank = 0;
#pragma unroll 8
  for (int j = 0; j < 32; ++j) {
    const float wj = __shfl_sync(0xffffffffu, w, j);
    rank += (wj > w || (wj == w && j < lane)) ? 1 : 0;
  }
  if (lane < p.B && rank < N - used) ++n;
  int inc = n;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += v;
  }
  pre = inc - n;
}

__device__ __forceinline__ long long wsd_gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <int NLW, int RS, int CPS>
struct WsdCfg {
  static constexpr int NT = (NW + NLW) * 32;
  static constexpr int REG_LAUNCH = (65536 / (NT * CPS)) & ~7;  // CPS CTAs per SM
  static constexpr int REG_LOAD = 32;
  static constexpr int REG_COMP = (REG_LAUNCH + (REG_LAUNCH - REG_LOAD) * NLW / NW) & ~7;
  static constexpr int BYTES = RS * SLOT + SLOT + HKV * 16 * 64 * 2 + 2 * RS * 8 + 64;
};

// CW (compute-warp gathers): RS 8, 4 loaders. Loader lw owns slot lw (tiles lw + 8 k); compute pair wq owns slot wq + 4
// (tiles wq + 4 + 8 k, its own odd tiles): both of its warps gather their kv head's 64-B halves of the tile as soon as
// they have released the slot's previous tile. 24 issuing warps per SM rather than 16 (page-size-1 gathers scale with
// the issuing warps); every slot still has one producer.
template <int NLW, int RS, int CPS, int CW = 0>
__global__ void __launch_bounds__(WsdCfg<NLW, RS, CPS>::NT, CPS) attn_wsd_kernel(const Params p) {
  // each slot must keep one producer (loader i % NLW) and one consumer pair (i % 4) across its phases: parity waits
  // cannot tell phase k from k - 2, so a second loader running two phases ahead would pass a stale "empty" wait
  static_assert(RS % NLW == 0 && RS % 4 == 0, "ring depth must be a multiple of the loader count and of 4");
  static_assert(!CW || (RS == 8 && NLW == 4), "compute-warp gathers: slots 0..3 loaders, 4..7 pairs");
  constexpr int L = 1;
  using Cfg = WsdCfg<NLW, RS, CPS>;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* ring = smem;
  unsigned char* newslot = ring + RS * SLOT;
  uint16_t* qs = reinterpret_cast<uint16_t*>(newslot + SLOT);       // [HKV][16 rows][64] f16
  uint64_t* full = reinterpret_cast<uint64_t*>(qs + HKV * 16 * 64);
  uint64_t* empty = full + RS;
  float* red = reinterpret_cast<float*>(ring);                       // merge scratch after the loop
  constexpr int QH = 16;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  int s = blockIdx.x, b = blockIdx.y, nsplit = p.splits, chunk = p.chunk, cbase = blockIdx.y * p.splits;
  if (p.mode & 256) {  // length-balanced: this CTA's request and its share of it
    int len, n, pre;
    wsd_split(p, gridDim.x, lane, len, n, pre);
    const unsigned own = __ballot_sync(0xffffffffu, lane < p.B && static_cast<int>(blockIdx.x) >= pre && static_cast<int>(blockIdx.x) < pre + n);
    if (own == 0u) return;  // a spare CTA (the whole block: every warp computes the same split)
    b = __ffs(own) - 1;
    nsplit = __shfl_sync(0xffffffffu, n, b);
    cbase = __shfl_sync(0xffffffffu, pre, b);
    s = blockIdx.x - cbase;
    chunk = ((__shfl_sync(0xffffffffu, len, b) + nsplit - 1) / nsplit + 15) & ~15;
  }
  const int group = p.Hq / HKV;
  const int seq = p.seq_lens[b];
  const int first = 0;
  const int k0 = first + s * chunk;
  const int k1 = min(seq, k0 + chunk);
  const bool tail = (s == nsplit - 1);
#ifdef WSD_PROBE
  long long* const prb = reinterpret_cast<long long*>(p.part_o + static_cast<size_t>((p.mode & 256) ? p.splits : p.B * p.splits) * HKV * 8 * 64)
                         + (blockIdx.y * gridDim.x + blockIdx.x) * 16;
  const long long t_entry = wsd_gtime();
#endif
  const int ncached_here = max(0, k1 - k0);
  const int ntiles_c = (ncached_here + 15) / 16;
  constexpr int ROW_BYTES = HKV * 64;
  const int qkv_stride = (2 * p.Hq + 2 * HKV) * 64;
  const bool dyn2 = (p.mode & 4096) != 0;
  const int kend = dyn2 ? seq : k1;  // end of the cached keys this CTA may take
  int* const slot_pos = reinterpret_cast<int*>(empty + RS);  // dyn2: each slot's key position (-1: its loader ended)
  __shared__ int s_ldone;  // dyn2: loader warps past their last claim
  if (tid == 0) {
    s_ldone = 0;
    for (int i = 0; i < RS; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(full + i))), "r"((CW && i >= 4) ? 64 : dyn2 ? 33 : 32) : "memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(empty + i))), "r"(2) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  gdc_launch();
  if (warp >= NW) {
    // ================= loaders: tiles lw, lw + NLW, ...; whole 128-B rows, 4 rows per instruction ==================
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(Cfg::REG_LOAD));
#endif
    const int lw = warp - NW;
    auto page_id = [&](int i) {  // lane l < 16: flat cache row of key l of tile i (0 if past the chunk)
      const int idx = i * 16 + (lane & 15);
      if (i >= ntiles_c || idx >= ncached_here) return 0;
      const int pos = k0 + idx;
      return p.page_table[static_cast<size_t>(b) * p.max_pages + pos / p.page_size] * p.page_size + pos % p.page_size;
    };
    if constexpr (CW) {  // this loader's slot lw: tiles lw, lw + RS, ... (the pairs fill slots 4..7)
#ifdef WSD_PROBE
      long long lwait = 0;
      const long long lstart = clock64();
#endif
      int idn = page_id(lw);
      for (int i = lw; i < ntiles_c; i += RS) {
        const int slot = lw;
        const int cur = idn;
        idn = page_id(i + RS);
        if (i >= RS) {
#ifdef WSD_PROBE
          const long long w0 = clock64();
#endif
          const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(empty + slot));
          const int par = ((i / RS) - 1) & 1;
          uint32_t done;
          do {
            asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                         : "=r"(done) : "r"(a), "r"(par) : "memory");
          } while (!done);
#ifdef WSD_PROBE
          lwait += clock64() - w0;
#endif
        }
        unsigned char* dst0 = ring + slot * SLOT;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int key = 4 * r + (lane >> 3), piece = lane & 7, head_kv = piece >> 2, sub = piece & 3;
          const int row = __shfl_sync(0xffffffffu, cur, key);
          const bool valid = i * 16 + key < ncached_here;
          const size_t off = static_cast<size_t>(row) * ROW_BYTES + piece * 16;
          cp_async16(dst0 + head_kv * KHEAD + key * KROW + sub * 16, p.k_cache + off, valid ? 16 : 0);
          cp_async16(dst0 + HKV * KHEAD + head_kv * VHEAD + key * VROW + sub * 16, p.v_cache + off, valid ? 16 : 0);
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(full + slot))) : "memory");
      }
      asm volatile("cp.async.wait_all;" ::: "memory");
#ifdef WSD_PROBE
      if (lw == 0 && lane == 0) { prb[12] = lwait; prb[13] = clock64() - lstart; }
#endif
      return;
    }
    constexpr int D = RS / NLW;  // tiles in flight per loader warp (slots lw, lw + NLW, ...)
    int ids[D];  // page ids of this loader's next D tiles: requested D tiles ahead, reloaded in place (no moves)
#pragma unroll
    for (int d = 0; d < D; ++d) ids[d] = page_id(lw + d * NLW);
#ifdef WSD_PROBE
    long long lwait = 0;
    const long long lstart = clock64();
#endif
    if (dyn2) {  // (host: RS == NLW, so D = 1 and slot = loader)
      int* const dcnt = reinterpret_cast<int*>(const_cast<float*>(p.inv_freq)) + 64;  // [b] claims, [32 + b] CTAs done
      const int ntr = (seq + 15) / 16;  // the request's tiles
      const int t0 = 2 * NLW * nsplit;  // claimed pairs start past the static ones
      int pend = 0;  // (lane 0) the pending claim: issued after tile 0's copies (PDL: the static pair needs no wait)
      int base = 2 * (s * NLW + lw);  // this pair's first tile (pair 0: static)
      int nbase = -1;                 // the next pair's, resolved at the pair's second tile
      auto pos_of = [&](int j) -> int {  // key 0 of local tile j of the current pair (j & 1), or -1
        const int t = base + (j & 1);
        return t < ntr ? 16 * t : -1;
      };
      auto ids_of = [&](int tp) {  // lane l < 16: flat cache row of key l of the tile at key tp (0 past the request)
        const int pos = tp + (lane & 15);
        if (tp < 0 || pos >= seq) return 0;
        return p.page_table[static_cast<size_t>(b) * p.max_pages + pos / p.page_size] * p.page_size + pos % p.page_size;
      };
      int tcur = pos_of(0);
      int idc = ids_of(tcur);
      for (int j = 0;; ++j) {
        const int i = lw + j * NLW, slot = lw;
        if (i >= RS) {
#ifdef WSD_PROBE
          const long long w0 = clock64();
#endif
          const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(empty + slot));
          const int par = ((i / RS) - 1) & 1;
          uint32_t done;
          do {
            asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                         : "=r"(done) : "r"(a), "r"(par) : "memory");
          } while (!done);
#ifdef WSD_PROBE
          lwait += clock64() - w0;
#endif
        }
        const uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(full + slot));
        if (tcur >= 0) {
          unsigned char* dst0 = ring + slot * SLOT;
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const int key = 4 * r + (lane >> 3), piece = lane & 7, head_kv = piece >> 2, sub = piece & 3;
            const int row = __shfl_sync(0xffffffffu, idc, key);
            const bool valid = tcur + key < seq;
            const size_t off = static_cast<size_t>(row) * ROW_BYTES + piece * 16;
            cp_async16(dst0 + head_kv * KHEAD + key * KROW + sub * 16, p.k_cache + off, valid ? 16 : 0);
            cp_async16(dst0 + HKV * KHEAD + head_kv * VHEAD + key * VROW + sub * 16, p.v_cache + off, valid ? 16 : 0);
          }
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(fb) : "memory");
        if (lane == 0) {
          slot_pos[slot] = tcur;  // published by the release arrive (the full barrier's 33rd)
          asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(fb) : "memory");
        }
        if (tcur < 0) break;  // the end marker is out
        if (j == 0 && lane == 0) {  // the first claim: after the static tile's copies (PDL overlap kept)
          gdc_wait();  // the previous launch's last CTAs reset the counters
          pend = atomicAdd(dcnt + b, 2);
        }
        // the next tile's position (after this tile's copies): same pair, or the next pair - its claim was issued
        // two tiles ago (the first one: one tile ago)
        int tnext;
        if ((j & 1) == 0) {
          tnext = pos_of(1);
        } else {
          if (lane == 0) nbase = t0 + pend;
          nbase = __shfl_sync(0xffffffffu, nbase, 0);
          if (lane == 0 && nbase < ntr) pend = atomicAdd(dcnt + b, 2);  // (no claim past the end: pend stays resolved)
          tnext = (nbase >= ntr) ? -1 : 16 * nbase;
        }
        const int idn = ids_of(tnext);
        if (j & 1) base = nbase;
        tcur = tnext;
        idc = idn;
      }
      asm volatile("cp.async.wait_all;" ::: "memory");
      if (lane == 0) {
        if (tcur < 0 && base == 2 * (s * NLW + lw)) gdc_wait();  // (ended at tile 0: no claim, but the counters below)
        if (pend < 0) __trap();  // (waits for this loader's last claim: claims are never negative)
        __threadfence();
        if (atomicAdd(&s_ldone, 1) == NLW - 1 && atomicAdd(dcnt + 32 + b, 1) == nsplit - 1) {
          __threadfence();  // the request's last CTA: every claim of every CTA is in; reset for the next launch
          dcnt[b] = 0;
          dcnt[32 + b] = 0;
        }
      }
#ifdef WSD_PROBE
      if (lw == 0 && lane == 0) { prb[12] = lwait; prb[13] = clock64() - lstart; }
#endif
      return;
    }
    for (int i0 = lw; i0 < ntiles_c; i0 += D * NLW) {
#pragma unroll
      for (int d = 0; d < D; ++d) {
        const int i = i0 + d * NLW;
        if (i >= ntiles_c) break;
        const int slot = i % RS;
        const int cur = ids[d];
        ids[d] = page_id(i + D * NLW);  // in flight across the slot wait (D = 1: one tile ahead)
        if (i >= RS) {
#ifdef WSD_PROBE
          const long long w0 = clock64();
#endif
          const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(empty + slot));
          const int par = ((i / RS) - 1) & 1;
          uint32_t done;
          do {
            asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                         : "=r"(done) : "r"(a), "r"(par) : "memory");
          } while (!done);
#ifdef WSD_PROBE
          lwait += clock64() - w0;
#endif
        }
        unsigned char* dst0 = ring + slot * SLOT;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int key = 4 * r + (lane >> 3), piece = lane & 7, head_kv = piece >> 2, sub = piece & 3;
          const int row = __shfl_sync(0xffffffffu, cur, key);
          const bool valid = i * 16 + key < ncached_here;
          const size_t off = static_cast<size_t>(row) * ROW_BYTES + piece * 16;
          cp_async16(dst0 + head_kv * KHEAD + key * KROW + sub * 16, p.k_cache + off, valid ? 16 : 0);
          cp_async16(dst0 + HKV * KHEAD + head_kv * VHEAD + key * VROW + sub * 16, p.v_cache + off, valid ? 16 : 0);
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(full + slot))) : "memory");
      }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
#ifdef WSD_PROBE
    if (lw == 0 && lane == 0) { prb[12] = lwait; prb[13] = clock64() - lstart; }
#endif
    return;
  }
  // ================= compute warps (8 warps, as in attn_decode_kernel) ==============================================
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(Cfg::REG_COMP));
#endif
  const int hh = warp & 1;
  const int wq = warp >> 1;
  // CW: this pair's own slot wq + 4 (tiles wq + 4 + 8 k); each warp gathers its kv head's 64-B halves of the rows
  auto page_id_c = [&](int i) {  // lane l < 16: flat cache row of key l of tile i (0 if past the chunk)
    const int idx = i * 16 + (lane & 15);
    if (i >= ntiles_c || idx >= ncached_here) return 0;
    const int pos = k0 + idx;
    return p.page_table[static_cast<size_t>(b) * p.max_pages + pos / p.page_size] * p.page_size + pos % p.page_size;
  };
  auto self_issue = [&](int t, int idv) {
    unsigned char* dst0 = ring + (t % RS) * SLOT;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int key = 8 * r + (lane >> 2), sub = lane & 3;
      const int row = __shfl_sync(0xffffffffu, idv, key);
      const bool valid = t * 16 + key < ncached_here;
      const size_t off = static_cast<size_t>(row) * ROW_BYTES + hh * 64 + sub * 16;
      cp_async16(dst0 + hh * KHEAD + key * KROW + sub * 16, p.k_cache + off, valid ? 16 : 0);
      cp_async16(dst0 + HKV * KHEAD + hh * VHEAD + key * VROW + sub * 16, p.v_cache + off, valid ? 16 : 0);
    }
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(full + (t % RS)))) : "memory");
  };
  int idn_self = 0;
  if constexpr (CW) {  // (inputs of the step: before the PDL wait)
    if (wq + 4 < ntiles_c) {
      const int id0 = page_id_c(wq + 4);
      idn_self = page_id_c(wq + 4 + RS);
      self_issue(wq + 4, id0);
    }
  }
  gdc_wait();
@@QPREP@@
@@NEWTOK@@
  asm volatile("bar.sync 1, %0;" ::"n"(NW * 32) : "memory");
@@QFRAG@@
@@CONSUME@@
  // --- tiles wq, wq + 4, ...: wait the slot's fill, consume, release ---------------------------------------------
#ifdef WSD_PROBE
  long long cwait = 0;
  int ncons = 0;
  const long long cstart = clock64();
  if (warp == 0 && lane == 0) prb[1] = wsd_gtime();
#endif
#if defined(WSD_T) && defined(WSD_TP)
  {  // software-pipelined: tile i + 4's smem reads and QK before tile i's softmax / PV; slots released after the reads
    auto wait_full = [&](int i) {
#ifdef WSD_PROBE
      const long long w0 = clock64();
#endif
      const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(full + i % RS));
      const int par = (i / RS) & 1;
      uint32_t done;
      do {
        asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                     : "=r"(done) : "r"(a), "r"(par) : "memory");
      } while (!done);
#ifdef WSD_PROBE
      cwait += clock64() - w0;
#endif
    };
    auto release = [&](int i) {
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(empty + i % RS))) : "memory");
    };
    float st[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t vb0[8], vb1[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) vb0[j] = vb1[j] = 0u;
    if (wq < ntiles_c && p.mode != 1) {
      wait_full(wq);
      qk_tile(ring + (wq % RS) * SLOT, st, vb0, vb1);
      release(wq);
    }
    for (int i = wq; i < ntiles_c && p.mode != 1; i += 4) {
      float st2[4] = {0.f, 0.f, 0.f, 0.f};
      uint32_t vc0[8], vc1[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) vc0[j] = vc1[j] = 0u;
      if (i + 4 < ntiles_c) {
        wait_full(i + 4);
        qk_tile(ring + ((i + 4) % RS) * SLOT, st2, vc0, vc1);
        release(i + 4);
      }
      soft_pv(min(16, ncached_here - 16 * i), st, vb0, vb1);
#pragma unroll
      for (int c = 0; c < 4; ++c) st[c] = st2[c];
#pragma unroll
      for (int j = 0; j < 8; ++j) { vb0[j] = vc0[j]; vb1[j] = vc1[j]; }
    }
  }
#else
  if (dyn2) {  // tiles in ring order; a slot word of -1 ends its loader's stream (skip that loader from then on)
    uint32_t live = 0;
    for (int ld = wq; ld < NLW; ld += 4) live |= 1u << ld;
    for (int i = wq; live; i += 4) {
      const int ld = i % NLW, slot = i % RS;
      if (!(live & (1u << ld))) continue;
      {
        const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(full + slot));
        const int par = (i / RS) & 1;
        uint32_t done;
        do {
          asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                       : "=r"(done) : "r"(a), "r"(par) : "memory");
        } while (!done);
      }
      const int tpos = *reinterpret_cast<volatile int*>(slot_pos + slot);
      if (tpos < 0) {
        live &= ~(1u << ld);
        continue;
      }
#ifdef WSD_PROBE
      ++ncons;
#endif
      consume(tpos, ring + slot * SLOT);
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(empty + slot))) : "memory");
    }
  } else
  for (int i = wq; i < ntiles_c; i += 4) {
    const int slot = i % RS;
#ifdef WSD_PROBE
    const long long w0 = clock64();
#endif
    {
      const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(full + slot));
      const int par = (i / RS) & 1;
      uint32_t done;
      do {
        asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                     : "=r"(done) : "r"(a), "r"(par) : "memory");
      } while (!done);
    }
#ifdef WSD_PROBE
    cwait += clock64() - w0;
#endif
#ifndef WSD_NOMATH  // (WSD_NOMATH probe build: the loaders and the ring handshake only, no tile math)
    consume(k0 + 16 * i, ring + slot * SLOT);
#endif
    __syncwarp();
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(empty + slot))) : "memory");
    if constexpr (CW) {
      if (slot >= 4 && i + RS < ntiles_c) {  // own slot: refill with tile i + RS once both warps of the pair released it
        const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(empty + slot));
        const int par = (i / RS) & 1;
        uint32_t done;
        do {
          asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                       : "=r"(done) : "r"(a), "r"(par) : "memory");
        } while (!done);
        const int cur = idn_self;
        idn_self = page_id_c(i + 2 * RS);
        self_issue(i + RS, cur);
      }
    }
  }
  if constexpr (CW) asm volatile("cp.async.wait_all;" ::: "memory");
#endif
  if (tail && wq == 0) consume(seq, newslot);
#ifdef WSD_PROBE
  if (lane == 0) prb[2 + warp] = wsd_gtime();
  if (warp == 0 && lane == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    prb[10] = cwait; prb[11] = clock64() - cstart;
    prb[15] = (dyn2 ? 4 * ncons : ntiles_c) | (static_cast<long long>(smid) << 32) | (static_cast<long long>(tail) << 48);
  }
#endif
  if (p.mode == 1) return;
#ifndef WSD_T
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    l_lo[m] += __shfl_xor_sync(0xffffffffu, l_lo[m], 1); l_lo[m] += __shfl_xor_sync(0xffffffffu, l_lo[m], 2);
    l_hi[m] += __shfl_xor_sync(0xffffffffu, l_hi[m], 1); l_hi[m] += __shfl_xor_sync(0xffffffffu, l_hi[m], 2);
  }
#else
#pragma unroll
  for (int o2 = 4; o2 < 32; o2 <<= 1) {
    lh[0] += __shfl_xor_sync(0xffffffffu, lh[0], o2);
    lh[1] += __shfl_xor_sync(0xffffffffu, lh[1], o2);
  }
#endif
  constexpr int ROWS = HKV * 8 * L;
  const size_t pbase = (static_cast<size_t>(cbase) + s) * ROWS;
  asm volatile("bar.sync 1, %0;" ::"n"(NW * 32) : "memory");  // every compute warp is done with the ring
  {
@@PARTIALS@@
  }
}

@@MERGE@@
struct WsdArgs {  // (declared identically in the .cu and the .cpp)
  int64_t qkv, cache_loc, k_cache, v_cache, page_table, seq_lens, inv_freq, part_o, part_ml, out;  // device pointers
  int64_t max_pages, B, Hq, page_size, splits, chunk, nlw, rs, cps, stream;
  int64_t pdl, bal, mpdl, fused, dyn2;
  double qk_scale, sm_scale, k_scale, v_scale;
};
static char g_wsd_msg[160];

// (no torch in the .cu: the torch-typed attn_wsd() binding is in WSD_CPP; nullptr or an error message)
const char* attn_wsd_launch(const WsdArgs* a) {
  const int64_t B = a->B, Hq = a->Hq, page_size = a->page_size, splits = a->splits, chunk = a->chunk;
  const int64_t nlw = a->nlw, rs = a->rs, cps = a->cps;
  const bool pdl = a->pdl, bal = a->bal, mpdl = a->mpdl, fused = a->fused == 1, fixref = a->fused == 2, dyn2 = a->dyn2;
  const double qk_scale = a->qk_scale, sm_scale = a->sm_scale, k_scale = a->k_scale, v_scale = a->v_scale;
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(a->qkv);
  p.cache_loc = reinterpret_cast<int64_t*>(a->cache_loc);
  p.k_cache = reinterpret_cast<uint8_t*>(a->k_cache);
  p.v_cache = reinterpret_cast<uint8_t*>(a->v_cache);
  p.page_table = reinterpret_cast<int32_t*>(a->page_table);
  p.seq_lens = reinterpret_cast<int32_t*>(a->seq_lens);
  p.inv_freq = reinterpret_cast<float*>(a->inv_freq);
  p.part_o = reinterpret_cast<float*>(a->part_o);
  p.part_ml = reinterpret_cast<float*>(a->part_ml);
  p.out = reinterpret_cast<__nv_bfloat16*>(a->out);
  p.B = B; p.L = 1; p.Hq = Hq; p.page_size = page_size; p.max_pages = a->max_pages;
  p.window = 0; p.splits = splits; p.chunk = chunk; p.rope = 0; p.mode = (bal ? 256 : 0) | (fused ? 1024 : 0) | (fixref ? 8192 : 0) | (dyn2 ? 4096 : 0);  // balanced: splits = N CTAs
  p.qk_scale = static_cast<float>(qk_scale);
  p.q_mult = static_cast<float>(sm_scale * k_scale);
  p.inv_k_scale = static_cast<float>(1.0 / k_scale); p.inv_v_scale = static_cast<float>(1.0 / v_scale);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(a->stream);
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  lc.attrs = at; lc.numAttrs = 1; lc.stream = stream;
  lc.gridDim = bal ? dim3(splits) : dim3(splits, B);
  cudaError_t err = cudaErrorInvalidValue;
#define WSD_CASE(NLW_, RS_, CPS_)                                                                              \
  if (nlw == NLW_ && rs == RS_ && cps == CPS_) {                                                               \
    using C = WsdCfg<NLW_, RS_, CPS_>;                                                                         \
    static bool ready = false;                                                                                 \
    if (!ready) { cudaFuncSetAttribute(attn_wsd_kernel<NLW_, RS_, CPS_>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::BYTES); ready = true; } \
    lc.blockDim = dim3(C::NT); lc.dynamicSmemBytes = C::BYTES;                                                 \
    err = cudaLaunchKernelEx(&lc, attn_wsd_kernel<NLW_, RS_, CPS_>, p);                                        \
  }
  // at most 24 loaders (1024 threads per block); CPS 2: two CTAs per SM (8 compute pairs per SM)
  WSD_CASE(24, 24, 1) WSD_CASE(8, 16, 2) WSD_CASE(8, 8, 2) WSD_CASE(16, 16, 1) WSD_CASE(8, 16, 1) WSD_CASE(10, 20, 2) WSD_CASE(4, 8, 3)
#undef WSD_CASE
  if (nlw == (256 | 4) && rs == 8 && cps == 2) {  // compute-warp gathers (nlw bit 8)
    using C = WsdCfg<4, 8, 2>;
    static bool ready = false;
    if (!ready) { cudaFuncSetAttribute(attn_wsd_kernel<4, 8, 2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::BYTES); ready = true; }
    lc.blockDim = dim3(C::NT); lc.dynamicSmemBytes = C::BYTES;
    err = cudaLaunchKernelEx(&lc, attn_wsd_kernel<4, 8, 2, 1>, p);
  }
  if (err != cudaSuccess) (void)cudaGetLastError();  // clear it: a failed launch must not poison later CUDA calls
  if (err != cudaSuccess) {
    snprintf(g_wsd_msg, sizeof(g_wsd_msg), "attn wsd launch: %s", cudaGetErrorString(err));
    return g_wsd_msg;
  }
  if (fused || fixref) return nullptr;  // the last CTA of each request merged it
  lc.gridDim = dim3(B * HKV * 8); lc.blockDim = dim3(MW * 32); lc.dynamicSmemBytes = 0;
  at[0].val.programmaticStreamSerializationAllowed = (pdl || mpdl) ? 1 : 0;
  err = bal ? cudaLaunchKernelEx(&lc, wsd_merge_kernel, p) : cudaLaunchKernelEx(&lc, attn_merge_kernel, p);
  if (err != cudaSuccess) {
    snprintf(g_wsd_msg, sizeof(g_wsd_msg), "attn merge launch: %s", cudaGetErrorString(err));
    return g_wsd_msg;
  }
  return nullptr;
}
"""
WSD_CPP = r"""
#include <ATen/cuda/CUDAContext.h>
struct WsdArgs {  // (declared identically in the .cu and the .cpp)
  int64_t qkv, cache_loc, k_cache, v_cache, page_table, seq_lens, inv_freq, part_o, part_ml, out;  // device pointers
  int64_t max_pages, B, Hq, page_size, splits, chunk, nlw, rs, cps, stream;
  int64_t pdl, bal, mpdl, fused, dyn2;
  double qk_scale, sm_scale, k_scale, v_scale;
};
const char* attn_wsd_launch(const WsdArgs* a);

void attn_wsd(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
              torch::Tensor seq_lens, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor out,
              int64_t B, int64_t Hq, int64_t page_size, double qk_scale, double sm_scale, double k_scale, double v_scale,
              int64_t splits, int64_t chunk, int64_t nlw, int64_t rs, int64_t cps, bool pdl, bool bal, bool mpdl, int64_t fused, bool dyn2) {
  WsdArgs a;
  a.qkv = reinterpret_cast<int64_t>(qkv.data_ptr());
  a.cache_loc = reinterpret_cast<int64_t>(cache_loc.data_ptr<int64_t>());
  a.k_cache = reinterpret_cast<int64_t>(k_cache.data_ptr());
  a.v_cache = reinterpret_cast<int64_t>(v_cache.data_ptr());
  a.page_table = reinterpret_cast<int64_t>(page_table.data_ptr<int32_t>());
  a.seq_lens = reinterpret_cast<int64_t>(seq_lens.data_ptr<int32_t>());
  a.inv_freq = reinterpret_cast<int64_t>(inv_freq.data_ptr<float>());
  a.part_o = reinterpret_cast<int64_t>(part_o.data_ptr<float>());
  a.part_ml = reinterpret_cast<int64_t>(part_ml.data_ptr<float>());
  a.out = reinterpret_cast<int64_t>(out.data_ptr());
  a.max_pages = page_table.size(1);
  a.B = B; a.Hq = Hq; a.page_size = page_size; a.splits = splits; a.chunk = chunk; a.nlw = nlw; a.rs = rs; a.cps = cps;
  a.stream = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());
  a.pdl = pdl; a.bal = bal; a.mpdl = mpdl; a.fused = fused; a.dyn2 = dyn2;
  a.qk_scale = qk_scale; a.sm_scale = sm_scale; a.k_scale = k_scale; a.v_scale = v_scale;
  const char* e = attn_wsd_launch(&a);
  TORCH_CHECK(e == nullptr, e ? e : "");
}
"""


def _between(src, start, end, include_start=True):
    i = src.index(start)
    j = src.index(end, i + len(start))
    return src[i if include_start else i + len(start):j]


def wsd_source(native_src):
    """The split-KV decode source (macros substituted) + the warp-specialized kernel assembled from its pieces."""
    k = native_src[native_src.index("template <int L>\n__global__ void __launch_bounds__(NW * 32) attn_decode_kernel"):]
    qprep = _between(k, "  // --- queries: rms * qk_scale", "  // --- the block's new tokens")
    newtok = _between(k, "  // --- the block's new tokens", "  __syncthreads();\n\n  // --- this warp's m-tile")
    newtok = newtok.replace("  unsigned char* newslot = ring + RS * SLOT;\n", "")
    qfrag = _between(k, "  // --- this warp's m-tile", "  // --- ring fill: tile i -> slot i % RS")
    consume = _between(k, "  // --- one 16-key tile of this warp's head", "  // --- main loop.")
    _T_CONSUME = _T_CONSUME_SRC
    _rv = "      if (__any_sync(0xffffffffu, a_lo != 1.f || a_hi != 1.f)) {  // the running max moved for some row: rescale O\n"
    assert consume.count(_rv) == 1
    if not WSD_RVOTE:  # (WSD_RVOTE: keep the decode kernel's warp-vote skip)
        consume = consume.replace(_rv, "      {  // rescale O unconditionally (x 1.0 is exact; a branch on a warp vote splits the heads' chains)\n")
    for _o, _n in (("  auto consume = [&](int i, const unsigned char* slot) {",
                    "  auto consume = [&](int tpos, const unsigned char* slot) {  // tpos: the tile's key 0 (seq: the new tokens)"),
                   ("const int tile_key0 = (i < ntiles_c) ? k0 + 16 * i : seq;", "const int tile_key0 = tpos;"),
                   ("const int nvalid = (i < ntiles_c) ? min(16, ncached_here - 16 * i) : L;",
                    "const int nvalid = (tpos < seq) ? min(16, kend - tpos) : L;")):
        assert consume.count(_o) == 1, (_o, consume.count(_o))
        consume = consume.replace(_o, _n)
    parts = _between(k, "  if (L == 1) {\n    __syncthreads();  // ring reads are done", "  } else {\n#pragma unroll\n    for (int m = 0; m < MT; ++m) {\n      const int r_lo")
    parts = parts.replace("  if (L == 1) {\n    __syncthreads();  // ring reads are done; reuse it as scratch: [NW warps][8 rows] m, l and [NW][8][64] O",
                          "  {  // reuse the ring as scratch: [NW warps][8 rows] m, l and [NW][8][64] O")
    parts = parts.replace("    __syncthreads();\n    if (warp < 2)", "    asm volatile(\"bar.sync 1, %0;\" ::\"n\"(NW * 32) : \"memory\");\n    if (warp < 2)")
    _sw = "    if (t == 0) { rm[warp * 8 + g] = m_lo[0]; rl[warp * 8 + g] = l_lo[0]; }\n#pragma unroll\n    for (int j = 0; j < 8; ++j) { ro[(warp * 8 + g) * 64 + 16 * t + j] = o[0][j][0]; ro[(warp * 8 + g) * 64 + 16 * t + 8 + j] = o[0][j][1]; }\n"
    assert parts.count(_sw) == 1
    parts = parts.replace(_sw, "#ifndef WSD_T\n" + _sw + "#else\n" + (
        "    if (g == 0) { rm[warp * 8 + 2 * t] = mh[0]; rm[warp * 8 + 2 * t + 1] = mh[1]; rl[warp * 8 + 2 * t] = lh[0]; rl[warp * 8 + 2 * t + 1] = lh[1]; }\n"
        "#pragma unroll\n    for (int mt = 0; mt < 4; ++mt)\n#pragma unroll\n      for (int c = 0; c < 4; ++c) ro[(warp * 8 + 2 * t + (c & 1)) * 64 + 8 * g + 2 * mt + (c >> 1)] = ot[mt][c];\n") + "#endif\n")
    _pst = """      float* po = p.part_o + (pbase + row) * 64 + 16 * t;
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < NWH; ++k) acc += ro[((warp + 2 * k) * 8 + g) * 64 + 16 * t + d] * cc[k];
        po[d] = acc;
      }
      if (t == 0) { p.part_ml[(pbase + row) * 2] = mm; p.part_ml[(pbase + row) * 2 + 1] = ll; }
"""
    assert parts.count(_pst) == 1, parts.count(_pst)
    parts = parts.replace(_pst, _FIXREF_STORE)
    parts = parts + _FUSED_SRC + _FIXREF_SRC
    parts = parts + "#ifdef WSD_PROBE\n  if (tid == 0) { prb[0] = t_entry; prb[14] = wsd_gtime(); }\n#endif\n"
    kern = WSD_KERNEL.replace("@@QPREP@@", qprep).replace("@@NEWTOK@@", newtok).replace("@@QFRAG@@", qfrag)
    kern = kern.replace("@@CONSUME@@", "#ifndef WSD_T\n" + consume + "#else\n" + _T_CONSUME + "#endif\n").replace("@@PARTIALS@@", parts + "  }")
    merge = _between(native_src, "__global__ void __launch_bounds__(MW * 32) attn_merge_kernel(const Params p) {", "\n}  // namespace okcattn")
    merge = merge.replace("attn_merge_kernel(const Params p)", "wsd_merge_kernel(const Params p)")
    merge = merge.replace("  const size_t base = static_cast<size_t>(b) * p.splits;\n", "  size_t base = static_cast<size_t>(b) * p.splits;\n  int nsplit = p.splits;\n")
    merge = merge.replace("  gdc_wait();\n", "  gdc_wait();\n  if (p.mode & 256) {  // balanced: request b's n_b partials at P_b\n    int len, n, pre;\n    wsd_split(p, p.splits, lane, len, n, pre);\n    nsplit = __shfl_sync(0xffffffffu, n, b);\n    base = __shfl_sync(0xffffffffu, pre, b);\n  }\n", 1)
    merge = merge.replace("  for (int s = warp; s < p.splits; s += MW) {", "  for (int s = warp; s < nsplit; s += MW) {")
    assert merge.count("nsplit") == 3, merge.count("nsplit")
    kern = kern.replace("@@MERGE@@", merge)
    head = native_src[:native_src.index("void attn_decode(")]  # ends after `using namespace okcattn;`
    # (torch-free .cu: the decode source's torch headers go; its device code and Params use none of them)
    for inc in ("#include <torch/extension.h>\n", "#include <ATen/cuda/CUDAContext.h>\n"):
        head = head.replace(inc, "")
    return head + "#include <cstdio>\n" + kern


# Fixed-reference merge (mode 8192): a row whose log2 offset m is within +-100 adds O 2^m and l 2^m (fp32, against the
# absolute reference 2^0: no max exchange, adds commute) into request b's accumulator rows behind the partial slots
# (zero at rest); the last CTA of the request to arrive normalizes them and writes the output, then zeroes them. A row
# outside +-100 (e.g. a key planted on the query, which puts the logit near 2^346) keeps its exact partial and flags the
# request; the last CTA then also folds those partials exactly, with the accumulator as one more partial at m = 0.
_FIXREF_STORE = r'''      float* po = p.part_o + (pbase + row) * 64 + 16 * t;
      float accv[16];
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < NWH; ++k) acc += ro[((warp + 2 * k) * 8 + g) * 64 + 16 * t + d] * cc[k];
        accv[d] = acc;
      }
      if ((p.mode & 8192) && fabsf(mm) <= 100.f) {
        const float f = exp2f(mm);
        float* ao = p.part_o + (static_cast<size_t>(p.splits) * ROWS + static_cast<size_t>(b) * ROWS + row) * 64 + 16 * t;
#pragma unroll
        for (int d = 0; d < 16; d += 4)
          asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(ao + d), "f"(accv[d] * f), "f"(accv[d + 1] * f),
                       "f"(accv[d + 2] * f), "f"(accv[d + 3] * f) : "memory");
        if (t == 0) {
          atomicAdd(p.part_ml + static_cast<size_t>(p.splits) * ROWS * 2 + static_cast<size_t>(b) * ROWS + row, ll * f);
          p.part_ml[(pbase + row) * 2] = -INFINITY;  // (nothing to fold for this row)
        }
      } else {
#pragma unroll
        for (int d = 0; d < 16; ++d) po[d] = accv[d];
        if (t == 0) {
          p.part_ml[(pbase + row) * 2] = mm;
          p.part_ml[(pbase + row) * 2 + 1] = ll;
          if ((p.mode & 8192) && mm != -INFINITY) reinterpret_cast<int*>(const_cast<float*>(p.inv_freq))[32 + b] = 1;
        }
      }
'''
_FIXREF_SRC = r'''
  if (p.mode & 8192) {
    __shared__ int s_lastf;
    int* const tick = reinterpret_cast<int*>(const_cast<float*>(p.inv_freq));  // [0, 32) tickets, [32, 64) exact-partial flags
    __threadfence();  // this thread's reds / partial stores before the ticket (only the merging warps issued any)
    asm volatile("bar.sync 1, %0;" ::"n"(NW * 32) : "memory");
    if (tid == 0) s_lastf = atomicAdd(tick + b, 1) == nsplit - 1;
    asm volatile("bar.sync 1, %0;" ::"n"(NW * 32) : "memory");
    if (s_lastf) {
      __threadfence();
      const int row = tid >> 4, q4 = tid & 15;  // 16 rows (kv head x 8 query heads) x 16 threads: dims 4 q4 .. 4 q4 + 3
      float* ao = p.part_o + (static_cast<size_t>(p.splits) * ROWS + static_cast<size_t>(b) * ROWS + row) * 64 + 4 * q4;
      float* al = p.part_ml + static_cast<size_t>(p.splits) * ROWS * 2 + static_cast<size_t>(b) * ROWS + row;
      const float4 a4 = __ldcg(reinterpret_cast<const float4*>(ao));
      float Ls = __ldcg(al), ac[4] = {a4.x, a4.y, a4.z, a4.w};
      float M = Ls > 0.f ? 0.f : -INFINITY;  // the accumulator is a partial at m = 0 (none if nothing was added)
      if (__ldcg(tick + 32 + b)) {
        for (int s2 = 0; s2 < nsplit; ++s2) {
          const size_t pr = (static_cast<size_t>(cbase) + s2) * ROWS + row;
          const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml + pr * 2));
          if (ml.x == -INFINITY) continue;
          const float4 o4 = __ldcg(reinterpret_cast<const float4*>(p.part_o + pr * 64 + 4 * q4));
          const float mn = fmaxf(M, ml.x);
          const float a = (M == -INFINITY) ? 0.f : exp2f(M - mn), c = exp2f(ml.x - mn);
          M = mn;
          Ls = Ls * a + ml.y * c;
          ac[0] = ac[0] * a + o4.x * c; ac[1] = ac[1] * a + o4.y * c; ac[2] = ac[2] * a + o4.z * c; ac[3] = ac[3] * a + o4.w * c;
        }
      }
      const float inv = (Ls > 0.f ? 1.f / Ls : 0.f) / p.inv_v_scale;  // / l, * v_scale
      float d[4] = {ac[0] * inv, ac[1] * inv, ac[2] * inv, ac[3] * inv};
      float ss = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
      const unsigned gm = 0xffffu << (lane & 16);  // the row's 16 threads share a half-warp
#pragma unroll
      for (int o2 = 8; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(gm, ss, o2);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const int head = (row >> 3) * group + (row & 7);
      const __nv_bfloat16* gp = p.qkv + static_cast<size_t>(b) * qkv_stride + (p.Hq + 2 * HKV) * 64 + head * 64 + 4 * q4;
      const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp));
      const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 2));
      __nv_bfloat16* op = p.out + static_cast<size_t>(b) * (p.Hq * 64) + head * 64 + 4 * q4;
      *reinterpret_cast<__nv_bfloat162*>(op) = __floats2bfloat162_rn(d[0] * rn / (1.f + expf(-g01.x)), d[1] * rn / (1.f + expf(-g01.y)));
      *reinterpret_cast<__nv_bfloat162*>(op + 2) = __floats2bfloat162_rn(d[2] * rn / (1.f + expf(-g23.x)), d[3] * rn / (1.f + expf(-g23.y)));
      *reinterpret_cast<float4*>(ao) = make_float4(0.f, 0.f, 0.f, 0.f);  // the accumulator is zero again for the next call
      if (q4 == 0) *al = 0.f;
      asm volatile("bar.sync 1, %0;" ::"n"(NW * 32) : "memory");  // every thread has read the flag
      if (tid == 0) { tick[b] = 0; tick[32 + b] = 0; }
    }
  }
'''
_FUSED_SRC = r'''
  if (p.mode & 1024) {
    // fused merge: request b's last CTA to arrive folds its nsplit partials (CTA order) and writes the output
    __shared__ int s_lastw;
    int* const tick = reinterpret_cast<int*>(const_cast<float*>(p.inv_freq));  // per-request tickets (self-resetting)
    asm volatile("bar.sync 1, %0;" ::"n"(NW * 32) : "memory");  // this CTA's partial is written
    if (tid == 0) {
      __threadfence();
      s_lastw = atomicAdd(tick + b, 1) == nsplit - 1;
    }
    asm volatile("bar.sync 1, %0;" ::"n"(NW * 32) : "memory");
    if (s_lastw) {
      __threadfence();
      const int row = tid >> 4, q4 = tid & 15;  // 16 rows (kv head x 8 query heads) x 16 threads: dims 4 q4 .. 4 q4 + 3
      float M = -INFINITY, Ls = 0.f, ac[4] = {0.f, 0.f, 0.f, 0.f};
      for (int s2 = 0; s2 < nsplit; ++s2) {
        const size_t pr = (static_cast<size_t>(cbase) + s2) * ROWS + row;
        const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml + pr * 2));
        const float4 o4 = __ldcg(reinterpret_cast<const float4*>(p.part_o + pr * 64 + 4 * q4));
        if (ml.x == -INFINITY) continue;
        const float mn = fmaxf(M, ml.x);
        const float a = (M == -INFINITY) ? 0.f : exp2f(M - mn), c = exp2f(ml.x - mn);
        M = mn;
        Ls = Ls * a + ml.y * c;
        ac[0] = ac[0] * a + o4.x * c; ac[1] = ac[1] * a + o4.y * c; ac[2] = ac[2] * a + o4.z * c; ac[3] = ac[3] * a + o4.w * c;
      }
      const float inv = (Ls > 0.f ? 1.f / Ls : 0.f) / p.inv_v_scale;  // / l, * v_scale
      float d[4] = {ac[0] * inv, ac[1] * inv, ac[2] * inv, ac[3] * inv};
      float ss = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
      const unsigned gm = 0xffffu << (lane & 16);  // the row's 16 threads share a half-warp
#pragma unroll
      for (int o2 = 8; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(gm, ss, o2);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const int head = (row >> 3) * group + (row & 7);
      const __nv_bfloat16* gp = p.qkv + static_cast<size_t>(b) * qkv_stride + (p.Hq + 2 * HKV) * 64 + head * 64 + 4 * q4;
      const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp));
      const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 2));
      __nv_bfloat16* op = p.out + static_cast<size_t>(b) * (p.Hq * 64) + head * 64 + 4 * q4;
      *reinterpret_cast<__nv_bfloat162*>(op) = __floats2bfloat162_rn(d[0] * rn / (1.f + expf(-g01.x)), d[1] * rn / (1.f + expf(-g01.y)));
      *reinterpret_cast<__nv_bfloat162*>(op + 2) = __floats2bfloat162_rn(d[2] * rn / (1.f + expf(-g23.x)), d[3] * rn / (1.f + expf(-g23.y)));
      if (tid == 0) tick[b] = 0;
    }
  }
'''
_T_CONSUME_SRC = r'''
  // --- transposed tile math (WSD_T): thread (g, t) owns query heads 2 t, 2 t + 1 of this warp's kv head: S^T keys g,
  //     g + 8; O^T dims 8 g + 2 mt (+ 1), mt = 0..3
  float ot[4][4];
  float mh[2] = {-INFINITY, -INFINITY}, lh[2] = {0.f, 0.f};
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) ot[mt][0] = ot[mt][1] = ot[mt][2] = ot[mt][3] = 0.f;
  // qk_tile: every shared-memory read of a tile (K, V fragments) and S^T; soft_pv: softmax + PV (registers only)
  auto qk_tile = [&](const unsigned char* slot, float (&st)[4], uint32_t (&vb0)[8], uint32_t (&vb1)[8]) {
    const unsigned char* ks = slot + hh * KHEAD;
    const unsigned char* vs = slot + HKV * KHEAD + hh * VHEAD;
    uint32_t kf[2][8];  // keys g (u 0), g + 8 (u 1): 16 bytes at dims 16 t .. 16 t + 15 (the QK dim order)
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const uint4 kraw = *reinterpret_cast<const uint4*>(ks + (8 * u + g) * KROW + 16 * t);
      const uint16_t* kp = reinterpret_cast<const uint16_t*>(&kraw);
#pragma unroll
      for (int q = 0; q < 8; ++q) kf[u][q] = e4m3x2_to_f16x2(kp[q]);
    }
    uint32_t vf[4][4];  // keys 2 t, 2 t + 1, 2 t + 8, 2 t + 9 at dims 8 g .. 8 g + 7
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int key = (r < 2) ? 2 * t + r : 2 * t + 8 + (r - 2);
      const uint2 vraw = *reinterpret_cast<const uint2*>(vs + key * VROW + 8 * g);
      const uint16_t* vp = reinterpret_cast<const uint16_t*>(&vraw);
#pragma unroll
      for (int q = 0; q < 4; ++q) vf[r][q] = e4m3x2_to_f16x2(vp[q]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {  // dim 8 g + j: keys (2 t, 2 t + 1) / (2 t + 8, 2 t + 9)
      const uint32_t sel = (j & 1) ? 0x7632u : 0x5410u;
      vb0[j] = __byte_perm(vf[0][j >> 1], vf[1][j >> 1], sel);
      vb1[j] = __byte_perm(vf[2][j >> 1], vf[3][j >> 1], sel);
    }
    st[0] = st[1] = st[2] = st[3] = 0.f;  // (key g, head 2 t), (g, 2 t + 1), (g + 8, 2 t), (g + 8, 2 t + 1)
#pragma unroll
    for (int s4 = 0; s4 < 4; ++s4) {
      const uint32_t a[4] = {kf[0][2 * s4], kf[1][2 * s4], kf[0][2 * s4 + 1], kf[1][2 * s4 + 1]};
      mma_f16(st, a, qa[0][2 * s4], qa[0][2 * s4 + 1]);
    }
  };
  auto soft_pv = [&](int nvalid, const float (&st)[4], const uint32_t (&vb0)[8], const uint32_t (&vb1)[8]) {
    const bool v0 = g < nvalid, v1 = g + 8 < nvalid;
    const float s0 = v0 ? st[0] * LOG2E : -INFINITY, s1 = v0 ? st[1] * LOG2E : -INFINITY;
    const float s2 = v1 ? st[2] * LOG2E : -INFINITY, s3 = v1 ? st[3] * LOG2E : -INFINITY;
    float tm0 = fmaxf(s0, s2), tm1 = fmaxf(s1, s3);
#pragma unroll
    for (int o2 = 4; o2 < 32; o2 <<= 1) {
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, o2));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, o2));
    }
    const float mn0 = fmaxf(mh[0], tm0), mn1 = fmaxf(mh[1], tm1);
    const float a0 = (mn0 == -INFINITY) ? 1.f : exp2f(mh[0] - mn0), a1 = (mn1 == -INFINITY) ? 1.f : exp2f(mh[1] - mn1);
    mh[0] = mn0; mh[1] = mn1;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) { ot[mt][0] *= a0; ot[mt][1] *= a1; ot[mt][2] *= a0; ot[mt][3] *= a1; }
    const float e0 = (s0 == -INFINITY) ? 0.f : exp2f(s0 - mn0), e1 = (s1 == -INFINITY) ? 0.f : exp2f(s1 - mn1);
    const float e2 = (s2 == -INFINITY) ? 0.f : exp2f(s2 - mn0), e3 = (s3 == -INFINITY) ? 0.f : exp2f(s3 - mn1);
    lh[0] = lh[0] * a0 + (e0 + e2);
    lh[1] = lh[1] * a1 + (e1 + e3);
    uint32_t pb0, pb1;  // P^T B fragments: (head g; keys 2 t, 2 t + 1) / (head g; keys 2 t + 8, 2 t + 9)
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(pb0) : "r"(pack_f16x2(e0, e1)));
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(pb1) : "r"(pack_f16x2(e2, e3)));
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const uint32_t a[4] = {vb0[2 * mt], vb0[2 * mt + 1], vb1[2 * mt], vb1[2 * mt + 1]};
      mma_f16(ot[mt], a, pb0, pb1);
    }
  };
  auto consume = [&](int tpos, const unsigned char* slot) {  // one tile, unpipelined; tpos: its key 0 (seq: new tokens)
    if (p.mode == 1) return;
    float st[4];
    uint32_t vb0[8], vb1[8];
    qk_tile(slot, st, vb0, vb1);
    soft_pv((tpos < seq) ? min(16, kend - tpos) : L, st, vb0, vb1);
  };
'''
_WSD_EXT = None
_WSD_BUF = {}


WSD_TRANS = False  # transposed tile math build (WSD_T)
WSD_RVOTE = False  # keep the decode kernel's warp-vote O-rescale skip (a separate build)
WSD_FIXREF = False  # fixed-reference atomic merge + last-arriver finalize (no merge kernel; bal only)
WSD_CW = False  # compute-warp gathers: 4 loaders + the compute pairs gathering their own slots (instance 4, 8, 2)
WSD_PIPE = False  # + software-pipelined consumer loop (WSD_TP; with WSD_TRANS)
WSD_NOMATH = False  # probe build: no tile math (the gather rate of the loader / ring structure)
WSD_PROBE = False  # probe build: per-CTA stamps after the partials (wsd_forward(..., probe=[]))


_WSD_EXTS = {}


def wsd_ext():
    global _WSD_EXT
    fk = (bool(WSD_PROBE), bool(WSD_TRANS), bool(WSD_PIPE and WSD_TRANS), bool(WSD_RVOTE), bool(WSD_NOMATH))
    _WSD_EXT = _WSD_EXTS.get(fk)
    if _WSD_EXT is None:
        import zlib
        src = _TEAMV3_ATTN_DECODE_SRC.replace("__OKC_ATTN_NG__", "3").replace("__OKC_ATTN_HALF__", "0").replace(  # noqa: F821
            "__OKC_ATTN_MAXCHUNK__", "4096")
        src = ("#define WSD_NOMATH 1\n" if WSD_NOMATH else "") + ("#define WSD_PROBE 1\n" if WSD_PROBE else "") + ("#define WSD_T 1\n" if WSD_TRANS else "") + ("#define WSD_TP 1\n" if (WSD_PIPE and WSD_TRANS) else "") + wsd_source(src)
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}{'a' if major == 9 else ''}"
        tag = format(zlib.crc32(src.encode()) & 0xFFFFFFFF, "08x")
        _WSD_EXT = lean_load_inline(f"okc_wsd_{tag}", cpp_sources=WSD_CPP, cuda_sources=src, functions=["attn_wsd"],
                                    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"])
        _WSD_EXTS[fk] = _WSD_EXT
    return _WSD_EXT


def wsd_forward(data, nlw=8, rs=16, cps=2, nsm=None, pdl=True, out=None, splits=None, bal=None, probe=None, mpdl=False, fused=False, dyn2=False,
                cw=False, fixref=None):
    """Decode (q_len 1) on a full-attention layer through attn_wsd_kernel and its merge: CPS CTAs per SM."""
    fixref = WSD_FIXREF if fixref is None else fixref
    if WSD_CW and cps == 2 and not dyn2 and not (WSD_TRANS and WSD_PIPE):  # module switch: compute-warp gathers
        nlw, rs, cw = 4, 8, True
    assert not cw or (nlw == 4 and rs == 8 and cps == 2 and not dyn2 and not (WSD_TRANS and WSD_PIPE)), "cw: (4, 8, 2), static path"
    cfg = data["config"]
    B, Hq = cfg["batch"], cfg["num_q_heads"]
    assert cfg["q_len"] == 1 and cfg["window"] == 0 and cfg["num_kv_heads"] == 2
    qkv = data["qkv"]
    dev = qkv.device
    nsm = nsm or torch.cuda.get_device_properties(dev).multi_processor_count
    bal = (B <= 32) if bal is None else bal
    if bal:  # length-balanced: N = CPS x SMs CTAs over the batch (split on the device from the lengths)
        splits, chunk = splits or cps * nsm, 16
        assert B <= 32 and splits >= B
    else:
        max_keys = data["page_table"].shape[1] * cfg["page_size"]
        splits = splits or max(1, cps * nsm // B)
        chunk = -(-max_keys // splits)
        chunk = -(-chunk // 16) * 16
        splits = max(1, -(-max_keys // chunk))
    if out is None:
        out = torch.empty((B, 1, Hq * 64), dtype=torch.bfloat16, device=dev)
    key = (str(dev), B, splits, bal)
    if key not in _WSD_BUF:  # buffers cached per shape (a per-call allocation would add a fill kernel)
        np_ = splits if bal else B * splits  # partials: one per CTA
        # [0, 32) merge tickets, [32, 64) fixed-reference exact-partial flags, [64, 128) dyn2 counters; partial slots plus
        # 32 accumulator rows (fixed-reference merge, zero at rest: zeros here, kept zero by the kernel)
        _WSD_BUF[key] = (torch.zeros(128, dtype=torch.float32, device=dev),
                         torch.zeros((np_ + 32, 16, 64), dtype=torch.float32, device=dev),
                         torch.zeros((np_ + 32, 16, 2), dtype=torch.float32, device=dev))
    inv, part_o, part_ml = _WSD_BUF[key]
    if probe is not None:  # the probe build writes 16 int64 per CTA after the partials (one flat buffer)
        np_ = part_o.shape[0]
        flat = torch.zeros(np_ * 16 * 64 + np_ * 32, dtype=torch.float32, device=dev)
        part_o = flat[: np_ * 16 * 64].view(np_, 16, 64)
        probe.append(flat[np_ * 16 * 64:].view(torch.int64).view(np_, 16))
    wsd_ext().attn_wsd(qkv.contiguous(), data["cache_loc"], data["k_cache"], data["v_cache"], data["page_table"],
                       data["seq_lens"], inv, part_o, part_ml, out, B, Hq, cfg["page_size"], float(cfg["qk_scale_factor"]),
                       float(cfg["sm_scale"]), float(cfg["kv_scale"]), float(cfg["kv_scale"]), splits, chunk,
                       nlw | (256 if cw else 0), rs, cps,  # (cw: the compute-warp-gathers instance, 4 loaders, ring 8)
                       pdl, bal, mpdl, 2 if (fixref and bal) else int(bool(fused and bal)),
                       bool(dyn2 and bal and rs == nlw and not (WSD_PIPE and WSD_TRANS)))
    return out

_CONTROL_KERNEL_FN = kernel_fn
_B1_OFF = os.environ.get("OKC_B1_OFF", "0") == "1"
_B1_S_SLIDING = int(os.environ.get("OKC_B1_S_SLIDING", "0")) or None
_B1_S_FULL = int(os.environ.get("OKC_B1_S_FULL", "0")) or None
_B1_D = int(os.environ.get("OKC_B1_DEPTH", "0")) or None
_B1_CL = int(os.environ.get("OKC_B1_CL", "0")) or None
B1_MODE = int(os.environ.get("OKC_B1_MODE", "0"))
B1_PROBE = B1_MODE != 0 or os.environ.get("OKC_B1_PROBE", "0") == "1"
B1_WARM = os.environ.get("OKC_B1_WARM", "1") == "1"


_B1V_S = int(os.environ.get("OKC_B1V_S", "0")) or None
# batch decode on full layers: the warp-specialized kernel (NLW loader warps, RS-tile ring)
_WSD_ON = os.environ.get("OKC_WSD", "1") == "1"
_WSD_NLW = int(os.environ.get("OKC_WSD_NLW", "8"))
_WSD_RS = int(os.environ.get("OKC_WSD_RS", "8"))
_WSD_CPS = int(os.environ.get("OKC_WSD_CPS", "2"))
_B1V_CL = int(os.environ.get("OKC_B1V_CL", "0")) or None


_B1_PDL = []


def kernel_fn(data):
    if not _B1_PDL:  # PDL launch attribute for decoder chains (OKC_B1_PDL=1); standalone steps launch without it
        attn_b1_ext().attn_b1_set_pdl(os.environ.get("OKC_B1_PDL", "0") == "1")
        _B1_PDL.append(1)
    cfg = data["config"]
    if not _B1_OFF and attn_b1_supported(data):
        if cfg["q_len"] == 8:
            return attn_b1_forward(data, splits=_B1V_S, cluster=_B1V_CL)
        return attn_b1_forward(data, splits=_B1_S_SLIDING if cfg["window"] else _B1_S_FULL, depth=_B1_D, cluster=_B1_CL)
    if (_WSD_ON and cfg["q_len"] == 1 and cfg["window"] == 0 and cfg["batch"] >= 2 and _team_v3_supported(data)):  # noqa: F821
        return wsd_forward(data, nlw=_WSD_NLW, rs=_WSD_RS, cps=_WSD_CPS, pdl=False)  # noqa: F821
    return _CONTROL_KERNEL_FN(data)
WSD_FIXREF = True  # fixed-reference atomic merge + last-arriver finalize on batched full-layer decode
# wsd2: batched decode (q_len 1) on full-attention layers in a flat-grid geometry.
#  * Grid: 2 CTAs per SM (3 with WSD2_OCC3) over the concatenated keys of all requests (each request: its seq cached
#    keys + the new token), in a virtual key space with SEGK filler keys before every request after the first, so a CTA
#    that crosses a request boundary is charged for its second segment. A CTA runs the segments its range touches, in
#    order.
#  * CTA: 4 consumer warps (kv head = warp & 1, key half = warp >> 1) and 4 producer warps.
#    Producers stream 64-key stages (K 8 KB + V 8 KB, 128-B rows, 16-B chunk c of row r at c ^ swz(r)) with 16-B
#    cp.async (page ids loaded one stage ahead) and complete on the stage's full barrier; consumers release it.
#  * Consumer math (mma.sync m16n8k16 f16, fp32 accumulate): rows 0-7 of the m16 tile are the kv head's 8 query heads.
#    QK: a lane reads its 16 contiguous key bytes (dims 16 t .. 16 t + 15), the query fragments use the same dim
#    order. PV: a lane reads 8 bytes (dims 8 g .. 8 g + 7) of 4 key rows; n-tile j column n is dim 8 n + j, so a lane
#    ends with dims 16 t .. 16 t + 15 of its head. Scores in log2 units, online softmax per 32 keys. WSD2_T switches to
#    transposed tiles (S^T = K Q^T, O^T = V^T P^T: no padding rows), WSD2_T2 to 32 keys per step.
#  * Merge: per segment, the two key halves of a head combine in shared memory; a row whose log2 offset m is within
#    +-100 adds O 2^m and l 2^m into fp32 accumulators against the absolute reference 2^0 (vector reds); otherwise it
#    keeps an exact partial. Each segment adds its key count to the request's ticket; the CTA that completes it
#    normalizes (folding any exact partials), applies v_scale, the output RMS norm and the sigmoid gate, writes the
#    request's output and zeroes the accumulators, tickets and flags for the next call.
#  * The new token: the CTA whose segment holds position seq normalizes and quantizes its key / value (both heads),
#    writes the cache row at cache_loc and attends it from shared memory.
import zlib

import torch

try:
    lean_load_inline  # noqa: B018  (defined earlier in this file)
except NameError:  # standalone import
    from lean_ext import LEAN_CHECK, lean_load_inline  # noqa: F401

WSD2_SRC = LEAN_CHECK + r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace okcw2 {
namespace {  // internal linkage: probe and plain builds load side by side without sharing template statics
constexpr int HKV = 2, GRP = 8, D = 64, ROWS = HKV * GRP;  // 16 output rows per request
constexpr int SK = 64, HALF = 32, ROWB = 128;              // keys per stage / per consumer warp, bytes per cache row
constexpr int STAGE = 2 * SK * ROWB;                        // K then V
constexpr int NC = 4, NP = 4, NT = 32 * (NC + NP);
constexpr int MAXSLOT = 64;                                 // exact-partial slots per request
constexpr float LOG2E = 1.4426950408889634f;

struct Params {
  const __nv_bfloat16* qkv;   // [B, (2 Hq + 2 HKV) * 64]
  const int64_t* cache_loc;   // [B]
  uint8_t* k_cache;           // [rows, HKV, 64] e4m3
  uint8_t* v_cache;
  const int32_t* page_table;  // [B, max_pages]
  const int32_t* seq_lens;    // [B] cached keys (the new token is position seq)
  float* acc_o;               // [32][16][64] zero at rest
  float* acc_l;               // [32][16]
  float* part;                // [32][MAXSLOT][16][66] exact partials (m, l, O)
  int* cnt;                   // [0, 32) key tickets, [32, 64) exact-slot counts; zero at rest
  __nv_bfloat16* out;         // [B, Hq * 64]
  int B, Hq, page_size, max_pages, segk;
  int window;                 // 0: full layer; else the sliding window (keys [seq + 1 - window, seq], RoPE)
  const float* inv_freq;      // [32] RoPE frequencies (sliding layers)
  float qk_scale, q_mult, inv_k_scale, inv_v_scale;
  long long* probe;
  int skew;                   // low-blockIdx CTAs' share of an SM pair's keys, x 2^16 (32768: even)           // W2_PROBE builds: 16 int64 per CTA (see W2P_*)
};
#ifdef W2_PROBE
// per-CTA probe words: globaltimer at start / after the partition / first data / last stage consumed / CTA end /
// last stage issued; smid; cached keys; segments; consumer warp 0 full-wait cycles; producer warp 0 empty-wait cycles;
// finalizes done
enum { W2P_T0, W2P_TPART, W2P_TDATA, W2P_TLOOP, W2P_TEND, W2P_TISSUE, W2P_SMID, W2P_KEYS, W2P_NSEG, W2P_CWAIT, W2P_PWAIT, W2P_NFIN };
__device__ __forceinline__ long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define W2P(i, v) (p.probe[static_cast<size_t>(blockIdx.x) * 16 + (i)] = static_cast<long long>(v))
#define W2P_DO(...) __VA_ARGS__
#else
#define W2P(i, v) ((void)0)
#define W2P_DO(...)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  uint32_t done;
  do {
    asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                 : "=r"(done) : "r"(smem_u32(b)), "r"(parity) : "memory");
  } while (!done);
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void cp_async_arrive(uint64_t* b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void gdc_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}
__device__ __forceinline__ uint32_t pack_f16x2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void mma_f16x4(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// chunk c of cache row r sits at c ^ swz(r): bit 2 differs between rows 2m / 2m + 1 (the QK phase reads chunks 4 hh .. 4 hh + 3
// of rows g = 2m, 2m + 1) and bits 1-2 differ between rows 0, 2, 4, 6 (the PV phase reads 2 chunks of rows 2 t)
__device__ __forceinline__ int swz(int r) { return (r & 6) ^ ((r & 1) << 2); }
// sin / cos of an fp32 angle with the range reduction in double (the reference's fp32 cos / sin of large angles)
__device__ __forceinline__ void rope_sincos(float x, float* s, float* c) {
  const double xd = static_cast<double>(x);
  const double k = rint(xd * 0.15915494309189535);
  __sincosf(static_cast<float>(fma(-k, 6.283185307179586, xd)), s, c);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  // rows 8-15 of A are zero (a1 = a3 = 0): only c[0], c[1] (rows g) are meaningful
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

#ifdef W2_OCC3
constexpr int W2_MINB = 3, W2_RP = 32, W2_RC = 128;  // 3 CTAs / SM: 4 x 32 + 4 x 128 = 8 x 80
#else
constexpr int W2_MINB = 2, W2_RP = 56, W2_RC = 200;  // 4 x 56 + 4 x 200 = 8 x 128
#endif
template <int ST>
__global__ void __launch_bounds__(NT, W2_MINB) wsd2_kernel(const Params p) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* ring = smem_raw;                                  // [ST][K 8 KB | V 8 KB]
  unsigned char* nstage = ring + ST * STAGE;                       // the new token's stage: K / V rows 0..31, row 0 real
  uint16_t* qs = reinterpret_cast<uint16_t*>(nstage + 2 * HALF * ROWB);  // [HKV][8][64] f16
  float* xm = reinterpret_cast<float*>(qs + HKV * GRP * D);       // half-1 warps: [2][8] m, [2][8] l, [2][8][64] O
  uint64_t* full = reinterpret_cast<uint64_t*>(xm + 2 * 8 * 2 + 2 * 8 * 64);
  uint64_t* empty = full + ST;
  __shared__ int s_seg_b[33], s_seg_k0[33], s_seg_k1[33], s_seg_seq[33], s_seg_lo[33], s_nseg;
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  W2P_DO(if (tid == 0) W2P(W2P_T0, gtimer());)
  // --- partition (warp 0; lane b = request b) -------------------------------------------------------------------
  if (warp == 0) {
    const int B = p.B;
    const int sq = lane < B ? p.seq_lens[lane] : 0;
    const int lo_b = p.window ? max(0, sq + 1 - p.window) : 0;  // the request's first key (sliding window)
    const long long n = lane < B ? static_cast<long long>(sq) + 1 - lo_b : 0;  // keys incl. the new token
    const long long v = lane < B ? n + (lane > 0 ? p.segk : 0) : 0;
    long long inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const long long y = __shfl_up_sync(0xffffffffu, inc, o);
      if (lane >= o) inc += y;
    }
    const long long tv = __shfl_sync(0xffffffffu, inc, 31);
    const long long G = gridDim.x;
    // CTA c's range [F(c) tv, F(c + 1) tv): F rises by skew per CTA over the first half of the grid (dispatched first:
    // one per SM, the faster SM-mate) and by 1 - skew over the second (x 2 / G; skew 1/2: even)
    const long long H = G >> 1, f = p.skew;
    auto F = [&](long long c) -> long long {  // (unnormalized)
      return c <= H ? c * f : H * f + (c - H) * (65536 - f);
    };
    const long long FG = F(G);
    const long long lo = tv * F(blockIdx.x) / FG, hi = tv * F(blockIdx.x + 1) / FG;
    const long long st = inc - v + (lane > 0 ? p.segk : 0), en = inc;  // this request's real keys [st, en)
    const bool meets = lane < B && n > 0 && max(lo, st) < min(hi, en);
    const unsigned m = __ballot_sync(0xffffffffu, meets);
    if (meets) {
      const int idx = __popc(m & ((1u << lane) - 1u));
      if (idx == 0 && p.page_size == 1) {  // the first segment's first two stages of page ids -> L2 (4 lines)
        const int32_t* pf = p.page_table + static_cast<size_t>(lane) * p.max_pages + lo_b + (max(lo, st) - st);
#pragma unroll
        for (int q = 0; q < 4; ++q) asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + 32 * q));
      }
      s_seg_b[idx] = lane;
      s_seg_k0[idx] = static_cast<int>(max(lo, st) - st);
      s_seg_k1[idx] = static_cast<int>(min(hi, en) - st);
      s_seg_seq[idx] = static_cast<int>(n - 1);  // (cached keys, counted from the window start)
      s_seg_lo[idx] = lo_b;
    }
    if (lane == 0) s_nseg = __popc(m);
  }
  if (tid == 0) {
    for (int i = 0; i < ST; ++i) { mbar_init(full + i, 32 * NP); mbar_init(empty + i, NC); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nseg = s_nseg;
  W2P_DO(if (tid == 0) { W2P(W2P_TPART, gtimer()); W2P(W2P_NSEG, nseg); unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm)); W2P(W2P_SMID, sm); })
  if (warp >= NC) {
    // ===================== producers: 64-key stages, thread q: chunk q & 7 of rows (q >> 3) + 16 k ==================
#if (defined(W2_REG) || defined(W2_OCC3)) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(W2_RP) : "memory");  // (balanced: the launch share)
#endif
    const int q = tid - 32 * NC, ch = q & 7, r0 = q >> 3;
    const int sw = (ch ^ swz(r0)) << 4;  // the swizzled chunk is the same for all 4 rows of this thread
    int gst = 0;  // stage counter across segments (ring parity)
    W2P_DO(long long pwait = 0;)
    for (int si = 0; si < nseg; ++si) {
      const int b = s_seg_b[si], k0 = s_seg_k0[si], seq = s_seg_seq[si];
      const int cend = min(s_seg_k1[si], seq);  // cached keys of this segment: [k0, cend)
      const int nst = cend > k0 ? (cend - k0 + SK - 1) / SK : 0;
      const int32_t* pt = p.page_table + static_cast<size_t>(b) * p.max_pages;
      const int lo_b = s_seg_lo[si];
      auto row_of = [&](int pos) -> uint32_t {  // cache row of key pos of the window (0 past the segment's cached keys)
        if (pos >= cend) return 0u;
        const int ap = lo_b + pos;
        return p.page_size == 1 ? static_cast<uint32_t>(__ldg(pt + ap))
                                : static_cast<uint32_t>(__ldg(pt + ap / p.page_size)) * p.page_size + ap % p.page_size;
      };
      // row ids two stages ahead: id0 this stage, id1 the next, id2 loaded now for the one after
      uint32_t id0[4], id1[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) { id0[k] = row_of(k0 + r0 + 16 * k); id1[k] = row_of(k0 + SK + r0 + 16 * k); }
      for (int i = 0; i < nst; ++i, ++gst) {
        uint32_t id2[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) id2[k] = row_of(k0 + SK * (i + 2) + r0 + 16 * k);
        const int sl = gst % ST;
        W2P_DO(const long long w0 = clock64();)
        if (gst >= ST) mbar_wait(empty + sl, ((gst / ST) - 1) & 1);
        W2P_DO(pwait += clock64() - w0;)
        unsigned char* kb = ring + sl * STAGE;
        unsigned char* vb = kb + SK * ROWB;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int r = r0 + 16 * k, pos = k0 + SK * i + r;
          const int bytes = pos < cend ? 16 : 0;
          const size_t off = static_cast<size_t>(id0[k]) * ROWB + ch * 16;
          cp_async16(smem_u32(kb + r * ROWB + sw), p.k_cache + off, bytes);
          cp_async16(smem_u32(vb + r * ROWB + sw), p.v_cache + off, bytes);
        }
        cp_async_arrive(full + sl);
#pragma unroll
        for (int k = 0; k < 4; ++k) { id0[k] = id1[k]; id1[k] = id2[k]; }
      }
    }
    W2P_DO(if (q == 0) { W2P(W2P_TISSUE, gtimer()); W2P(W2P_PWAIT, pwait); })
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }
  // ========================= consumers ===========================================================================
#if (defined(W2_REG) || defined(W2_OCC3)) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(W2_RC) : "memory");
#endif
  const int hh = warp & 1, kh = warp >> 1;  // kv head, key half of each stage
  const int ctid = tid;                     // 0 .. 127
  const int qkv_stride = (2 * p.Hq + 2 * HKV) * D;
  // the new token's stage rows 1..31 stay zero (masked keys still multiply V: no garbage)
  for (int i = ctid; i < 2 * HALF * ROWB / 16; i += 32 * NC) reinterpret_cast<uint4*>(nstage)[i] = make_uint4(0u, 0u, 0u, 0u);
  gdc_wait();  // queries / the new token come from the previous kernel
  int gst = 0;
  W2P_DO(long long cwait = 0, keys = 0; int nfin = 0; bool first = true;)
  for (int si = 0; si < nseg; ++si) {
    const int b = s_seg_b[si], k0 = s_seg_k0[si], k1 = s_seg_k1[si], seq = s_seg_seq[si];
    const int cend = min(k1, seq);
    const int nst = cend > k0 ? (cend - k0 + SK - 1) / SK : 0;
    const bool has_new = k1 > seq;  // this segment holds position seq (the new token)
    W2P_DO(keys += max(0, cend - k0);)
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");  // the previous segment's readers of qs / xm are done
    // --- queries of request b: rms * qk_scale -> bf16 -> * q_mult -> f16 (16 rows x 8 threads x 8 dims) ----------
    {
      const int row = ctid >> 3, part = ctid & 7, head = (row >> 3) * (p.Hq / HKV) + (row & 7);
      const __nv_bfloat16* qp = p.qkv + static_cast<size_t>(b) * qkv_stride + head * D + 8 * part;
      const uint4 u = *reinterpret_cast<const uint4*>(qp);
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
      float v[8], ss = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b2[k]); v[2 * k] = f.x; v[2 * k + 1] = f.y; ss += f.x * f.x + f.y * f.y; }
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f) * p.qk_scale;
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] *= inv;
      if (p.window) {  // RoPE at position seq (pairs 2 i, 2 i + 1: i = 4 part + k)
        const float pos = static_cast<float>(s_seg_lo[si] + seq);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          float sn, cs;
          rope_sincos(pos * __ldg(p.inv_freq + 4 * part + k), &sn, &cs);
          const float e = v[2 * k], o = v[2 * k + 1];
          v[2 * k] = e * cs - o * sn;
          v[2 * k + 1] = e * sn + o * cs;
        }
      }
      uint32_t* dst = reinterpret_cast<uint32_t*>(qs + row * D + 8 * part);
#pragma unroll
      for (int k = 0; k < 4; ++k) dst[k] = pack_f16x2(bf16_round(v[2 * k]) * p.q_mult, bf16_round(v[2 * k + 1]) * p.q_mult);
    }
    // --- the new token: normalized / quantized key and value of both heads, cache row + the new stage's row 0 ------
    if (has_new && ctid < HKV * 8) {
      const int hk = ctid >> 3, c = ctid & 7;  // head, 8-byte chunk
      const __nv_bfloat16* row = p.qkv + static_cast<size_t>(b) * qkv_stride;
      const __nv_bfloat16* kin = row + p.Hq * D + hk * D;
      const __nv_bfloat16* vin = row + p.Hq * D + HKV * D + hk * D;
      float ss = 0.f;
#pragma unroll
      for (int d = 0; d < D; d += 8) {
        const uint4 u = *reinterpret_cast<const uint4*>(kin + d);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b2[k]); ss += f.x * f.x + f.y * f.y; }
      }
      const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const uint4 ku = *reinterpret_cast<const uint4*>(kin + 8 * c);
      const uint4 vu = *reinterpret_cast<const uint4*>(vin + 8 * c);
      const __nv_bfloat162* kb2 = reinterpret_cast<const __nv_bfloat162*>(&ku);
      const __nv_bfloat162* vb2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
      uint8_t kq[8], vq[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 kf = __bfloat1622float2(kb2[k]);
        const float2 vf = __bfloat1622float2(vb2[k]);
        kf.x *= inv; kf.y *= inv;
        if (p.window) {  // RoPE at position seq (pair i = 4 c + k)
          float sn, cs;
          rope_sincos(static_cast<float>(s_seg_lo[si] + seq) * __ldg(p.inv_freq + 4 * c + k), &sn, &cs);
          const float e = kf.x, o = kf.y;
          kf.x = e * cs - o * sn;
          kf.y = e * sn + o * cs;
        }
        kq[2 * k] = __nv_cvt_float_to_fp8(kf.x * p.inv_k_scale, __NV_SATFINITE, __NV_E4M3);
        kq[2 * k + 1] = __nv_cvt_float_to_fp8(kf.y * p.inv_k_scale, __NV_SATFINITE, __NV_E4M3);
        vq[2 * k] = __nv_cvt_float_to_fp8(vf.x * p.inv_v_scale, __NV_SATFINITE, __NV_E4M3);
        vq[2 * k + 1] = __nv_cvt_float_to_fp8(vf.y * p.inv_v_scale, __NV_SATFINITE, __NV_E4M3);
      }
      const uint2 kw = *reinterpret_cast<uint2*>(kq), vw = *reinterpret_cast<uint2*>(vq);
      const int64_t loc = p.cache_loc[b];
      *reinterpret_cast<uint2*>(p.k_cache + static_cast<size_t>(loc) * ROWB + hk * D + 8 * c) = kw;
      *reinterpret_cast<uint2*>(p.v_cache + static_cast<size_t>(loc) * ROWB + hk * D + 8 * c) = vw;
      // row 0 of the new stage (swizzle of row 0 is the identity)
      *reinterpret_cast<uint2*>(nstage + hk * D + 8 * c) = kw;
      *reinterpret_cast<uint2*>(nstage + HALF * ROWB + hk * D + 8 * c) = vw;
    }
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
    // this lane's query fragments: row g of head hh, dims 16 t .. 16 t + 15 (the QK dim order)
    uint32_t qa[8];
    {
      const uint4* qr = reinterpret_cast<const uint4*>(qs + (hh * GRP + g) * D + 16 * t);
      const uint4 x0 = qr[0], x1 = qr[1];
      qa[0] = x0.x; qa[1] = x0.y; qa[2] = x0.z; qa[3] = x0.w; qa[4] = x1.x; qa[5] = x1.y; qa[6] = x1.z; qa[7] = x1.w;
    }
#ifdef W2_T
    // --- transposed tiles: S^T = K Q^T (m16 = 16 keys, n8 = 8 heads), O^T = V^T P^T (m16 = 16 dims): no padding rows.
    //     Lane (g, t) owns heads 2 t, 2 t + 1 of kv head hh: S^T keys g, g + 8; O^T dims 8 g + 2 mt (+ 1), mt = 0..3
    float ot[4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) ot[mt][0] = ot[mt][1] = ot[mt][2] = ot[mt][3] = 0.f;
    float mh[2] = {-INFINITY, -INFINITY}, lh[2] = {0.f, 0.f};  // heads 2 t, 2 t + 1: running log2 offset, partial l
    auto tile = [&](const unsigned char* kb, const unsigned char* vb, int krow0, int nvalid, bool sw) {
      uint32_t kf[2][8];  // keys g (u 0), g + 8 (u 1): dims 16 t .. 16 t + 15 of head hh (the QK dim order)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int r = krow0 + 8 * u + g;
        const int cofs = (sw ? ((4 * hh + t) ^ swz(r)) : (4 * hh + t)) << 4;
        const uint4 kr = *reinterpret_cast<const uint4*>(kb + r * ROWB + cofs);
        const uint16_t* kp = reinterpret_cast<const uint16_t*>(&kr);
#pragma unroll
        for (int q = 0; q < 8; ++q) kf[u][q] = e4m3x2_f16x2(kp[q]);
      }
      uint32_t vf[4][4];  // keys 2 t, 2 t + 1, 2 t + 8, 2 t + 9: dims 8 g .. 8 g + 7
#pragma unroll
      for (int rr = 0; rr < 4; ++rr) {
        const int r = krow0 + (rr < 2 ? 2 * t + rr : 2 * t + 8 + (rr - 2));
        const int c = 4 * hh + (g >> 1);
        const int cofs = ((sw ? (c ^ swz(r)) : c) << 4) + 8 * (g & 1);
        const uint2 vr = *reinterpret_cast<const uint2*>(vb + r * ROWB + cofs);
        const uint16_t* vp = reinterpret_cast<const uint16_t*>(&vr);
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) vf[rr][qq] = e4m3x2_f16x2(vp[qq]);
      }
      float st[4] = {0.f, 0.f, 0.f, 0.f};  // (key g, head 2 t), (g, 2 t + 1), (g + 8, 2 t), (g + 8, 2 t + 1)
#pragma unroll
      for (int s4 = 0; s4 < 4; ++s4) mma_f16x4(st, kf[0][2 * s4], kf[1][2 * s4], kf[0][2 * s4 + 1], kf[1][2 * s4 + 1], qa[2 * s4], qa[2 * s4 + 1]);
      const bool v0 = g < nvalid, v1 = g + 8 < nvalid;
      const float s0 = v0 ? st[0] * LOG2E : -INFINITY, s1 = v0 ? st[1] * LOG2E : -INFINITY;
      const float s2 = v1 ? st[2] * LOG2E : -INFINITY, s3 = v1 ? st[3] * LOG2E : -INFINITY;
      float tm0 = fmaxf(s0, s2), tm1 = fmaxf(s1, s3);
#pragma unroll
      for (int o2 = 4; o2 < 32; o2 <<= 1) {
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, o2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, o2));
      }
      const float mn0 = fmaxf(mh[0], tm0), mn1 = fmaxf(mh[1], tm1);  // (finite: the caller passes nvalid >= 1)
      const float a0 = exp2f(mh[0] - mn0), a1 = exp2f(mh[1] - mn1);  // (mh = -inf: 0)
      mh[0] = mn0; mh[1] = mn1;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) { ot[mt][0] *= a0; ot[mt][1] *= a1; ot[mt][2] *= a0; ot[mt][3] *= a1; }
      const float e0 = exp2f(s0 - mn0), e1 = exp2f(s1 - mn1), e2 = exp2f(s2 - mn0), e3 = exp2f(s3 - mn1);  // (-inf: 0)
      lh[0] = lh[0] * a0 + (e0 + e2);
      lh[1] = lh[1] * a1 + (e1 + e3);
      uint32_t pb0, pb1;  // P^T B fragments: (head g; keys 2 t, 2 t + 1) / (head g; keys 2 t + 8, 2 t + 9)
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(pb0) : "r"(pack_f16x2(e0, e1)));
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(pb1) : "r"(pack_f16x2(e2, e3)));
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {  // A = V^T rows dim 8 g + 2 mt (g) / + 1 (g + 8), keys (2 t, 2 t + 1) / (+ 8)
        const uint32_t sl0 = 0x5410u, sl1 = 0x7632u;
        mma_f16x4(ot[mt], __byte_perm(vf[0][mt], vf[1][mt], sl0), __byte_perm(vf[0][mt], vf[1][mt], sl1),
                  __byte_perm(vf[2][mt], vf[3][mt], sl0), __byte_perm(vf[2][mt], vf[3][mt], sl1), pb0, pb1);
      }
    };
#ifdef W2_T2
    // 32 keys at once (sub-tiles u2 = 0 / 1 = keys krow0 + 16 u2 ..): two independent QK chains, one max / rescale,
    // PV as 2 k-steps per m-tile
    auto tile32 = [&](const unsigned char* kb, const unsigned char* vb, int krow0, int nvalid) {
      float st[2][4];
      uint32_t vf[2][4][4];
#pragma unroll
      for (int u2 = 0; u2 < 2; ++u2) {
        uint32_t kf[2][8];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int r = krow0 + 16 * u2 + 8 * u + g;
          const uint4 kr = *reinterpret_cast<const uint4*>(kb + r * ROWB + (((4 * hh + t) ^ swz(r)) << 4));
          const uint16_t* kp = reinterpret_cast<const uint16_t*>(&kr);
#pragma unroll
          for (int q = 0; q < 8; ++q) kf[u][q] = e4m3x2_f16x2(kp[q]);
        }
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
          const int r = krow0 + 16 * u2 + (rr < 2 ? 2 * t + rr : 2 * t + 8 + (rr - 2));
          const int c = 4 * hh + (g >> 1);
          const uint2 vr = *reinterpret_cast<const uint2*>(vb + r * ROWB + ((c ^ swz(r)) << 4) + 8 * (g & 1));
          const uint16_t* vp = reinterpret_cast<const uint16_t*>(&vr);
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) vf[u2][rr][qq] = e4m3x2_f16x2(vp[qq]);
        }
        st[u2][0] = st[u2][1] = st[u2][2] = st[u2][3] = 0.f;
#pragma unroll
        for (int s4 = 0; s4 < 4; ++s4) mma_f16x4(st[u2], kf[0][2 * s4], kf[1][2 * s4], kf[0][2 * s4 + 1], kf[1][2 * s4 + 1], qa[2 * s4], qa[2 * s4 + 1]);
      }
      float sv[2][4];
      float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
      for (int u2 = 0; u2 < 2; ++u2) {
        const bool v0 = 16 * u2 + g < nvalid, v1 = 16 * u2 + g + 8 < nvalid;
        sv[u2][0] = v0 ? st[u2][0] * LOG2E : -INFINITY; sv[u2][1] = v0 ? st[u2][1] * LOG2E : -INFINITY;
        sv[u2][2] = v1 ? st[u2][2] * LOG2E : -INFINITY; sv[u2][3] = v1 ? st[u2][3] * LOG2E : -INFINITY;
        tm0 = fmaxf(tm0, fmaxf(sv[u2][0], sv[u2][2]));
        tm1 = fmaxf(tm1, fmaxf(sv[u2][1], sv[u2][3]));
      }
#pragma unroll
      for (int o2 = 4; o2 < 32; o2 <<= 1) {
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, o2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, o2));
      }
      const float mn0 = fmaxf(mh[0], tm0), mn1 = fmaxf(mh[1], tm1);
      const float a0 = exp2f(mh[0] - mn0), a1 = exp2f(mh[1] - mn1);
      mh[0] = mn0; mh[1] = mn1;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) { ot[mt][0] *= a0; ot[mt][1] *= a1; ot[mt][2] *= a0; ot[mt][3] *= a1; }
      lh[0] *= a0; lh[1] *= a1;
#pragma unroll
      for (int u2 = 0; u2 < 2; ++u2) {
        const float e0 = exp2f(sv[u2][0] - mn0), e1 = exp2f(sv[u2][1] - mn1), e2 = exp2f(sv[u2][2] - mn0), e3 = exp2f(sv[u2][3] - mn1);
        lh[0] += e0 + e2;
        lh[1] += e1 + e3;
        uint32_t pb0, pb1;
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(pb0) : "r"(pack_f16x2(e0, e1)));
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(pb1) : "r"(pack_f16x2(e2, e3)));
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          const uint32_t sl0 = 0x5410u, sl1 = 0x7632u;
          mma_f16x4(ot[mt], __byte_perm(vf[u2][0][mt], vf[u2][1][mt], sl0), __byte_perm(vf[u2][0][mt], vf[u2][1][mt], sl1),
                    __byte_perm(vf[u2][2][mt], vf[u2][3][mt], sl0), __byte_perm(vf[u2][2][mt], vf[u2][3][mt], sl1), pb0, pb1);
        }
      }
    };
#endif
    for (int i = 0; i < nst; ++i, ++gst) {
      const int sl = gst % ST;
      W2P_DO(const long long w0 = clock64();)
      mbar_wait(full + sl, (gst / ST) & 1);
      W2P_DO(cwait += clock64() - w0; if (first && ctid == 0) { W2P(W2P_TDATA, gtimer()); first = false; })
      const unsigned char* kb = ring + sl * STAGE;
      const int nv = cend - (k0 + SK * i + HALF * kh);
#ifdef W2_T2
      if (nv > 0) tile32(kb, kb + SK * ROWB, HALF * kh, min(HALF, nv));
#else
      if (nv > 0) tile(kb, kb + SK * ROWB, HALF * kh, min(16, nv), true);
      if (nv > 16) tile(kb, kb + SK * ROWB, HALF * kh + 16, min(16, nv - 16), true);
#endif
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + sl);
    }
    W2P_DO(if (si == nseg - 1 && ctid == 0) W2P(W2P_TLOOP, gtimer());)
    if (si == nseg - 1) gdc_launch();  // dependents launch once the gathers are done (producers count by exiting)
    if (has_new && kh == 0) tile(nstage, nstage + HALF * ROWB, 0, 1, false);
    // --- combine the key halves (warps kh = 1 -> shared memory -> kh = 0); l over the lanes g of each head first ------
#pragma unroll
    for (int o2 = 4; o2 < 32; o2 <<= 1) { lh[0] += __shfl_xor_sync(0xffffffffu, lh[0], o2); lh[1] += __shfl_xor_sync(0xffffffffu, lh[1], o2); }
    float* xmh = xm + 2 * 8 * 2 + hh * 8 * 64;  // [head][dim]
    if (kh == 1) {
      if (g == 0) {
#pragma unroll
        for (int e = 0; e < 2; ++e) { xm[hh * 8 + 2 * t + e] = mh[e]; xm[16 + hh * 8 + 2 * t + e] = lh[e]; }
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int q = 0; q < 4; ++q) xmh[(2 * t + (q & 1)) * 64 + 8 * g + 2 * mt + (q >> 1)] = ot[mt][q];
    }
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
    if (kh == 0) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {  // head 2 t + e: dims 8 g + d, d = 2 mt + q -> ot[mt][2 q + e]
        const int head = 2 * t + e, row = hh * GRP + head;
        const float m1 = xm[hh * 8 + head], l1 = xm[16 + hh * 8 + head];
        const float mm = fmaxf(mh[e], m1);
        const float c0 = (mh[e] == -INFINITY) ? 0.f : exp2f(mh[e] - mm), c1 = (m1 == -INFINITY) ? 0.f : exp2f(m1 - mm);
        const float ll = lh[e] * c0 + l1 * c1;
        float acc[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[d] = ot[d >> 1][2 * (d & 1) + e] * c0 + xmh[head * 64 + 8 * g + d] * c1;
        if (mm != -INFINITY && fabsf(mm) <= 100.f) {  // (uniform over the 8 lanes g of this head)
          const float f = exp2f(mm);
          float* ao = p.acc_o + (static_cast<size_t>(b) * ROWS + row) * D + 8 * g;
          asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(ao), "f"(acc[0] * f), "f"(acc[1] * f),
                       "f"(acc[2] * f), "f"(acc[3] * f) : "memory");
          asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(ao + 4), "f"(acc[4] * f), "f"(acc[5] * f),
                       "f"(acc[6] * f), "f"(acc[7] * f) : "memory");
          if (g == 0) atomicAdd(p.acc_l + b * ROWS + row, ll * f);
        } else if (mm != -INFINITY) {  // an exact partial for the last CTA to fold (rare: logits beyond 2^100)
          int slot = 0;
          if (g == 0) slot = atomicAdd(p.cnt + 32 + b, 1);
          slot = min(__shfl_sync(__activemask(), slot, t), MAXSLOT - 1);
          float* pr = p.part + ((static_cast<size_t>(b) * MAXSLOT + slot) * ROWS + row) * 66;
          if (g == 0) { pr[0] = mm; pr[1] = ll; }
#pragma unroll
          for (int d = 0; d < 8; ++d) pr[2 + 8 * g + d] = acc[d];
        }
      }
      __threadfence();
    }
#else
    float o[8][2];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j][0] = o[j][1] = 0.f;
    float mrow = -INFINITY, lrow = 0.f;  // this lane's row g: running log2 offset, partial sum of P
    // one 32-key half (keys key0 .. key0 + 31 of a stage at kb / vb; positions >= vend are masked)
    auto half = [&](const unsigned char* kb, const unsigned char* vb, int krow0, int nvalid, bool sw) {
      float s[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int r = krow0 + 8 * nt + g;
        const int cofs = (sw ? ((4 * hh + t) ^ swz(r)) : (4 * hh + t)) << 4;
        const uint4 kr = *reinterpret_cast<const uint4*>(kb + r * ROWB + cofs);
        const uint16_t* kp = reinterpret_cast<const uint16_t*>(&kr);
        s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int st = 0; st < 4; ++st) mma_f16(s[nt], qa[2 * st], qa[2 * st + 1], e4m3x2_f16x2(kp[2 * st]), e4m3x2_f16x2(kp[2 * st + 1]));
      }
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int kk = 8 * nt + 2 * t + e;
          const float v = kk < nvalid ? s[nt][e] * LOG2E : -INFINITY;
          s[nt][e] = v;
          mx = fmaxf(mx, v);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mn = fmaxf(mrow, mx);
      if (mn == -INFINITY) return;  // nothing valid yet
      const float a = (mrow == -INFINITY) ? 0.f : exp2f(mrow - mn);
      mrow = mn;
      lrow *= a;
#pragma unroll
      for (int j = 0; j < 8; ++j) { o[j][0] *= a; o[j][1] *= a; }
      float pv[4][2];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float e2 = s[nt][e] == -INFINITY ? 0.f : exp2f(s[nt][e] - mn);
          pv[nt][e] = e2;
          lrow += e2;
        }
      // PV: k-step ks = keys 16 ks .. 16 ks + 15 of the half; A = P (a0: keys 2t, 2t+1; a2: keys 2t+8, 2t+9)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint32_t pa0 = pack_f16x2(pv[2 * ks][0], pv[2 * ks][1]), pa2 = pack_f16x2(pv[2 * ks + 1][0], pv[2 * ks + 1][1]);
        uint32_t vf[4][4];  // key rows 2t, 2t+1, 2t+8, 2t+9: dims 8 g .. 8 g + 7 of head hh, f16
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
          const int r = krow0 + 16 * ks + (rr < 2 ? 2 * t + rr : 2 * t + 8 + (rr - 2));
          const int c = 4 * hh + (g >> 1);
          const int cofs = ((sw ? (c ^ swz(r)) : c) << 4) + 8 * (g & 1);
          const uint2 vr = *reinterpret_cast<const uint2*>(vb + r * ROWB + cofs);
          const uint16_t* vp = reinterpret_cast<const uint16_t*>(&vr);
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) vf[rr][qq] = e4m3x2_f16x2(vp[qq]);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {  // n-tile j: column g = dim 8 g + j
          const uint32_t sel = (j & 1) ? 0x7632u : 0x5410u;
          const uint32_t b0 = __byte_perm(vf[0][j >> 1], vf[1][j >> 1], sel), b1 = __byte_perm(vf[2][j >> 1], vf[3][j >> 1], sel);
          float c4[4] = {o[j][0], o[j][1], 0.f, 0.f};
          mma_f16(c4, pa0, pa2, b0, b1);
          o[j][0] = c4[0]; o[j][1] = c4[1];
        }
      }
    };
    for (int i = 0; i < nst; ++i, ++gst) {
      const int sl = gst % ST;
      W2P_DO(const long long w0 = clock64();)
      mbar_wait(full + sl, (gst / ST) & 1);
      W2P_DO(cwait += clock64() - w0; if (first && ctid == 0) { W2P(W2P_TDATA, gtimer()); first = false; })
      const unsigned char* kb = ring + sl * STAGE;
      const int nvalid = min(HALF, cend - (k0 + SK * i + HALF * kh));
      if (nvalid > 0) half(kb, kb + SK * ROWB, HALF * kh, nvalid, true);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + sl);
    }
    W2P_DO(if (si == nseg - 1 && ctid == 0) W2P(W2P_TLOOP, gtimer());)
    if (si == nseg - 1) gdc_launch();  // dependents launch once the gathers are done (producers count by exiting)
    if (has_new && kh == 0) half(nstage, nstage + HALF * ROWB, 0, 1, false);
    // --- combine the two key halves of each head (half 1 -> shared memory -> half 0) -----------------------------
    // lane (g, t) holds O[g][16 t + j] (o[j][0]) and O[g][16 t + 8 + j] (o[j][1]); l is the lane's partial sum
    lrow += __shfl_xor_sync(0xffffffffu, lrow, 1);
    lrow += __shfl_xor_sync(0xffffffffu, lrow, 2);
    float* xmh = xm + 2 * 8 * 2 + hh * 8 * 64;
    if (kh == 1) {
      if (t == 0) { xm[hh * 8 + g] = mrow; xm[16 + hh * 8 + g] = lrow; }
#pragma unroll
      for (int j = 0; j < 8; ++j) { xmh[g * 64 + 16 * t + j] = o[j][0]; xmh[g * 64 + 16 * t + 8 + j] = o[j][1]; }
    }
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
    if (kh == 0) {
      const float m1 = xm[hh * 8 + g], l1 = xm[16 + hh * 8 + g];
      const float mm = fmaxf(mrow, m1);
      const float c0 = (mrow == -INFINITY) ? 0.f : exp2f(mrow - mm), c1 = (m1 == -INFINITY) ? 0.f : exp2f(m1 - mm);
      const float ll = lrow * c0 + l1 * c1;
      float acc[16];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[j] = o[j][0] * c0 + xmh[g * 64 + 16 * t + j] * c1;
        acc[8 + j] = o[j][1] * c0 + xmh[g * 64 + 16 * t + 8 + j] * c1;
      }
      const int row = hh * GRP + g;
      if (mm != -INFINITY && fabsf(mm) <= 100.f) {
        const float f = exp2f(mm);
        float* ao = p.acc_o + (static_cast<size_t>(b) * ROWS + row) * D + 16 * t;
#pragma unroll
        for (int d = 0; d < 16; d += 4)
          asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(ao + d), "f"(acc[d] * f), "f"(acc[d + 1] * f),
                       "f"(acc[d + 2] * f), "f"(acc[d + 3] * f) : "memory");
        if (t == 0) atomicAdd(p.acc_l + b * ROWS + row, ll * f);
      } else if (mm != -INFINITY) {  // an exact partial for the last CTA to fold (rare: logits beyond 2^100)
        int slot = 0;
        if (t == 0) slot = atomicAdd(p.cnt + 32 + b, 1);
        slot = min(__shfl_sync(__activemask(), slot, lane & ~3), MAXSLOT - 1);
        float* pr = p.part + ((static_cast<size_t>(b) * MAXSLOT + slot) * ROWS + row) * 66;
        if (t == 0) { pr[0] = mm; pr[1] = ll; }
#pragma unroll
        for (int d = 0; d < 16; ++d) pr[2 + 16 * t + d] = acc[d];
      }
      __threadfence();
    }
#endif
    // --- ticket: this segment's keys; the CTA that completes the request finalizes it -----------------------------
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
    if (ctid == 0) {
      const int mine = k1 - k0;
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p.cnt + b), "r"(mine) : "memory");
      s_last = (old + mine == seq + 1);
    }
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
    if (s_last) {
      W2P_DO(++nfin;)
      __threadfence();
      const int row = ctid >> 3, part = ctid & 7;  // 16 rows x 8 threads x 8 dims
      float* ao = p.acc_o + (static_cast<size_t>(b) * ROWS + row) * D + 8 * part;
      const float4 a0 = __ldcg(reinterpret_cast<const float4*>(ao)), a1 = __ldcg(reinterpret_cast<const float4*>(ao + 4));
      float L = __ldcg(p.acc_l + b * ROWS + row);
      float ac[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      float M = L > 0.f ? 0.f : -INFINITY;
      const int nslot = min(__ldcg(p.cnt + 32 + b), MAXSLOT);
      for (int sl2 = 0; sl2 < nslot; ++sl2) {
        const float* pr = p.part + ((static_cast<size_t>(b) * MAXSLOT + sl2) * ROWS + row) * 66;
        const float m2 = __ldcg(pr), l2 = __ldcg(pr + 1);
        if (!(m2 > -INFINITY)) continue;
        const float mn = fmaxf(M, m2);
        const float ca = (M == -INFINITY) ? 0.f : exp2f(M - mn), cb = exp2f(m2 - mn);
        M = mn;
        L = L * ca + l2 * cb;
#pragma unroll
        for (int d = 0; d < 8; ++d) ac[d] = ac[d] * ca + __ldcg(pr + 2 + 8 * part + d) * cb;
      }
      const float inv = (L > 0.f ? 1.f / L : 0.f) / p.inv_v_scale;
      float ss = 0.f;
#pragma unroll
      for (int d = 0; d < 8; ++d) { ac[d] *= inv; ss += ac[d] * ac[d]; }
#pragma unroll
      for (int o2 = 1; o2 < 8; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const int head = (row >> 3) * (p.Hq / HKV) + (row & 7);
      const __nv_bfloat16* gp = p.qkv + static_cast<size_t>(b) * qkv_stride + (p.Hq + 2 * HKV) * D + head * D + 8 * part;
      const uint4 gu = *reinterpret_cast<const uint4*>(gp);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gu);
      uint4 ov;
      __nv_bfloat162* o2v = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 gf = __bfloat1622float2(g2[k]);
        o2v[k] = __floats2bfloat162_rn(ac[2 * k] * rn / (1.f + __expf(-gf.x)), ac[2 * k + 1] * rn / (1.f + __expf(-gf.y)));
      }
      *reinterpret_cast<uint4*>(p.out + static_cast<size_t>(b) * (p.Hq * D) + head * D + 8 * part) = ov;
      *reinterpret_cast<float4*>(ao) = make_float4(0.f, 0.f, 0.f, 0.f);  // accumulators back to zero for the next call
      *reinterpret_cast<float4*>(ao + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0) p.acc_l[b * ROWS + row] = 0.f;
      for (int sl2 = 0; sl2 < nslot; ++sl2)  // exact slots back to empty
        if (part == 0) p.part[((static_cast<size_t>(b) * MAXSLOT + sl2) * ROWS + row) * 66] = -INFINITY;
      asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
      if (ctid == 0) { p.cnt[b] = 0; p.cnt[32 + b] = 0; }
    }
  }
  W2P_DO(if (ctid == 0) { W2P(W2P_TEND, gtimer()); W2P(W2P_KEYS, keys); W2P(W2P_CWAIT, cwait); W2P(W2P_NFIN, nfin); })
}

template <int ST>
const char* launch(const Params& prm, int grid, bool pdl, cudaStream_t stream) {
  const int bytes = ST * STAGE + 2 * HALF * ROWB + HKV * GRP * D * 2 + (2 * 8 * 2 + 2 * 8 * 64) * 4 + 2 * ST * 8;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(wsd2_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); attr = true; }
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  lc.attrs = at; lc.numAttrs = 1; lc.stream = stream;
  lc.gridDim = dim3(grid); lc.blockDim = dim3(NT); lc.dynamicSmemBytes = bytes;
  const cudaError_t err = cudaLaunchKernelEx(&lc, wsd2_kernel<ST>, prm);
  if (err != cudaSuccess) { (void)cudaGetLastError(); return cudaGetErrorString(err); }
  return nullptr;
}
}  // namespace
}  // namespace okcw2

void wsd2(int64_t qkv, int64_t cache_loc, int64_t k_cache, int64_t v_cache, int64_t page_table, int64_t seq_lens,
          int64_t acc_o, int64_t acc_l, int64_t part, int64_t cnt, int64_t out, int64_t B, int64_t Hq, int64_t page_size,
          int64_t max_pages, int64_t segk, double qk_scale, double sm_scale, double k_scale, double v_scale, int64_t grid,
          int64_t stages, bool pdl, int64_t probe, int64_t window, int64_t inv_freq, int64_t skew, int64_t stream) {
  using namespace okcw2;
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);
  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);
  p.page_table = reinterpret_cast<const int32_t*>(page_table);
  p.seq_lens = reinterpret_cast<const int32_t*>(seq_lens);
  p.acc_o = reinterpret_cast<float*>(acc_o);
  p.acc_l = reinterpret_cast<float*>(acc_l);
  p.part = reinterpret_cast<float*>(part);
  p.cnt = reinterpret_cast<int*>(cnt);
  p.out = reinterpret_cast<__nv_bfloat16*>(out);
  p.B = static_cast<int>(B); p.Hq = static_cast<int>(Hq); p.page_size = static_cast<int>(page_size);
  p.max_pages = static_cast<int>(max_pages); p.segk = static_cast<int>(segk);
  p.qk_scale = static_cast<float>(qk_scale);
  p.q_mult = static_cast<float>(sm_scale * k_scale);
  p.inv_k_scale = static_cast<float>(1.0 / k_scale);
  p.inv_v_scale = static_cast<float>(1.0 / v_scale);
  p.probe = reinterpret_cast<long long*>(probe);
  p.skew = static_cast<int>(skew);
  p.window = static_cast<int>(window);
  p.inv_freq = reinterpret_cast<const float*>(inv_freq);
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  TORCH_CHECK(stages >= 2 && stages <= 6, "wsd2: stages must be 2 .. 6");
  const int g = static_cast<int>(grid);
  const char* e = stages == 3 ? launch<3>(p, g, pdl, st) : stages == 4 ? launch<4>(p, g, pdl, st)
                  : stages == 5 ? launch<5>(p, g, pdl, st) : stages == 6 ? launch<6>(p, g, pdl, st) : launch<2>(p, g, pdl, st);
  TORCH_CHECK(e == nullptr, "wsd2 launch failed: ", e ? e : "");
}
"""

WSD2_CPP = ("void wsd2(int64_t qkv, int64_t cache_loc, int64_t k_cache, int64_t v_cache, int64_t page_table, "
            "int64_t seq_lens, int64_t acc_o, int64_t acc_l, int64_t part, int64_t cnt, int64_t out, int64_t B, int64_t Hq, "
            "int64_t page_size, int64_t max_pages, int64_t segk, double qk_scale, double sm_scale, double k_scale, "
            "double v_scale, int64_t grid, int64_t stages, bool pdl, int64_t probe, int64_t window, int64_t inv_freq, "
            "int64_t skew, int64_t stream);")
_W2 = {}
WSD2_STAGES = 3  # (two CTAs per SM on the H200)
WSD2_SEGK = 400
WSD2_PROBE = False  # probe build: 16 int64 per CTA (wsd2_forward(..., probe=[]) appends the [grid, 16] tensor)
WSD2_T = False  # transposed tiles: S^T = K Q^T / O^T = V^T P^T (half the MMAs, no padding rows)
WSD2_REG = False  # setmaxnreg 56 / 200 (producers / consumers) on Hopper
WSD2_T2 = False  # (with WSD2_T) a warp's 32 keys per step: both QK chains, one max / rescale
WSD2_SKEW = 0.5  # low-blockIdx CTAs' share of an SM pair's keys (0.5: even)
WSD2_OCC3 = False  # 3 CTAs per SM (stages 3, setmaxnreg 32 / 128): 12 consumer warps per SM


def wsd2_ext():
    key = ("e", bool(WSD2_PROBE), bool(WSD2_T), bool(WSD2_REG), bool(WSD2_T and WSD2_T2), bool(WSD2_OCC3))
    if key not in _W2:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}{'a' if major == 9 else ''}"
        src = ("#define W2_PROBE 1\n" if WSD2_PROBE else "") + ("#define W2_T 1\n" if WSD2_T else "") + (
            "#define W2_REG 1\n" if WSD2_REG else "") + ("#define W2_T2 1\n" if WSD2_T and WSD2_T2 else "") + (
            "#define W2_OCC3 1\n" if WSD2_OCC3 else "") + WSD2_SRC
        name = "okc_wsd2_" + format(zlib.crc32(src.encode()) & 0xFFFFFFFF, "08x")
        _W2[key] = lean_load_inline(name, cpp_sources=WSD2_CPP, cuda_sources=src, functions=["wsd2"],
                                    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"])
    return _W2[key]


def _w2_inv_freq(theta, dev):
    key = ("f", theta, str(dev))
    if key not in _W2:  # (the reference's RoPE frequencies, fp32)
        _W2[key] = ((1.0 / (theta ** (torch.arange(0, 64, 2, device=dev).float() / 64))) if theta else
                    torch.zeros(32, device=dev)).contiguous()
    return _W2[key]


def wsd2_forward(data, out=None, pdl=True, grid=None, stages=None, segk=None, probe=None, skew=None):
    """Batched decode (q_len 1) on a full layer: [B, 1, Hq * 64] bf16."""
    cfg = data["config"]
    B, Hq = cfg["batch"], cfg["num_q_heads"]
    assert cfg["q_len"] == 1 and cfg["num_kv_heads"] == 2 and B <= 32 and (cfg["window"] == 0) == (not cfg["rope_theta"])
    qkv = data["qkv"].contiguous()
    dev = qkv.device
    if out is None:
        out = torch.empty((B, 1, Hq * 64), dtype=torch.bfloat16, device=dev)
    key = str(dev)
    if key not in _W2:
        _W2[key] = (torch.zeros((32, 16, 64), dtype=torch.float32, device=dev), torch.zeros((32, 16), dtype=torch.float32, device=dev),
                    torch.full((32, 64, 16, 66), float("-inf"), dtype=torch.float32, device=dev),
                    torch.zeros(64, dtype=torch.int32, device=dev))
    acc_o, acc_l, part, cnt = _W2[key]
    props = torch.cuda.get_device_properties(dev)
    grid = grid or (3 if WSD2_OCC3 else 2) * props.multi_processor_count
    if WSD2_OCC3:
        stages = 3
    stages = stages or WSD2_STAGES
    pr = 0
    if probe is not None:
        assert WSD2_PROBE, "probe needs the WSD2_PROBE build"
        pt = torch.zeros((int(grid), 16), dtype=torch.int64, device=dev)
        probe.append(pt)
        pr = pt.data_ptr()
    wsd2_ext().wsd2(qkv.data_ptr(), data["cache_loc"].data_ptr(), data["k_cache"].data_ptr(), data["v_cache"].data_ptr(),
                    data["page_table"].data_ptr(), data["seq_lens"].data_ptr(), acc_o.data_ptr(), acc_l.data_ptr(),
                    part.data_ptr(), cnt.data_ptr(), out.data_ptr(), B, Hq, cfg["page_size"], data["page_table"].shape[1],
                    WSD2_SEGK if segk is None else segk, float(cfg["qk_scale_factor"]), float(cfg["sm_scale"]),
                    float(cfg["kv_scale"]), float(cfg["kv_scale"]), int(grid), int(stages), bool(pdl), pr, int(cfg["window"]),
                    _w2_inv_freq(float(cfg["rope_theta"]), dev).data_ptr(), int(round(65536 * (WSD2_SKEW if skew is None else skew))),
                    torch.cuda.current_stream().cuda_stream)
    return out
WSD2_ON = True
_wsd_forward_prod = wsd_forward  # noqa: F821


# wsd_forward: batches of up to 32 requests (head dim 64, 2 kv heads) go to wsd2, the rest to the loader-warp kernel.
def wsd_forward(data, **kw):  # noqa: F811
    cfg = data["config"]
    if WSD2_ON and cfg["batch"] <= 32 and cfg["head_dim"] == 64 and cfg["num_kv_heads"] == 2:
        return wsd2_forward(data, pdl=kw.get("pdl", False), out=kw.get("out"))
    return _wsd_forward_prod(data, **kw)
# wsv2: batched verify (q_len L <= 8) on full-attention layers in the flat-grid geometry of wsd2.
#  * Grid: 2 CTAs per SM over the concatenated keys of all requests (each: seq cached keys + L new tokens), SEGK filler
#    keys before every request after the first. A CTA runs the segments its range touches; the L new tokens of a request
#    belong to the segment holding the request's end (which writes their cache rows and attends them first).
#  * CTA: 4 producer warps (16-B cp.async of 64-key stages: K rows 128-B swizzled = the wgmma K-major SW128 operand,
#    V rows plain) and 2 consumer warpgroups, one per kv head: M = 64 rows (8 tokens x 8 query heads), wgmma e4m3.
#    QK: S = Q K^T (Q = e4m3 of the unit-RMS query from registers, reloaded from smem per stage; K straight off the ring).
#    P = 2^(sc (s - m) + 8) as e4m3 codes round(8 x + 120) via one FFMA against a rounding magic number (no MUFU).
#    PV: O += P V^T with V^T [72 rows x 64 key slots] built per stage by the warpgroup (each warp 16 keys, byte
#    transposes), double-buffered; row 64 is all ones, so the same wgmma yields l = sum of the quantized P.
#  * Merge: rows with log2 offset within +-100 add O 2^m, l 2^m into fp32 accumulators against the absolute reference
#    (8-B vector reds); other rows take an exact (row, m, l, O) slot. The CTA that completes a request's key count
#    normalizes, applies v_scale, the output RMS norm and the sigmoid gate, and returns the accumulators to zero.
#  * Other GPUs (no wgmma): a per-thread emulation of the two wgmma shapes on the same smem / register layouts, for
#    correctness tests.
import zlib

import torch

try:
    lean_load_inline  # noqa: B018  (defined earlier in this file)
except NameError:  # standalone import
    from lean_ext import LEAN_CHECK, lean_load_inline  # noqa: F401

WSV2_SRC = LEAN_CHECK + r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define V2_WG 1  // Hopper wgmma
#endif

namespace okcv2 {
namespace {  // internal linkage (probe and plain builds side by side)
constexpr int HKV = 2, GRP = 8, D = 64, LMAX = 8, ROWS = LMAX * HKV * GRP;  // 128 output rows per request: token x head
constexpr int SK = 64, ROWB = 128, KST = SK * ROWB, STAGE = 2 * KST;      // stage: K (128-B swizzled) | V (plain rows)
constexpr int VTH = 9 * 4 * 128, VTB = 2 * VTH;  // V^T of one head: [72 rows (64 dims, ones, 7 zero)][64 slots] / both
constexpr int NP = 4, NC = 8, NT = 32 * (NP + NC);
constexpr int MAXSLOT = 64, SLOTW = 68;  // exact slots per request: (row, m, l, O[64], pad)
constexpr float MAGIC = 12582912.f;      // 1.5 * 2^23: the low byte of the sum's bits is the rounded integer
#ifdef V2_WG
#ifdef V2_OCC1
constexpr int REG_P = 56, REG_C = 224;   // one CTA per SM: 4 x 56 + 8 x 224 = 12 x 168
#else
constexpr int REG_P = 32, REG_C = 104;   // setmaxnreg: 4 x 32 + 8 x 104 = 12 x 80 (the launch share at 2 CTAs / SM)
#endif
#endif

struct Params {
  const __nv_bfloat16* qkv;   // [B, L, (2 Hq + 2 HKV) * 64]
  const int64_t* cache_loc;   // [B * L]
  uint8_t* k_cache;           // [rows, HKV, 64] e4m3
  uint8_t* v_cache;
  const int32_t* page_table;  // [B, max_pages]
  const int32_t* seq_lens;    // [B] cached keys
  float* acc_o;               // [32][128][64] zero at rest
  float* acc_l;               // [32][128]
  float* part;                // [32][MAXSLOT][SLOTW] exact slots; word 0 = row (int, -1 empty)
  int* cnt;                   // [0, 32) key tickets, [32, 64) slot counts; zero at rest
  __nv_bfloat16* out;         // [B, L, Hq * 64]
  int B, Hq, L, page_size, max_pages, segk;
  int window;                 // 0: full layer; else the sliding window (RoPE on queries / new keys)
  const float* inv_freq;      // [32] RoPE frequencies
  float sc, inv_k_scale, inv_v_scale;  // sc: raw e4m3 score -> log2 units (qk_scale * sm_scale * k_scale * log2 e)
  long long* probe;
  int skew;                   // low-blockIdx CTAs' share of an SM pair's keys, x 2^16 (32768: even)
};
#ifdef V2_LAZY
#define V2_LAZY_DO(...) __VA_ARGS__
#else
#define V2_LAZY_DO(...)
#endif
#ifdef V2_PROBE
enum { V2P_T0, V2P_TPART, V2P_TDATA, V2P_TLOOP, V2P_TEND, V2P_TISSUE, V2P_SMID, V2P_KEYS, V2P_NSEG, V2P_CWAIT, V2P_PWAIT, V2P_NFIN,
       V2P_PH_TR, V2P_PH_BAR, V2P_PH_SOFT, V2P_PH_PV };  // consumer warp 0 phase cycles: QK issue + V^T, WG barrier +
                                                        // QK wait, softmax, PV issue + wait
__device__ __forceinline__ long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define V2P(i, v) (p.probe[static_cast<size_t>(blockIdx.x) * 16 + (i)] = static_cast<long long>(v))
#define V2P_DO(...) __VA_ARGS__
#else
#define V2P(i, v) ((void)0)
#define V2P_DO(...)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  uint32_t done;
  do {
    asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}"
                 : "=r"(done) : "r"(smem_u32(b)), "r"(parity) : "memory");
  } while (!done);
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void cp_async_arrive(uint64_t* b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void gdc_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}
__device__ __forceinline__ void rope_sincos(float x, float* s, float* c) {  // (range reduction in double)
  const double xd = static_cast<double>(x);
  const double k = rint(xd * 0.15915494309189535);
  __sincosf(static_cast<float>(fma(-k, 6.283185307179586, xd)), s, c);
}
__device__ __forceinline__ float2 bf2f(uint32_t u) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u)); }
// K rows: 16-B chunk c of row r at c ^ (r & 7) (128-B swizzle, 1 KB atoms)
__device__ __forceinline__ int k_off(int r, int c) { return r * ROWB + ((c ^ (r & 7)) << 4); }
// V^T element (row n, key slot k): no-swizzle K-major core matrices (8 rows x 16 slots = 128 B): LBO 1152, SBO 128.
// Slot 16 hb + 4 t + e of a 32-key block <-> key 16 hb + 8 (e >> 1) + 2 t + (e & 1) (the S columns lane t holds), so the
// accumulator-to-A-fragment step for P needs no shuffles
__device__ __forceinline__ int vt_off(int n, int k) { return ((k >> 4) * 9 + (n >> 3)) * 128 + (n & 7) * 16 + (k & 15); }
__device__ __forceinline__ void fence_acc32(float (&c)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(c[i]));
}
__device__ __forceinline__ void fence_acc36(float (&c)[36]) {
#pragma unroll
  for (int i = 0; i < 36; ++i) asm volatile("" : "+f"(c[i]));
}

#ifdef V2_WG
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void wg_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory"); }
__device__ __forceinline__ uint64_t desc_sw128(const void* s) {  // K-major, 128-B swizzle: SBO 1024
  return (uint64_t)((smem_u32(s) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint64_t desc_plain(const void* s, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((smem_u32(s) & 0x3FFFF) >> 4) | ((uint64_t)((lbo & 0x3FFFF) >> 4) << 16) | ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32);
}
// S (+)= Q K^T over dims koff .. koff + 31 of the stage's K rows (sd = 0: overwrite)
__device__ __forceinline__ void qk_step(float (&c)[32], const uint32_t (&a)[4], const unsigned char* kst, int koff, int sd) {
  const uint64_t db = desc_sw128(kst + koff);
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
// the same with Q from smem: [64 rows][2 heads x 64 B] e4m3, 128-B swizzled like the K stage (V2_QSS builds)
__device__ __forceinline__ void qk_step_ss(float (&c)[32], const unsigned char* qt, const unsigned char* kst, int koff, int sd) {
  const uint64_t da = desc_sw128(qt + koff), db = desc_sw128(kst + koff);
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
               : "l"(da), "l"(db), "r"(sd));
}
// O += P V over key slots 32 st .. of the V^T tile vth (one head)
__device__ __forceinline__ void pv_step(float (&c)[36], const uint32_t (&a)[4], const unsigned char* vth, int st) {
  const uint64_t db = desc_plain(vth + 2 * 9 * 128 * st, 9 * 128, 128);
  asm volatile("wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, 1, 1, 1;\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31]), "+f"(c[32]), "+f"(c[33]), "+f"(c[34]), "+f"(c[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
#else
// --- emulation of the two wgmma shapes (same operand layouts; synchronous) ---------------------------------------------
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
__device__ __forceinline__ void wg_wait0() {}
__device__ __forceinline__ void wg_wait1() {}
__device__ __forceinline__ float e4f(uint32_t byte) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(byte & 0xFF), __NV_E4M3);
  return __half2float(__half(h));
}
// the quad's A fragments: rows lo (16 w4 + g) / hi (+ 8), k = 4 i .. 4 i + 3 for word i (i < 4: a0 / a1 of lane i,
// i >= 4: a2 / a3 of lane i - 4)
__device__ __forceinline__ void quad_a(const uint32_t (&a)[4], uint32_t (&lo)[8], uint32_t (&hi)[8]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    lo[q] = __shfl_sync(0xffffffffu, a[0], (lane & ~3) | q);
    hi[q] = __shfl_sync(0xffffffffu, a[1], (lane & ~3) | q);
    lo[4 + q] = __shfl_sync(0xffffffffu, a[2], (lane & ~3) | q);
    hi[4 + q] = __shfl_sync(0xffffffffu, a[3], (lane & ~3) | q);
  }
}
__device__ void qk_step(float (&c)[32], const uint32_t (&a)[4], const unsigned char* kst, int koff, int sd) {
  const int t = threadIdx.x & 3;
  uint32_t lo[8], hi[8];
  quad_a(a, lo, hi);
  for (int j = 0; j < 8; ++j)
    for (int e = 0; e < 2; ++e) {
      const int n = 8 * j + 2 * t + e;
      float slo = 0.f, shi = 0.f;
      for (int k = 0; k < 32; ++k) {
        const int kb = koff + k;
        const float bv = e4f(kst[k_off(n, kb >> 4) + (kb & 15)]);
        slo += e4f(lo[k >> 2] >> (8 * (k & 3))) * bv;
        shi += e4f(hi[k >> 2] >> (8 * (k & 3))) * bv;
      }
      c[4 * j + e] = (sd ? c[4 * j + e] : 0.f) + slo;
      c[4 * j + 2 + e] = (sd ? c[4 * j + 2 + e] : 0.f) + shi;
    }
}
__device__ void qk_step_ss(float (&c)[32], const unsigned char* qt, const unsigned char* kst, int koff, int sd) {
  const int lane = threadIdx.x & 31, t = lane & 3, mlo = 16 * (((threadIdx.x >> 5) - NP) & 3) + (lane >> 2);
  for (int j = 0; j < 8; ++j)
    for (int e = 0; e < 2; ++e) {
      const int n = 8 * j + 2 * t + e;
      float slo = 0.f, shi = 0.f;
      for (int k = 0; k < 32; ++k) {
        const int kb = koff + k;
        const float bv = e4f(kst[k_off(n, kb >> 4) + (kb & 15)]);
        slo += e4f(qt[k_off(mlo, kb >> 4) + (kb & 15)]) * bv;
        shi += e4f(qt[k_off(mlo + 8, kb >> 4) + (kb & 15)]) * bv;
      }
      c[4 * j + e] = (sd ? c[4 * j + e] : 0.f) + slo;
      c[4 * j + 2 + e] = (sd ? c[4 * j + 2 + e] : 0.f) + shi;
    }
}
__device__ void pv_step(float (&c)[36], const uint32_t (&a)[4], const unsigned char* vth, int st) {
  const int t = threadIdx.x & 3;
  uint32_t lo[8], hi[8];
  quad_a(a, lo, hi);
  for (int j = 0; j < 9; ++j)
    for (int e = 0; e < 2; ++e) {
      const int n = 8 * j + 2 * t + e;
      float slo = 0.f, shi = 0.f;
      for (int k = 0; k < 32; ++k) {
        const float bv = e4f(vth[vt_off(n, 32 * st + k)]);
        slo += e4f(lo[k >> 2] >> (8 * (k & 3))) * bv;
        shi += e4f(hi[k >> 2] >> (8 * (k & 3))) * bv;
      }
      c[4 * j + e] += slo;
      c[4 * j + 2 + e] += shi;
    }
}
#endif

template <int ST>
#ifdef V2_WG
#ifdef V2_OCC1
__global__ void __launch_bounds__(NT, 1) wsv2_kernel(const Params p) {
#else
__global__ void __launch_bounds__(NT, 2) wsv2_kernel(const Params p) {
#endif
#else
__global__ void __launch_bounds__(NT, 1) wsv2_kernel(const Params p) {
#endif
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* ring = smem_raw;                  // [ST][K 8 KB swizzled | V 8 KB plain]
  unsigned char* newk = ring + ST * STAGE;         // the new tokens' K rows (swizzled; rows >= L zero)
  unsigned char* vtr = newk + KST;                 // V^T of the cached stages: [2 buffers][2 heads][VTH]
  unsigned char* newvt = vtr + 2 * VTB;            // V^T of the new tokens: [2 heads][VTH]
  uint4* qsm = reinterpret_cast<uint4*>(newvt + VTB);  // query A fragments [2 heads][2 k steps][128 threads]
  unsigned char* qt = newvt + VTB;  // (V2_QSS: the same 8 KB as the query tile [64 M rows][2 heads x 64 B], swizzled)
  uint64_t* full = reinterpret_cast<uint64_t*>(qsm + 2 * 2 * 128);
  uint64_t* empty = full + ST;
  uint64_t* done = empty + ST;  // always complete (phase 0): the new stage's wait
  __shared__ int s_seg_b[33], s_seg_k0[33], s_seg_k1[33], s_seg_seq[33], s_seg_lo[33], s_nseg, s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int L = p.L;
  V2P_DO(if (tid == 0) V2P(V2P_T0, gtimer());)
  // --- partition (warp 0; lane b = request b): keys of request b = seq + L ------------------------------------------
  if (warp == 0) {
    const int B = p.B;
    const int sq = lane < B ? p.seq_lens[lane] : 0;
    const int lo_b = p.window ? max(0, sq + 1 - p.window) : 0;  // the request's first key (token 0's window start)
    const long long n = lane < B ? static_cast<long long>(sq) + L - lo_b : 0;
    const long long v = lane < B ? n + (lane > 0 ? p.segk : 0) : 0;
    long long inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const long long y = __shfl_up_sync(0xffffffffu, inc, o);
      if (lane >= o) inc += y;
    }
    const long long tv = __shfl_sync(0xffffffffu, inc, 31);
    const long long G = gridDim.x;
    // CTA c's range [F(c) tv, F(c + 1) tv): F rises by skew per CTA over the first half of the grid (dispatched first:
    // one per SM, the faster SM-mate) and by 1 - skew over the second (x 2 / G; skew 1/2: even)
    const long long H = G >> 1, f = p.skew;
    auto F = [&](long long c) -> long long {  // (unnormalized)
      return c <= H ? c * f : H * f + (c - H) * (65536 - f);
    };
    const long long FG = F(G);
    const long long lo = tv * F(blockIdx.x) / FG, hi = tv * F(blockIdx.x + 1) / FG;
    const long long st = inc - n, en = inc;  // this request's real keys [st, en)
    const bool meets = lane < B && max(lo, st) < min(hi, en);
    const unsigned m = __ballot_sync(0xffffffffu, meets);
    if (meets) {
      const int idx = __popc(m & ((1u << lane) - 1u));
      if (idx == 0 && p.page_size == 1) {  // the first segment's first two stages of page ids -> L2 (4 lines)
        const int32_t* pf = p.page_table + static_cast<size_t>(lane) * p.max_pages + lo_b + (max(lo, st) - st);
#pragma unroll
        for (int q = 0; q < 4; ++q) asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + 32 * q));
      }
      s_seg_b[idx] = lane;
      s_seg_k0[idx] = static_cast<int>(max(lo, st) - st);
      s_seg_k1[idx] = static_cast<int>(min(hi, en) - st);
      s_seg_seq[idx] = static_cast<int>(n - L);  // (cached keys from the window start)
      s_seg_lo[idx] = lo_b;
    }
    if (lane == 0) s_nseg = __popc(m);
  }
  if (tid == 32) {
    for (int i = 0; i < ST; ++i) { mbar_init(full + i, 32 * NP); mbar_init(empty + i, NC); }
    mbar_init(done, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_arrive(done);
  }
  __syncthreads();
  const int nseg = s_nseg;
  V2P_DO(if (tid == 0) { V2P(V2P_TPART, gtimer()); V2P(V2P_NSEG, nseg); unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm)); V2P(V2P_SMID, sm); })
  if (warp < NP) {
    // ===================== producers: thread q -> 16-B chunk q & 7 of rows (q >> 3) + 16 k of every stage =============
#ifdef V2_WG
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(REG_P));
#endif
    const int q = tid, ch = q & 7, r0 = q >> 3;
    const int kswz = (ch ^ (r0 & 7)) << 4;  // (rows r0 + 16 k share r0's swizzle)
    int gst = 0;
    V2P_DO(long long pwait = 0;)
    for (int si = 0; si < nseg; ++si) {
      const int b = s_seg_b[si], k0 = s_seg_k0[si], seq = s_seg_seq[si];
      const int cend = min(s_seg_k1[si], seq);  // cached keys of this segment: [k0, cend)
      const int nst = cend > k0 ? (cend - k0 + SK - 1) / SK : 0;
      const int32_t* pt = p.page_table + static_cast<size_t>(b) * p.max_pages;
      const int lo_b = s_seg_lo[si];
      auto row_of = [&](int pos) -> uint32_t {
        if (pos >= cend) return 0u;
        const int ap = lo_b + pos;
        return p.page_size == 1 ? static_cast<uint32_t>(__ldg(pt + ap))
                                : static_cast<uint32_t>(__ldg(pt + ap / p.page_size)) * p.page_size + ap % p.page_size;
      };
      uint32_t id0[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) id0[k] = row_of(k0 + r0 + 16 * k);
      for (int i = 0; i < nst; ++i, ++gst) {
        uint32_t id1[4];  // the next stage's rows, loaded under this stage's wait and copies
#pragma unroll
        for (int k = 0; k < 4; ++k) id1[k] = row_of(k0 + SK * (i + 1) + r0 + 16 * k);
        const int sl = gst % ST;
        V2P_DO(const long long w0 = clock64();)
        if (gst >= ST) mbar_wait(empty + sl, ((gst / ST) - 1) & 1);
        V2P_DO(pwait += clock64() - w0;)
        unsigned char* kb = ring + sl * STAGE;
        unsigned char* vb = kb + KST;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int r = r0 + 16 * k;
          const int bytes = k0 + SK * i + r < cend ? 16 : 0;
          const size_t off = static_cast<size_t>(id0[k]) * ROWB + ch * 16;
          cp_async16(smem_u32(kb + r * ROWB + kswz), p.k_cache + off, bytes);
          cp_async16(smem_u32(vb + r * ROWB + ch * 16), p.v_cache + off, bytes);
        }
        cp_async_arrive(full + sl);
#pragma unroll
        for (int k = 0; k < 4; ++k) id0[k] = id1[k];
      }
    }
    V2P_DO(if (q == 0) { V2P(V2P_TISSUE, gtimer()); V2P(V2P_PWAIT, pwait); })
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }
  // ========================= consumers: warpgroup h = kv head h; warp w4 -> rows 16 w4 + g (lo) / + 8 (hi) ===========
#ifdef V2_WG
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(REG_C));
#endif
  const int ctid = tid - 32 * NP, h = ctid >> 7, wt = ctid & 127, w4 = wt >> 5, g = lane >> 2, t = lane & 3;
  const int tok[2] = {2 * w4, 2 * w4 + 1};  // the tokens of rows lo / hi (query head 8 h + g)
  const int qkv_stride = (2 * p.Hq + 2 * HKV) * D;
  // V^T tiles: zero, row 64 all ones (e4m3 1.0 = 0x38); the new tokens' K rows zero
  for (int i = ctid; i < 3 * VTB / 16; i += 32 * NC) {
    const int o = 16 * i, blk = (o % VTH) >> 7;
    const uint32_t v = (blk % 9 == 8 && (o & 127) < 16) ? 0x38383838u : 0u;
    reinterpret_cast<uint4*>(vtr)[i] = make_uint4(v, v, v, v);
  }
  for (int i = ctid; i < KST / 16; i += 32 * NC) reinterpret_cast<uint4*>(newk)[i] = make_uint4(0u, 0u, 0u, 0u);
  gdc_wait();  // queries / the new tokens come from the previous kernel
  int gst = 0, vbuf = 0;
  V2P_DO(long long cwait = 0, keys = 0; int nfin = 0; bool first = true; long long ph[4] = {0, 0, 0, 0}; long long pt0 = 0;)
  for (int si = 0; si < nseg; ++si) {
    const int b = s_seg_b[si], k0 = s_seg_k0[si], k1 = s_seg_k1[si], seq = s_seg_seq[si];
    const int cend = min(k1, seq);
    const int nst = cend > k0 ? (cend - k0 + SK - 1) / SK : 0;
    const bool has_new = k1 == seq + L;  // this segment ends the request: it owns the L new tokens
    const int seq_abs = s_seg_lo[si] + seq;  // (seq: cached keys from the window start)
    // sliding: token j sees window keys kr >= j + woff (woff = 0 once the window binds); full layers: none masked
    const int woff = p.window ? seq_abs + 1 - p.window - s_seg_lo[si] : -LMAX;
    V2P_DO(keys += max(0, cend - k0);)
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");  // the previous segment's readers of qsm / new tiles are done
    const __nv_bfloat16* qkvb = p.qkv + static_cast<size_t>(b) * L * qkv_stride;
    // --- query A fragments: e4m3 of the unit-RMS query (rows lo / hi: dims 4 t + 16 c + e of this quad) ---------------
    {
      float v[2][16];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const bool qok = tok[r] < L;  // (missing tokens: zero queries, never written out)
        const __nv_bfloat16* qr = qkvb + (qok ? tok[r] : 0) * qkv_stride + (8 * h + g) * D + 4 * t;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint2 u = *reinterpret_cast<const uint2*>(qr + 16 * c);
          const float2 a = bf2f(u.x), bb = bf2f(u.y);
          v[r][4 * c] = qok ? a.x : 0.f; v[r][4 * c + 1] = qok ? a.y : 0.f;
          v[r][4 * c + 2] = qok ? bb.x : 0.f; v[r][4 * c + 3] = qok ? bb.y : 0.f;
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += v[r][i] * v[r][i];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int i = 0; i < 16; ++i) v[r][i] *= inv;
        if (p.window) {  // RoPE at the token's position: pairs (dims 16 c + 4 t + 2 e, + 1) = pair 8 c + 2 t + e
          const float pos = static_cast<float>(seq_abs + tok[r]);
#pragma unroll
          for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              float sn, cs;
              rope_sincos(pos * __ldg(p.inv_freq + 8 * c + 2 * t + e), &sn, &cs);
              const float a = v[r][4 * c + 2 * e], o = v[r][4 * c + 2 * e + 1];
              v[r][4 * c + 2 * e] = a * cs - o * sn;
              v[r][4 * c + 2 * e + 1] = a * sn + o * cs;
            }
        }
      }
#ifdef V2_QSS
      // M rows 16 w4 + g (lo) / + 8 (hi): dims 16 c + 4 t .. -> chunk 4 h + c, bytes 4 t ..
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c)
          *reinterpret_cast<uint32_t*>(qt + k_off(16 * w4 + g + 8 * r, 4 * h + c) + 4 * t) =
              fp8x4(v[r][4 * c], v[r][4 * c + 1], v[r][4 * c + 2], v[r][4 * c + 3]);
      fence_proxy_async();
#else
#pragma unroll
      for (int st = 0; st < 2; ++st)
        qsm[(2 * h + st) * 128 + wt] = make_uint4(fp8x4(v[0][8 * st], v[0][8 * st + 1], v[0][8 * st + 2], v[0][8 * st + 3]),
                                                  fp8x4(v[1][8 * st], v[1][8 * st + 1], v[1][8 * st + 2], v[1][8 * st + 3]),
                                                  fp8x4(v[0][8 * st + 4], v[0][8 * st + 5], v[0][8 * st + 6], v[0][8 * st + 7]),
                                                  fp8x4(v[1][8 * st + 4], v[1][8 * st + 5], v[1][8 * st + 6], v[1][8 * st + 7]));
#endif
    }
    // --- the new tokens (64 threads: token ni, kv head nh, dims 16 q4 ..): cache rows, K rows and V^T slots -------------
    if (has_new && ctid < 64) {
      const int ni = ctid >> 3, nh = (ctid >> 2) & 1, q4 = ctid & 3;
      const bool nok = ni < L;
      const __nv_bfloat16* kr = qkvb + (nok ? ni : 0) * qkv_stride + p.Hq * D + nh * D + 16 * q4;
      const uint4 kin[2] = {*reinterpret_cast<const uint4*>(kr), *reinterpret_cast<const uint4*>(kr + 8)};
      const uint4 vin[2] = {*reinterpret_cast<const uint4*>(kr + HKV * D), *reinterpret_cast<const uint4*>(kr + HKV * D + 8)};
      float kf[16], vf[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t kw = (i < 4) ? (&kin[0].x)[i] : (&kin[1].x)[i - 4], vw = (i < 4) ? (&vin[0].x)[i] : (&vin[1].x)[i - 4];
        const float2 a2 = bf2f(kw), b2 = bf2f(vw);
        kf[2 * i] = a2.x; kf[2 * i + 1] = a2.y; vf[2 * i] = b2.x; vf[2 * i + 1] = b2.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) ss += kf[i] * kf[i];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f), iv = p.inv_v_scale;
#pragma unroll
      for (int i = 0; i < 16; ++i) kf[i] *= inv;
      if (p.window) {  // RoPE at the new token's position: pairs 8 q4 + k
        const float pos = static_cast<float>(seq_abs + ni);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          float sn, cs;
          rope_sincos(pos * __ldg(p.inv_freq + 8 * q4 + k), &sn, &cs);
          const float a = kf[2 * k], o = kf[2 * k + 1];
          kf[2 * k] = a * cs - o * sn;
          kf[2 * k + 1] = a * sn + o * cs;
        }
      }
      const float ik = p.inv_k_scale;
      uint32_t kq[4], vq[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        kq[i] = fp8x4(kf[4 * i] * ik, kf[4 * i + 1] * ik, kf[4 * i + 2] * ik, kf[4 * i + 3] * ik);
        vq[i] = fp8x4(vf[4 * i] * iv, vf[4 * i + 1] * iv, vf[4 * i + 2] * iv, vf[4 * i + 3] * iv);
      }
      if (nok) {
        const uint4 k16 = make_uint4(kq[0], kq[1], kq[2], kq[3]), v16 = make_uint4(vq[0], vq[1], vq[2], vq[3]);
        const int64_t loc = p.cache_loc[static_cast<size_t>(b) * L + ni];
        *reinterpret_cast<uint4*>(p.k_cache + static_cast<size_t>(loc) * ROWB + nh * D + 16 * q4) = k16;
        *reinterpret_cast<uint4*>(p.v_cache + static_cast<size_t>(loc) * ROWB + nh * D + 16 * q4) = v16;
        *reinterpret_cast<uint4*>(newk + k_off(ni, 4 * nh + q4)) = k16;
        unsigned char* vt = newvt + nh * VTH;
        const int sl = 4 * (ni >> 1) + (ni & 1);
#pragma unroll
        for (int e = 0; e < 16; ++e) vt[vt_off(16 * q4 + e, sl)] = static_cast<unsigned char>(vq[e >> 2] >> (8 * (e & 3)));
      }
      fence_proxy_async();  // generic stores -> wgmma operand reads
    }
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
    float O[36];
#pragma unroll
    for (int i = 0; i < 36; ++i) O[i] = 0.f;
    float M[2] = {-INFINITY, -INFINITY};  // running max of the raw scores, rows lo / hi
    const float sc = p.sc, sc8 = 8.f * sc;
    V2_LAZY_DO(const float lazy_thr = 0.75f / sc;)
    // one 64-key stage per iteration (i = -1: the new tokens, first): QK, V^T (cached stages), softmax with masks,
    // O += P V. ONE wgmma code site and branch-free waits (a second site or a conditional spin makes ptxas serialize
    // every wgmma of the kernel, C7511 / C7520): the new stage waits on an always-complete barrier
    for (int i = has_new ? -1 : 0; i < nst; ++i) {
      const bool cached = i >= 0;
      const int sl = gst % ST;
      const unsigned char* kst = cached ? ring + sl * STAGE : newk;
      unsigned char* vth = cached ? vtr + vbuf * VTB + h * VTH : newvt + h * VTH;
      const int nvalid = cached ? min(SK, cend - (k0 + SK * i)) : L;
      V2P_DO(const long long w0 = clock64();)
      mbar_wait(cached ? full + sl : done, cached ? (gst / ST) & 1 : 0);
      V2P_DO(pt0 = clock64();)
      V2P_DO(if (cached) { cwait += clock64() - w0; if (first && ctid == 0) { V2P(V2P_TDATA, gtimer()); first = false; } })
      __syncwarp();
      float S[32];
#ifdef V2_QSS
      wg_fence();
      qk_step_ss(S, qt, kst, 64 * h, 0);
      qk_step_ss(S, qt, kst, 64 * h + 32, 1);
      wg_commit();
#else
      {
        uint32_t qa[2][4];
#pragma unroll
        for (int st = 0; st < 2; ++st) {
          const uint4 u = qsm[(2 * h + st) * 128 + wt];
          qa[st][0] = u.x; qa[st][1] = u.y; qa[st][2] = u.z; qa[st][3] = u.w;
        }
        wg_fence();
        qk_step(S, qa[0], kst, 64 * h, 0);
        qk_step(S, qa[1], kst, 64 * h + 32, 1);
        wg_commit();
      }
#endif
      if (cached) {
        // V^T of key group w4 (keys 16 w4 ..) for head h: lane -> dims 4 (lane & 15) .., keys (2 t, 2 t + 1, 2 t + 8,
        // 2 t + 9) for t = 2 (lane >> 4) + tt; store i holds dim (i + rot) & 3 (spreads the stores over the banks)
        const int d4 = lane & 15, tp = lane >> 4, rot = (d4 >> 1) & 3;
        const unsigned char* src = kst + KST + 16 * w4 * ROWB + 64 * h + 4 * d4;
        uint32_t o[4][2];
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int tq = 2 * tp + tt;
          const int kk[4] = {2 * tq, 2 * tq + 1, 2 * tq + 8, 2 * tq + 9};
          uint32_t w[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint32_t x = *reinterpret_cast<const uint32_t*>(src + kk[e] * ROWB);
            w[e] = __funnelshift_r(x, x, 8 * rot);
          }
          const uint32_t a = __byte_perm(w[0], w[1], 0x5140), bq = __byte_perm(w[0], w[1], 0x7362);
          const uint32_t c = __byte_perm(w[2], w[3], 0x5140), d = __byte_perm(w[2], w[3], 0x7362);
          o[0][tt] = __byte_perm(a, c, 0x5410);
          o[1][tt] = __byte_perm(a, c, 0x7632);
          o[2][tt] = __byte_perm(bq, d, 0x5410);
          o[3][tt] = __byte_perm(bq, d, 0x7632);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q)
          *reinterpret_cast<uint2*>(vth + vt_off(4 * d4 + ((q + rot) & 3), 16 * w4) + 8 * tp) = make_uint2(o[q][0], o[q][1]);
#ifndef V2_LATESYNC
        fence_proxy_async();
#endif
      }
      V2P_DO({ const long long c = clock64(); ph[0] += c - pt0; pt0 = c; })
#ifndef V2_LATESYNC
      if (h == 0) asm volatile("bar.sync 2, 128;" ::: "memory");  // the head's V^T tile is complete (constant
      else asm volatile("bar.sync 3, 128;" ::: "memory");         //  barrier ids: a register id reserves all 16)
#endif
      wg_wait0();  // QK done
      fence_acc32(S);
      __syncwarp();
      if (cached && lane == 0) mbar_arrive(empty + sl);  // this warp's K and V reads of the stage are done
      V2P_DO({ const long long c = clock64(); ph[1] += c - pt0; pt0 = c; })
      // softmax: masks (cached: keys < nvalid; new: keys < L and key <= the row's token), row max, P codes
      const int kr0 = k0 + SK * i;  // the stage's first key (from the window start)
      if (!cached || nvalid < SK || kr0 < LMAX - 1 + woff) {  // (edge stages only)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int kk = 8 * j + 2 * t + (c & 1);
            if (!(kk < nvalid && (cached ? kr0 + kk >= tok[c >> 1] + woff : kk <= tok[c >> 1]))) S[4 * j + c] = -INFINITY;
          }
      }
      float tm[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) tm[c >> 1] = fmaxf(tm[c >> 1], S[4 * j + c]);
      float mref[2], scl[2] = {1.f, 1.f};
#ifdef V2_LAZY
      // lazy offset: M moves only when a score exceeds it by more than the e4m3 headroom of P (code <= 126 = 448:
      // sc (s - M) <= 0.75); any such row in the warp sends the warp through the exact max / rescale
      const bool move = __any_sync(0xffffffffu, tm[0] > M[0] + lazy_thr || tm[1] > M[1] + lazy_thr);
#else
      const bool move = true;
#endif
      if (move) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          tm[r] = fmaxf(tm[r], __shfl_xor_sync(0xffffffffu, tm[r], 1));
          tm[r] = fmaxf(tm[r], __shfl_xor_sync(0xffffffffu, tm[r], 2));
          const float mn = fmaxf(M[r], tm[r]);  // (finite: every stage has a valid key for every row)
          scl[r] = exp2f(fmaxf(M[r] - mn, -1e30f) * sc);  // (M = -inf on the first stage: 0)
          M[r] = mn;
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) mref[r] = MAGIC + 120.f - sc8 * M[r];
      auto pack = [&](int bk, uint32_t (&pq)[4]) {  // P codes of keys 32 bk .. 32 bk + 31 -> the e4m3 A fragment
        uint32_t e[4][4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
#pragma unroll
          for (int c = 0; c < 4; ++c)
            e[jj][c] = __float_as_uint(fmaxf(fmaf(S[4 * (4 * bk + jj) + c], sc8, mref[c >> 1]), MAGIC));
        pq[0] = __byte_perm(__byte_perm(e[0][0], e[0][1], 0x0040), __byte_perm(e[1][0], e[1][1], 0x0040), 0x5410);
        pq[1] = __byte_perm(__byte_perm(e[0][2], e[0][3], 0x0040), __byte_perm(e[1][2], e[1][3], 0x0040), 0x5410);
        pq[2] = __byte_perm(__byte_perm(e[2][0], e[2][1], 0x0040), __byte_perm(e[3][0], e[3][1], 0x0040), 0x5410);
        pq[3] = __byte_perm(__byte_perm(e[2][2], e[2][3], 0x0040), __byte_perm(e[3][2], e[3][3], 0x0040), 0x5410);
      };
      if (move) {
#pragma unroll
        for (int j = 0; j < 9; ++j) { O[4 * j] *= scl[0]; O[4 * j + 1] *= scl[0]; O[4 * j + 2] *= scl[1]; O[4 * j + 3] *= scl[1]; }
      }
      uint32_t pa0[4], pa1[4];
#ifdef V2_LATESYNC
      // the head's V^T tile is complete: the warpgroup meets only here, after every warp's softmax (its stage reads are
      // done, so the empty arrive above did not need the barrier); the proxy fence of its V^T stores comes late too
      if (cached) fence_proxy_async();
      if (h == 0) asm volatile("bar.sync 2, 128;" ::: "memory");
      else asm volatile("bar.sync 3, 128;" ::: "memory");
#endif
#ifdef V2_PVSPLIT
      // the first half's PV runs under the second half's packing
      pack(0, pa0);
      V2P_DO({ const long long c = clock64(); ph[2] += c - pt0; pt0 = c; })
      wg_fence();
      pv_step(O, pa0, vth, 0);
      pack(1, pa1);
      wg_fence();
      pv_step(O, pa1, vth, 1);
#else
      pack(0, pa0);
      pack(1, pa1);
      V2P_DO({ const long long c = clock64(); ph[2] += c - pt0; pt0 = c; })
      wg_fence();
      pv_step(O, pa0, vth, 0);
      pv_step(O, pa1, vth, 1);
#endif
      wg_commit();
      // retire the PV here: a PV left in flight across the loop edge under the next QK also makes ptxas serialize
      wg_wait0();
      fence_acc36(O);
      V2P_DO({ const long long c = clock64(); ph[3] += c - pt0; })
      if (cached) { ++gst; vbuf ^= 1; }
    }
    V2P_DO(if (si == nseg - 1 && ctid == 0) V2P(V2P_TLOOP, gtimer());)
    if (si == nseg - 1) gdc_launch();
    // --- merge: rows against the absolute reference (true O = O 2^(sc M - 8)); l = column 64 (lane t = 0) -------------
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float l = __shfl_sync(0xffffffffu, O[32 + 2 * r], lane & ~3);
      if (tok[r] >= L || M[r] == -INFINITY) continue;  // (warp-uniform / quad-uniform)
      const int row = tok[r] * 16 + 8 * h + g;
      const float m2 = sc * M[r] - 8.f;
      if (fabsf(m2) <= 100.f) {
        const float f = exp2f(m2);
        float* ao = p.acc_o + (static_cast<size_t>(b) * ROWS + row) * D + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; ++j)
          asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(ao + 8 * j), "f"(O[4 * j + 2 * r] * f),
                       "f"(O[4 * j + 2 * r + 1] * f) : "memory");
        if (t == 0) atomicAdd(p.acc_l + b * ROWS + row, l * f);
      } else {  // an exact slot for the last CTA to fold (logits beyond 2^100)
        int slot = 0;
        if (t == 0) slot = atomicAdd(p.cnt + 32 + b, 1);
        slot = min(__shfl_sync(__activemask(), slot, lane & ~3), MAXSLOT - 1);
        float* pr = p.part + (static_cast<size_t>(b) * MAXSLOT + slot) * SLOTW;
        if (t == 0) { pr[1] = m2; pr[2] = l; }
#pragma unroll
        for (int j = 0; j < 8; ++j) { pr[3 + 8 * j + 2 * t] = O[4 * j + 2 * r]; pr[3 + 8 * j + 2 * t + 1] = O[4 * j + 2 * r + 1]; }
        __threadfence();
        __syncwarp(__activemask());
        if (t == 0) reinterpret_cast<int*>(pr)[0] = row;
      }
    }
    __threadfence();
    // --- ticket: this segment's keys; the CTA that completes the request finalizes it ------------------------------------
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
    if (ctid == 0) {
      const int mine = k1 - k0;
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p.cnt + b), "r"(mine) : "memory");
      s_last = (old + mine == seq + L);
    }
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");
    if (s_last) {
      V2P_DO(++nfin;)
      __threadfence();
      const int nslot = min(__ldcg(p.cnt + 32 + b), MAXSLOT);
      // L * 16 rows x 8 parts of 8 dims over 4 passes; every pass's loads first (one L2 round trip, not four)
      constexpr int NPS = LMAX * 16 * 8 / (32 * NC);
      float4 A0[NPS], A1[NPS];
      float LS[NPS];
      uint4 GU[NPS];
#pragma unroll
      for (int ps = 0; ps < NPS; ++ps) {
        const int it = ps * 32 * NC + ctid, row = min(it >> 3, L * 16 - 1), part = it & 7;
        const float* ao = p.acc_o + (static_cast<size_t>(b) * ROWS + row) * D + 8 * part;
        A0[ps] = __ldcg(reinterpret_cast<const float4*>(ao));
        A1[ps] = __ldcg(reinterpret_cast<const float4*>(ao + 4));
        LS[ps] = __ldcg(p.acc_l + b * ROWS + row);
        GU[ps] = *reinterpret_cast<const uint4*>(qkvb + (row >> 4) * qkv_stride + (p.Hq + 2 * HKV) * D + (row & 15) * D + 8 * part);
      }
#pragma unroll
      for (int ps = 0; ps < NPS; ++ps) {
        const int it = ps * 32 * NC + ctid, row = min(it >> 3, L * 16 - 1), part = it & 7;
        const bool act = it < L * 16 * 8;
        float* ao = p.acc_o + (static_cast<size_t>(b) * ROWS + row) * D + 8 * part;
        float Lsum = LS[ps];
        float ac[8] = {A0[ps].x, A0[ps].y, A0[ps].z, A0[ps].w, A1[ps].x, A1[ps].y, A1[ps].z, A1[ps].w};
        float Mx = Lsum > 0.f ? 0.f : -INFINITY;
        for (int s2 = 0; s2 < nslot; ++s2) {  // (exact slots: rare)
          const float* pr = p.part + (static_cast<size_t>(b) * MAXSLOT + s2) * SLOTW;
          if (__ldcg(reinterpret_cast<const int*>(pr)) != row) continue;
          const float m2 = __ldcg(pr + 1), l2 = __ldcg(pr + 2);
          const float mn = fmaxf(Mx, m2);
          const float ca = (Mx == -INFINITY) ? 0.f : exp2f(Mx - mn), cb = exp2f(m2 - mn);
          Mx = mn;
          Lsum = Lsum * ca + l2 * cb;
#pragma unroll
          for (int d = 0; d < 8; ++d) ac[d] = ac[d] * ca + __ldcg(pr + 3 + 8 * part + d) * cb;
        }
        const float inv = (Lsum > 0.f ? 1.f / Lsum : 0.f) / p.inv_v_scale;
        float ss = 0.f;
#pragma unroll
        for (int d = 0; d < 8; ++d) { ac[d] *= inv; ss += ac[d] * ac[d]; }
#pragma unroll
        for (int o2 = 1; o2 < 8; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
        const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
        const int tk = row >> 4, head = row & 15;
        if (act) {
          const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&GU[ps]);
          uint4 ov;
          __nv_bfloat162* o2v = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 gf = __bfloat1622float2(g2[k]);
            o2v[k] = __floats2bfloat162_rn(ac[2 * k] * rn / (1.f + __expf(-gf.x)), ac[2 * k + 1] * rn / (1.f + __expf(-gf.y)));
          }
          *reinterpret_cast<uint4*>(p.out + (static_cast<size_t>(b) * L + tk) * (p.Hq * D) + head * D + 8 * part) = ov;
          *reinterpret_cast<float4*>(ao) = make_float4(0.f, 0.f, 0.f, 0.f);  // back to zero for the next call
          *reinterpret_cast<float4*>(ao + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          if (part == 0) p.acc_l[b * ROWS + row] = 0.f;
        }
      }
      asm volatile("bar.sync 1, %0;" ::"n"(32 * NC) : "memory");  // (slot reads done)
      if (ctid < nslot) reinterpret_cast<int*>(p.part + (static_cast<size_t>(b) * MAXSLOT + ctid) * SLOTW)[0] = -1;
      if (ctid == 0) { p.cnt[b] = 0; p.cnt[32 + b] = 0; }
    }
  }
  V2P_DO(if (ctid == 0) { V2P(V2P_TEND, gtimer()); V2P(V2P_KEYS, keys); V2P(V2P_CWAIT, cwait); V2P(V2P_NFIN, nfin);
                          for (int q = 0; q < 4; ++q) V2P(V2P_PH_TR + q, ph[q]); })
}

template <int ST>
const char* launch(const Params& prm, int grid, bool pdl, cudaStream_t stream) {
  const int bytes = ST * STAGE + KST + 3 * VTB + 2 * 2 * 128 * 16 + (2 * ST + 1) * 8;
  cudaError_t err = cudaFuncSetAttribute(wsv2_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (err != cudaSuccess) { (void)cudaGetLastError(); return cudaGetErrorString(err); }
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  lc.attrs = at; lc.numAttrs = 1; lc.stream = stream;
  lc.gridDim = dim3(grid); lc.blockDim = dim3(NT); lc.dynamicSmemBytes = bytes;
  err = cudaLaunchKernelEx(&lc, wsv2_kernel<ST>, prm);
  if (err != cudaSuccess) { (void)cudaGetLastError(); return cudaGetErrorString(err); }
  return nullptr;
}
}  // namespace
}  // namespace okcv2

void wsv2(int64_t qkv, int64_t cache_loc, int64_t k_cache, int64_t v_cache, int64_t page_table, int64_t seq_lens,
          int64_t acc_o, int64_t acc_l, int64_t part, int64_t cnt, int64_t out, int64_t B, int64_t Hq, int64_t L,
          int64_t page_size, int64_t max_pages, int64_t segk, double sc, double k_scale, double v_scale, int64_t grid,
          int64_t stages, bool pdl, int64_t probe, int64_t window, int64_t inv_freq, int64_t skew, int64_t stream) {
  using namespace okcv2;
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);
  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);
  p.page_table = reinterpret_cast<const int32_t*>(page_table);
  p.seq_lens = reinterpret_cast<const int32_t*>(seq_lens);
  p.acc_o = reinterpret_cast<float*>(acc_o);
  p.acc_l = reinterpret_cast<float*>(acc_l);
  p.part = reinterpret_cast<float*>(part);
  p.cnt = reinterpret_cast<int*>(cnt);
  p.out = reinterpret_cast<__nv_bfloat16*>(out);
  p.B = static_cast<int>(B); p.Hq = static_cast<int>(Hq); p.L = static_cast<int>(L);
  p.page_size = static_cast<int>(page_size); p.max_pages = static_cast<int>(max_pages); p.segk = static_cast<int>(segk);
  p.sc = static_cast<float>(sc);
  p.inv_k_scale = static_cast<float>(1.0 / k_scale);
  p.inv_v_scale = static_cast<float>(1.0 / v_scale);
  p.probe = reinterpret_cast<long long*>(probe);
  p.skew = static_cast<int>(skew);
  p.window = static_cast<int>(window);
  p.inv_freq = reinterpret_cast<const float*>(inv_freq);
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  TORCH_CHECK(stages >= 2 && stages <= 8, "wsv2: stages must be 2 .. 8");
  TORCH_CHECK(L >= 1 && L <= LMAX && B <= 32, "wsv2: L in 1..8, B <= 32");
  const int g = static_cast<int>(grid);
  const char* e = stages == 4 ? launch<4>(p, g, pdl, st) : stages == 3 ? launch<3>(p, g, pdl, st)
                  : stages == 6 ? launch<6>(p, g, pdl, st) : stages == 8 ? launch<8>(p, g, pdl, st) : launch<2>(p, g, pdl, st);
  TORCH_CHECK(e == nullptr, "wsv2 launch failed: ", e ? e : "");
}
"""

WSV2_CPP = ("void wsv2(int64_t qkv, int64_t cache_loc, int64_t k_cache, int64_t v_cache, int64_t page_table, "
            "int64_t seq_lens, int64_t acc_o, int64_t acc_l, int64_t part, int64_t cnt, int64_t out, int64_t B, int64_t Hq, "
            "int64_t L, int64_t page_size, int64_t max_pages, int64_t segk, double sc, double k_scale, double v_scale, "
            "int64_t grid, int64_t stages, bool pdl, int64_t probe, int64_t window, int64_t inv_freq, int64_t skew, "
            "int64_t stream);")
_V2 = {}
WSV2_STAGES = 4
WSV2_SEGK = 600
WSV2_PROBE = False  # probe build: 16 int64 per CTA (wsv2_forward(..., probe=[]) appends the [grid, 16] tensor)
WSV2_QSS = False  # QK with the query tile from smem (wgmma SS form): no A-fragment registers
WSV2_SKEW = 0.5  # low-blockIdx CTAs' share of an SM pair's keys (0.5: even)
WSV2_LATESYNC = False  # warpgroup V^T barrier just before the PV (after the softmax)
WSV2_PVSPLIT = False  # issue the first half's PV before packing the second half's P codes
WSV2_OCC1 = False  # one CTA per SM (launch bound 1, setmaxnreg 56 / 224; grid = SMs, deeper rings)
WSV2_LAZY = False  # lazy softmax offset: no per-stage max / rescale unless a score passes the e4m3 headroom


def wsv2_ext():
    key = ("e", bool(WSV2_PROBE), bool(WSV2_QSS), bool(WSV2_LAZY), bool(WSV2_OCC1), bool(WSV2_PVSPLIT), bool(WSV2_LATESYNC))
    if key not in _V2:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}{'a' if major == 9 else ''}"
        src = ("#define V2_PROBE 1\n" if WSV2_PROBE else "") + ("#define V2_QSS 1\n" if WSV2_QSS else "") + (
            "#define V2_LAZY 1\n" if WSV2_LAZY else "") + ("#define V2_OCC1 1\n" if WSV2_OCC1 else "") + (
            "#define V2_PVSPLIT 1\n" if WSV2_PVSPLIT else "") + ("#define V2_LATESYNC 1\n" if WSV2_LATESYNC else "") + WSV2_SRC
        name = "okc_wsv2_" + format(zlib.crc32(src.encode()) & 0xFFFFFFFF, "08x")
        _V2[key] = lean_load_inline(name, cpp_sources=WSV2_CPP, cuda_sources=src, functions=["wsv2"],
                                    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"])
    return _V2[key]


def _v2_inv_freq(theta, dev):
    key = ("f", theta, str(dev))
    if key not in _V2:  # (the reference's RoPE frequencies, fp32)
        _V2[key] = ((1.0 / (theta ** (torch.arange(0, 64, 2, device=dev).float() / 64))) if theta else
                    torch.zeros(32, device=dev)).contiguous()
    return _V2[key]


def wsv2_forward(data, out=None, pdl=True, grid=None, stages=None, segk=None, probe=None, skew=None):
    """Batched verify (q_len <= 8) on a full layer: [B, L, Hq * 64] bf16."""
    cfg = data["config"]
    B, Hq, L = cfg["batch"], cfg["num_q_heads"], cfg["q_len"]
    assert 1 <= L <= 8 and (cfg["window"] == 0) == (not cfg["rope_theta"]) and cfg["num_kv_heads"] == 2 and B <= 32
    qkv = data["qkv"].contiguous()
    dev = qkv.device
    if out is None:
        out = torch.empty((B, L, Hq * 64), dtype=torch.bfloat16, device=dev)
    key = str(dev)
    if key not in _V2:
        _V2[key] = (torch.zeros((32, 128, 64), dtype=torch.float32, device=dev), torch.zeros((32, 128), dtype=torch.float32, device=dev),
                    torch.full((32, 64, 68), -1, dtype=torch.int32, device=dev), torch.zeros(64, dtype=torch.int32, device=dev))
    acc_o, acc_l, part, cnt = _V2[key]
    props = torch.cuda.get_device_properties(dev)
    grid = grid or (1 if WSV2_OCC1 else 2) * props.multi_processor_count
    stages = stages or (WSV2_STAGES if props.major == 9 else 3)
    pr = 0
    if probe is not None:
        assert WSV2_PROBE, "probe needs the WSV2_PROBE build"
        ptt = torch.zeros((int(grid), 16), dtype=torch.int64, device=dev)
        probe.append(ptt)
        pr = ptt.data_ptr()
    sc = float(cfg["qk_scale_factor"]) * float(cfg["sm_scale"]) * float(cfg["kv_scale"]) * 1.4426950408889634
    wsv2_ext().wsv2(qkv.data_ptr(), data["cache_loc"].data_ptr(), data["k_cache"].data_ptr(), data["v_cache"].data_ptr(),
                    data["page_table"].data_ptr(), data["seq_lens"].data_ptr(), acc_o.data_ptr(), acc_l.data_ptr(),
                    part.data_ptr(), cnt.data_ptr(), out.data_ptr(), B, Hq, L, cfg["page_size"], data["page_table"].shape[1],
                    WSV2_SEGK if segk is None else segk, sc, float(cfg["kv_scale"]), float(cfg["kv_scale"]), int(grid),
                    int(stages), bool(pdl), pr, int(cfg["window"]), _v2_inv_freq(float(cfg["rope_theta"]), dev).data_ptr(),
                    int(round(65536 * (WSV2_SKEW if skew is None else skew))),
                    torch.cuda.current_stream().cuda_stream)
    return out
WSV2_ON = True  # batched verify on full layers
WSV2_SLIDING = False  # batched verify on sliding layers
_kernel_fn_126 = kernel_fn  # noqa: F821


def kernel_fn(data):  # noqa: F811
    cfg = data["config"]
    if (cfg["q_len"] == 8 and 2 <= cfg["batch"] <= 32 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64 and (WSV2_SLIDING if cfg["window"] else WSV2_ON)
            and (cfg["window"] == 0) == (not cfg["rope_theta"]) and torch.cuda.get_device_capability(data["qkv"].device)[0] == 9):
        return wsv2_forward(data, pdl=False)  # noqa: F821
    return _kernel_fn_126(data)
# Active configuration: batched verify on full layers stays on the ws kernel (wsv2 off); wsd2 runs 3 CTAs per SM
# (WSD2_OCC3: 3 stages, setmaxnreg 32 / 128) with the transposed 32-key consumer.
WSV2_ON = False
WSD2_T = True
WSD2_T2 = True
WSD2_REG = True
WSD2_STAGES = 4
WSD2_OCC3 = True
WSD2_B1_GRID = 0  # > 0: batch-1 decode on full layers through wsd2 on this many CTAs
WSV2_B1_GRID = 0  # > 0: batch-1 verify on full layers through wsv2 on this many CTAs
_kernel_fn_128 = kernel_fn  # noqa: F821


def kernel_fn(data):  # noqa: F811
    cfg = data["config"]
    if (cfg["batch"] == 1 and cfg["window"] == 0 and not cfg["rope_theta"] and cfg["num_q_heads"] == 16
            and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64
            and torch.cuda.get_device_capability(data["qkv"].device)[0] == 9):
        if cfg["q_len"] == 1 and WSD2_B1_GRID:
            return wsd2_forward(data, pdl=False, grid=WSD2_B1_GRID)  # noqa: F821
        if cfg["q_len"] == 8 and WSV2_B1_GRID:
            return wsv2_forward(data, pdl=False, grid=WSV2_B1_GRID)  # noqa: F821
    return _kernel_fn_128(data)
