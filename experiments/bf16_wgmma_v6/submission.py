# okc: problem=muse_bf16_proj
"""Hopper WGMMA larger-batch probe over the validated v5 dispatcher.

Small pooler batches use the transposed split-K path. Remaining dense batches
retain cuBLAS after measured Triton regressions. Weights retain their required
precision; the router alone uses its explicitly permitted BF16 hi/lo split.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


@triton.jit
def _matmul(X, W, LO, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            SPLIT: tl.constexpr, ROUTER: tl.constexpr, WN: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    s = tl.program_id(2)
    ki = s * (K // SPLIT) + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(K // SPLIT // BK):
        kk = ki + i * BK
        a = tl.load(X + mi[:, None] * K + kk[None, :], mi[:, None] < M, 0)
        b = tl.load(W + kk[:, None] * WN + ni[None, :], ni[None, :] < N, 0)
        acc = tl.dot(a, b, acc)
        if ROUTER:
            lo = tl.load(LO + kk[:, None] * WN + ni[None, :], ni[None, :] < N, 0)
            acc = tl.dot(a, lo, acc)
    tl.store(P + s * M * N + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _finish(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    sums = tl.full((B,), 0, tl.float32)
    for s in range(SPLIT):
        sums += tl.load(P + s * SIZE + idx, idx < SIZE, 0)
    tl.store(Y + idx, sums, idx < SIZE)


@triton.jit
def _gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    for start in range(K // BK):
        kk = start * BK + cols
        a = tl.load(X + kk).to(tl.float32)
        b = tl.load(W + rows[:, None] * K + kk[None, :],
                    rows[:, None] < N, 0).to(tl.float32)
        acc = tl.fma(b, a[None, :], acc)
    result = tl.sum(acc, 1)
    tl.store(Y + rows, result, rows < N)


def prepare(weights: Weights):
    state = dict(weights)
    hi = weights['router'].to(torch.bfloat16)
    state['router_hi'] = hi
    state['router_lo'] = (weights['router'] - hi.float()).to(torch.bfloat16)
    state['transposed'] = {name: state[name].T.contiguous()
                           for name in ('router_hi', 'router_lo', 'pooler', 'qkv_gate', 'o', 'unpooler')}
    return state


def kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:
    """Return x @ W[:n].T, where W is weights[config["layer"]]: bfloat16, or float32 for the router."""
    x = data['x']
    layer = data['config']['layer']
    m, k = x.shape
    n = data['config']['n']
    router = layer == 'router'
    if m == 1 and not router:
        result = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _gemv[(triton.cdiv(n, 4),)](x, weights[layer], result, n, k, 4, 1024,
                                    num_warps=4)
        return result
    if not router and 64 <= m <= 256:
        # BM >= 64 selects SM90 WGMMA instead of the older warp-level MMA.
        # Split only to provide enough CTAs; write directly when unsplit.
        bm, bn, bk = 64, 64, 128
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        split = min(8, triton.next_power_of_2(triton.cdiv(256, tiles)))
        w = weights['transposed'][layer]
        result = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        partial = (result if split == 1 else
                   torch.empty((split, m, n), device=x.device, dtype=torch.float32))
        _matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
            x, w, w, partial, m, n, k, bm, bn, bk, split, False, w.shape[1],
            num_warps=4, num_stages=3)
        if split != 1:
            _finish[(triton.cdiv(m*n, 512),)](partial, result, m*n, split, 512)
        return result
    if not router and (layer != 'pooler' or m > 16):
        return x @ weights[layer][:n].T
    packed = weights['transposed']
    w = packed['router_hi'] if router else packed[layer]
    lo = packed['router_lo'] if router else w
    bm, bn, bk = (16 if m <= 32 else 32), 64, 128
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    split = min(16, triton.next_power_of_2(triton.cdiv(256, tiles)))
    result = torch.empty((m, n), device=x.device,
                         dtype=torch.float32 if router else torch.bfloat16)
    partial = (result if split == 1 else
               torch.empty((split, m, n), device=x.device, dtype=torch.float32))
    _matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        x, w, lo, partial, m, n, k, bm, bn, bk, split, router, w.shape[1],
        num_warps=4, num_stages=3)
    if split != 1:
        _finish[(triton.cdiv(m*n, 512),)](partial, result, m*n, split, 512)
    return result
