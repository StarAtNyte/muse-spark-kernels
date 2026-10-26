"""Weight-major NVFP4 expert kernels for Hopper (decode / short-verify batches of Muse Spark Open).

The experts are held once, in a layout whose per-lane 128-bit loads are exactly the mma A fragments of the weight-major
(swap-AB) GEMM.  Batches of up to ``WM_MAX_TOKENS`` tokens run on these kernels; larger batches convert the layer's weights
to Marlin's layout into a shared scratch buffer and use the stock Marlin kernels, so no second copy of the experts is kept.
"""

import threading
from typing import Optional

import torch

WM_MAX_TOKENS = 24576
_NUM_LOCAL = 32
_HIDDEN = 4096
_INTER = 4096
_TOP_K = 8
_SM_MAJOR = 9

_CUDA_SRC = r'''__CUDA__'''
_CPP_SRC = r'''__CPP__'''

_module = None
_module_lock = threading.Lock()
_scratch = None
_empty_i32 = None


def _get_module():
    global _module
    if _module is None:
        with _module_lock:
            if _module is None:
                import os

                from torch.utils.cpp_extension import load_inline

                last_error = None
                for build_dir in (None, "/tmp/muse_wm_experts_build"):   # the default cache may be read-only in a container
                    try:
                        if build_dir is not None:
                            os.makedirs(build_dir, exist_ok=True)
                        mod = load_inline(
                            "muse_wm_experts_v1",
                            _CPP_SRC,
                            cuda_sources=_CUDA_SRC,
                            functions=["wm_route", "wm_gemm_run", "wm_set_pdl_t", "wm_permx", "conv_raw_to_wm_t", "conv_wm_to_marlin_t"],
                            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90,code=sm_90"],
                            build_directory=build_dir,
                            verbose=False,
                        )
                        break
                    except Exception as error:  # noqa: BLE001
                        last_error = error
                else:
                    raise last_error
                mod.wm_set_pdl_t(1)
                _module = mod
    return _module


# (max tokens) -> (tiles per CTA, warps per CTA, ring variant, 8-token groups per work item); chosen on the H200 for the kernel benchmark (batch sizes 1..256)
_CFG = [(1, (2, 4, 0, 1)), (8, (2, 2, 0, 1)), (16, (2, 4, 0, 2)), (32, (4, 4, 1, 4)), (10**9, (4, 4, 4, 4))]


def _cfg(m: int):
    for bound, cfg in _CFG:
        if m <= bound:
            return cfg
    return _CFG[-1][1]


def supports(layer) -> bool:
    """The wm path is written for this model's expert shape under 8-way expert parallelism on SM90."""
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == _SM_MAJOR
        and layer.num_local_experts == _NUM_LOCAL
        and layer.global_num_experts == 256
        and layer.hidden_size == _HIDDEN
        and layer.intermediate_size_per_partition == _INTER
        and layer.top_k == _TOP_K
    )


def _process_global_scale(gs: torch.Tensor) -> torch.Tensor:
    """gs * 2^119 in bf16, as nvfp4_marlin_process_global_scale does for bfloat16 activations."""
    return (gs.to(torch.bfloat16) * (2.0 ** 119)).contiguous()


class WmExperts:
    """Weight-major copies of one layer's experts plus the small tensors Marlin needs to be fed on demand."""

    def __init__(self, w13, s13, w2, s2, gs13_marlin, gs2_marlin):
        self.w13, self.s13, self.w2, self.s2 = w13, s13, w2, s2
        self.gs13_marlin, self.gs2_marlin = gs13_marlin, gs2_marlin    # bf16 [32], as Marlin expects them
        self.gs13 = (gs13_marlin.float() * (2.0 ** -119)).contiguous()  # exact power-of-two rescale back to the real scale
        self.gs2 = (gs2_marlin.float() * (2.0 ** -119)).contiguous()


def build_wm_experts(w13_raw, w13_sf, w13_gs, w2_raw, w2_sf, w2_gs) -> WmExperts:
    """raw HF-layout NVFP4 tensors of one layer's local experts -> weight-major layout (does not free the inputs)."""
    mod = _get_module()
    dev = w13_raw.device
    E, R13, K2 = w13_raw.shape
    out13 = torch.empty((E, R13 // 16, K2 * 2 // 64, 32, 4), dtype=torch.int32, device=dev)
    s13 = torch.empty((E, R13 // 16, K2 * 2 // 256, 32, 8), dtype=torch.uint8, device=dev)
    mod.conv_raw_to_wm_t(w13_raw.contiguous(), w13_sf.contiguous().view(torch.uint8), out13, s13)
    E2, R2, K22 = w2_raw.shape
    out2 = torch.empty((E2, R2 // 16, K22 * 2 // 64, 32, 4), dtype=torch.int32, device=dev)
    s2 = torch.empty((E2, R2 // 16, K22 * 2 // 256, 32, 8), dtype=torch.uint8, device=dev)
    mod.conv_raw_to_wm_t(w2_raw.contiguous(), w2_sf.contiguous().view(torch.uint8), out2, s2)
    return WmExperts(out13, s13, out2, s2, _process_global_scale(w13_gs), _process_global_scale(w2_gs))


class _Scratch:
    """One Marlin-layout copy of an expert bank, shared by every layer (layers run one after the other on a stream)."""

    def __init__(self, device):
        k16 = _HIDDEN // 16
        self.w13 = torch.empty((_NUM_LOCAL, k16, 2 * _INTER // 64 * 128), dtype=torch.int32, device=device)
        self.s13 = torch.empty((_NUM_LOCAL, k16, 2 * _INTER), dtype=torch.float8_e4m3fn, device=device)
        self.w2 = torch.empty((_NUM_LOCAL, _INTER // 16, _HIDDEN // 64 * 128), dtype=torch.int32, device=device)
        self.s2 = torch.empty((_NUM_LOCAL, _INTER // 16, _HIDDEN), dtype=torch.float8_e4m3fn, device=device)


def reserve_scratch(device) -> None:
    global _scratch
    if _scratch is None:
        _scratch = _Scratch(device)


def marlin_view(wm: WmExperts):
    """Convert one layer's weights into the shared Marlin scratch; returns (w13, s13, gs13, w2, s2, gs2)."""
    mod = _get_module()
    reserve_scratch(wm.w13.device)
    sc = _scratch
    mod.conv_wm_to_marlin_t(wm.w13, wm.s13, sc.w13, sc.s13.view(torch.uint8), 2 * _INTER, _HIDDEN)
    mod.conv_wm_to_marlin_t(wm.w2, wm.s2, sc.w2, sc.s2.view(torch.uint8), _HIDDEN, _INTER)
    return sc.w13, sc.s13, wm.gs13_marlin, sc.w2, sc.s2, wm.gs2_marlin


def expert_outputs(wm: WmExperts, hidden_states: torch.Tensor, topk_ids: torch.Tensor, local_start: int):
    """Routed experts for up to WM_MAX_TOKENS tokens.  Returns (y [m, top_k, H] bf16 with unspecified rows for routes
    outside this GPU, local expert id per route (-1 outside) [m, top_k] int32)."""
    global _empty_i32
    mod = _get_module()
    x = hidden_states.contiguous()
    m = x.shape[0]
    dev = x.device
    if m == 0:
        return x.new_empty((0, _TOP_K, _HIDDEN)), torch.empty((0, _TOP_K), dtype=torch.int32, device=dev)
    if _empty_i32 is None or _empty_i32.device != dev:
        _empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
    nt, nw, ring, nch = _cfg(m)
    lmax = (m + 7) // 8 * 8
    ids = topk_ids.to(torch.int32).contiguous()
    cnt = torch.empty(_NUM_LOCAL, dtype=torch.int32, device=dev)
    lst = torch.empty(_NUM_LOCAL * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(4 * (_NUM_LOCAL + (m * _TOP_K) // (nch * 8) + 8), dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(m * _TOP_K, dtype=torch.int32, device=dev)
    mod.wm_route(ids, _empty_i32, local_start, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
    maxslots = min(m * _TOP_K, _NUM_LOCAL + (m * _TOP_K) // (nch * 8))
    inter = torch.empty((m * _TOP_K, _INTER), dtype=torch.bfloat16, device=dev)
    y = torch.empty((m * _TOP_K, _HIDDEN), dtype=torch.bfloat16, device=dev)
    if ring == 7:  # wgmma path reads its activations in fragment order (the first GEMM writes `inter` that way itself)
        xp = torch.empty_like(x)
        mod.wm_permx(x, xp)
        x = xp
    mod.wm_gemm_run(0, nch, nt, nw, ring, 0, wm.w13, wm.s13, wm.gs13, x, inter, cnt, lst, act, nact, woff, maxslots, lmax)
    mod.wm_gemm_run(1, nch, nt, nw, ring, 0, wm.w2, wm.s2, wm.gs2, inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
    return y.view(m, _TOP_K, _HIDDEN), lids.view(m, _TOP_K)
