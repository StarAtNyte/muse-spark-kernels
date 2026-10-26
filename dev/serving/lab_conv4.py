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


_WMX_SRC = '"""Weight-major NVFP4 expert kernels for Hopper (decode / short-verify batches of Muse Spark Open).\n\nThe experts are held once, in a layout whose per-lane 128-bit loads are exactly the mma A fragments of the weight-major\n(swap-AB) GEMM.  Batches of up to ``WM_MAX_TOKENS`` tokens run on these kernels; larger batches convert the layer\'s weights\nto Marlin\'s layout into a shared scratch buffer and use the stock Marlin kernels, so no second copy of the experts is kept.\n"""\n\nimport threading\nfrom typing import Optional\n\nimport torch\n\nWM_MAX_TOKENS = 24576\n_NUM_LOCAL = 32\n_HIDDEN = 4096\n_INTER = 4096\n_TOP_K = 8\n_SM_MAJOR = 9\n\n_CUDA_SRC = r\'\'\'\n#include <cuda_bf16.h>\n#include <cuda_runtime.h>\n#include <stdint.h>\n\ntypedef __nv_bfloat16 bf16;\ntypedef __nv_bfloat162 bf162;\n\n#define HD 4096\n#define HD_ 4096\n#define ID 4096\n#define KC 64            // k64 chunks along K = 4096\n#define KCG 16           // groups of 4 chunks (scale words)\n#define RT0 512          // 16-row tiles of w13 (gate 256 + up 256)\n#define RT1 256          // 16-row tiles of w2\n\nconstexpr uint32_t MASK = 0x81C081C0u;\n\n// Weight word -> bf16x2 holding the E2M1 value times 2^-126 (denormal-linear encoding).\ntemplate <int J>\n__device__ __forceinline__ uint32_t dec(uint32_t w) {\n  if constexpr (J == 0) {\n    return w & MASK;\n  } else if constexpr (J < 3) {\n    return __funnelshift_r(w, w, 3 * J) & MASK;\n  } else {\n    return ((w & 0x001B001Bu) * 0x1084u) & MASK;  // sources m0@4 m1@0 m2@1 sign@3 (+16 for the high half)\n  }\n}\n\n__device__ __forceinline__ uint32_t scale2(uint32_t q, int c) {\n  // byte c of q holds b\'; scale = bf16 bits (b\'<<4) + 0x7600 = S * 2^118\n  uint32_t bits = ((q >> (8 * c)) & 0xFFu) << 4;\n  return (bits + 0x7600u) * 0x10001u;\n}\n\n__device__ __forceinline__ uint32_t hmul(uint32_t a, uint32_t b) {\n  uint32_t r;\n  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));\n  return r;\n}\n\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }\n\n__device__ __forceinline__ void pdl_wait() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void pdl_trigger() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n#endif\n}\n\nstatic int g_pdl = 0;\nstatic int g_persist = 0;\nextern "C" void wm_set_persist(int v) { g_persist = v; }\nextern "C" void wm_set_pdl(int v) { g_pdl = v; }\n\ntemplate <typename... KArgs, typename... Args>\nstatic void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, bool pdl,\n                     Args... args) {\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = grid;\n  cfg.blockDim = block;\n  cfg.dynamicSmemBytes = smem;\n  cfg.stream = st;\n  cudaLaunchAttribute attr[1];\n  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n  attr[0].val.programmaticStreamSerializationAllowed = 1;\n  cfg.attrs = attr;\n  cfg.numAttrs = pdl ? 1 : 0;\n  cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);\n}\n\n__global__ void route_kernel(const int* __restrict__ ids, int npair, const int* __restrict__ ls_ptr, int ls_val, int lmax,\n                             int* __restrict__ cnt, int* __restrict__ lst, int* __restrict__ act,\n                             int* __restrict__ nact, int* __restrict__ local_ids, int* __restrict__ woff, int grp, int nslots) {\n  __shared__ int s_cnt[32];\n  pdl_wait();\n  const int ls = ls_ptr ? *ls_ptr : ls_val;\n  pdl_trigger();\n  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;\n  for (int i = threadIdx.x; i < nslots; i += blockDim.x) reinterpret_cast<int4*>(act)[i] = make_int4(0, 0, 0, 0);  // work items {expert, offset, tokens, 0}\n  __syncthreads();\n  for (int p = threadIdx.x; p < npair; p += blockDim.x) {\n    int e = (ids[p] - ls) & 255;  // experts are ordered mod 256 starting at this GPU\'s first expert\n    bool ok = e < 32;\n    local_ids[p] = ok ? e : -1;\n    if (ok) {\n      int pos = atomicAdd(&s_cnt[e], 1);\n      lst[e * lmax + pos] = p;\n    }\n  }\n  __syncthreads();\n  if (threadIdx.x < 32) {\n    int lane = threadIdx.x;\n    int c = s_cnt[lane];\n    cnt[lane] = c;\n    // work items: one per group of `grp` tokens of an expert, so a hot expert spreads over several CTAs\n    const int groups = (c + grp - 1) / grp;\n    int scan = groups;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int v = __shfl_up_sync(0xffffffffu, scan, d);\n      if (lane >= d) scan += v;\n    }\n    const int start = scan - groups;\n    for (int j = 0; j < groups; j++)\n      reinterpret_cast<int4*>(act)[start + j] = make_int4(lane, j * grp, min(c - j * grp, grp), 1);\n    if (lane == 31) *nact = scan;\n  }\n}\n\n__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),\n               "l"(gsrc));\n}\n__device__ __forceinline__ void cp16z(void* sdst, const void* gsrc, uint32_t nbytes) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),\n               "l"(gsrc), "r"(nbytes));\n}\ntemplate <int NTB_, int NCH_>\n__host__ __device__ constexpr int ring2_stages() {\n  return (14336 / (512 * (NTB_ + 2 * NCH_))) > 6 ? 6 : ((14336 / (512 * (NTB_ + 2 * NCH_))) < 3 ? 3 : (14336 / (512 * (NTB_ + 2 * NCH_))));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N>\n__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n\ntemplate <int NTB, int NCH, int NW>\n__device__ __forceinline__ void wm_epilogue(float (&acc)[NTB][NCH][4], float* red, const float gs, const int* lst,\n                                            int ntok, int tbase, int mode, int nt, bf16* OUT, const int bx) {\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n#pragma unroll\n  for (int n = 0; n < NTB; n++)\n#pragma unroll\n    for (int c = 0; c < NCH; c++)\n#pragma unroll\n      for (int r = 0; r < 4; r++)\n        red[(((warp * NTB + n) * NCH + c) * 4 + r) * 32 + lane] = acc[n][c][r];\n  __syncthreads();\n  for (int item = threadIdx.x; item < NCH * 4 * 32; item += NW * 32) {\n    const int li = item & 31, r = (item >> 5) & 3, c = item >> 7;\n    const int row = (li >> 2) + 8 * (r >> 1);\n    const int col = c * 8 + 2 * (li & 3) + (r & 1);\n    if (col >= ntok) continue;\n    const int p = __ldcg(lst + col);\n#pragma unroll\n    for (int n = 0; n < NTB; n++) {\n      if (mode == 0 && (n & 1)) continue;\n      float s = 0.f;\n#pragma unroll\n      for (int w = 0; w < NW; w++) s += red[(((w * NTB + n) * NCH + c) * 4 + r) * 32 + li];\n      if (mode == 0) {\n        float s2 = 0.f;\n#pragma unroll\n        for (int w = 0; w < NW; w++) s2 += red[(((w * NTB + n + 1) * NCH + c) * 4 + r) * 32 + li];\n        const float gate = bfr(s * gs), up = bfr(s2 * gs);\n        const float sil = gate / (1.0f + __expf(-gate));\n        OUT[(size_t)p * ID + (bx * (nt / 2) + ((tbase + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);\n      } else {\n        OUT[(size_t)p * HD + (bx * nt + tbase + n) * 16 + row] = __float2bfloat16(s * gs);\n      }\n    }\n  }\n  __syncthreads();\n}\n\n// RING == false: every lane issues its weight loads up front, group by group (best for small problems).\n// RING == true : weights stream through a per-warp cp.async ring in shared memory (lane-private slots).\ntemplate <int MODE, int NT, int NTB, int NCH, int NW, int RING, int S, int DBG>\n__device__ __forceinline__ void wm_body(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq,\n                                        const float* __restrict__ GS, const bf16* __restrict__ X,\n                                        bf16* __restrict__ OUT, const int* __restrict__ lst, int e, int ntok,\n                                        int tbase, uint4* smem, const int bx) {\n  constexpr int RT = MODE == 0 ? RT0 : RT1;\n  constexpr int XW = MODE == 0 ? HD : ID;\n  constexpr int CPW = 64 / NW;  // chunks of k64 per warp\n  constexpr int GPW = CPW / 4;  // scale groups per warp\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  float* red = reinterpret_cast<float*>(smem);\n\n  int tile[NTB];\n#pragma unroll\n  for (int n = 0; n < NTB; n++) {\n    const int nn = tbase + n;\n    if (MODE == 0) tile[n] = (nn & 1) ? RT0 / 2 + bx * (NT / 2) + (nn >> 1) : bx * (NT / 2) + (nn >> 1);\n    else tile[n] = bx * NT + nn;\n  }\n\n  const bf16* xp[NCH];\n#pragma unroll\n  for (int c = 0; c < NCH; c++) {\n    int col = c * 8 + g;\n    xp[c] = nullptr;\n    if (col < ntok) {\n      int p = __ldcg(lst + col);\n      xp[c] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + 16 * t + warp * CPW * 64;\n    }\n  }\n\n  float acc[NTB][NCH][4];\n#pragma unroll\n  for (int n = 0; n < NTB; n++)\n#pragma unroll\n    for (int c = 0; c < NCH; c++)\n#pragma unroll\n      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;\n\n  uint32_t sink = 0;\n  const uint4* wsrc[NTB];\n  const uint2* ssrc[NTB];\n#pragma unroll\n  for (int n = 0; n < NTB; n++) {\n    wsrc[n] = Wq + ((size_t)e * RT + tile[n]) * KC * 32 + warp * CPW * 32 + lane;\n    ssrc[n] = Sq + ((size_t)e * RT + tile[n]) * KCG * 32 + warp * GPW * 32 + lane;\n  }\n\n  if (RING == 0) {\n#pragma unroll\n    for (int q = 0; q < GPW; q++) {\n      uint2 sc[NTB];\n      uint4 wv[NTB][4];\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        sc[n] = __ldg(ssrc[n] + q * 32);\n#pragma unroll\n        for (int c = 0; c < 4; c++) {\n          if constexpr (DBG & 8) wv[n][c] = make_uint4(lane * 7 + c, lane * 13 + q, lane + n, lane * 3);\n          else wv[n][c] = __ldg(wsrc[n] + (q * 4 + c) * 32);\n        }\n      }\n#pragma unroll\n      for (int c = 0; c < 4; c++) {\n        uint4 xa[NCH], xb[NCH];\n#pragma unroll\n        for (int h = 0; h < NCH; h++) {\n          if (xp[h] && !(DBG & 4)) {\n            const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (q * 4 + c) * 64);\n            xa[h] = __ldcg(xv);\n            xb[h] = __ldcg(xv + 1);\n          } else {\n            xa[h] = make_uint4(0, 0, 0, 0);\n            xb[h] = make_uint4(0, 0, 0, 0);\n          }\n        }\n#pragma unroll\n        for (int n = 0; n < NTB; n++) {\n          const uint32_t s0 = scale2(sc[n].x, c);\n          const uint32_t s1 = scale2(sc[n].y, c);\n          const uint32_t ww[4] = {wv[n][c].x, wv[n][c].y, wv[n][c].z, wv[n][c].w};\n#pragma unroll\n          for (int ks = 0; ks < 4; ks++) {\n            const uint32_t w = ww[ks];\n            uint32_t a0, a1, a2, a3;\n            if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }\n            else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }\n#pragma unroll\n            for (int h = 0; h < NCH; h++) {\n              const uint4 xs = ks < 2 ? xa[h] : xb[h];\n              if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);\n              else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);\n            }\n          }\n        }\n      }\n    }\n  } else if (RING == 2) {\n    constexpr int XS = NCH * 2;\n    constexpr int STG = NTB + XS;\n    uint4* ring = smem + warp * (S * STG * 32);\n    auto issue = [&](int c) {\n      if (c < CPW && !(DBG & 8)) {\n#pragma unroll\n        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * STG + n) * 32 + lane], wsrc[n] + c * 32);\n      }\n      if (c < CPW) {\n#pragma unroll\n        for (int h = 0; h < NCH; h++) {\n          const bool ok = xp[h] != nullptr && !(DBG & 4);\n          const bf16* src = ok ? xp[h] + c * 64 : X;\n          cp16z(&ring[((c % S) * STG + NTB + 2 * h) * 32 + lane], src, ok ? 16u : 0u);\n          cp16z(&ring[((c % S) * STG + NTB + 2 * h + 1) * 32 + lane], src + 8, ok ? 16u : 0u);\n        }\n      }\n      cp_commit();\n    };\n#pragma unroll\n    for (int c = 0; c < S - 1; c++) issue(c);\n    uint2 scur[NTB], snxt[NTB];\n#pragma unroll\n    for (int n = 0; n < NTB; n++) {\n      scur[n] = __ldg(ssrc[n]);\n      snxt[n] = GPW > 1 ? __ldg(ssrc[n] + 32) : scur[n];\n    }\n#pragma unroll 1\n    for (int i = 0; i < CPW; i++) {\n      cp_wait<S - 2>();\n      issue(i + S - 1);\n      const int cq = i & 3;\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        const uint4 wv = ring[((i % S) * STG + n) * 32 + lane];\n        const uint32_t s0 = scale2(scur[n].x, cq);\n        const uint32_t s1 = scale2(scur[n].y, cq);\n        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};\n#pragma unroll\n        for (int ks = 0; ks < 4; ks++) {\n          const uint32_t w = ww[ks];\n          uint32_t a0, a1, a2, a3;\n          if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }\n          else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }\n#pragma unroll\n          for (int h = 0; h < NCH; h++) {\n            const uint4 xs = ring[((i % S) * STG + NTB + 2 * h + (ks >> 1)) * 32 + lane];\n            if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);\n            else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);\n          }\n        }\n      }\n      if (cq == 3) {\n#pragma unroll\n        for (int n = 0; n < NTB; n++) {\n          scur[n] = snxt[n];\n          if ((i >> 2) + 2 < GPW) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);\n        }\n      }\n    }\n    cp_wait<0>();\n  } else {\n    uint4* ring = smem + warp * (S * NTB * 32);\n    auto issue = [&](int c) {\n      if (c < CPW && !(DBG & 8)) {\n#pragma unroll\n        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * NTB + n) * 32 + lane], wsrc[n] + c * 32);\n      }\n      cp_commit();\n    };\n#pragma unroll\n    for (int c = 0; c < S - 1; c++) issue(c);\n\n    uint2 scur[NTB], snxt[NTB];\n#pragma unroll\n    for (int n = 0; n < NTB; n++) {\n      scur[n] = __ldg(ssrc[n]);\n      snxt[n] = GPW > 1 ? __ldg(ssrc[n] + 32) : scur[n];\n    }\n    uint4 xcur[NCH][2], xnxt[NCH][2];\n#pragma unroll\n    for (int h = 0; h < NCH; h++) {\n      if (xp[h] && !(DBG & 4)) {\n        const uint4* xv = reinterpret_cast<const uint4*>(xp[h]);\n        xcur[h][0] = __ldcg(xv);\n        xcur[h][1] = __ldcg(xv + 1);\n      } else {\n        xcur[h][0] = make_uint4(0, 0, 0, 0);\n        xcur[h][1] = make_uint4(0, 0, 0, 0);\n      }\n      xnxt[h][0] = xcur[h][0];\n      xnxt[h][1] = xcur[h][1];\n    }\n\n#pragma unroll 1\n    for (int i = 0; i < CPW; i++) {\n      cp_wait<S - 2>();\n      issue(i + S - 1);\n#pragma unroll\n      for (int h = 0; h < NCH; h++) {\n        if (xp[h] && i + 1 < CPW && !(DBG & 4)) {\n          const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (i + 1) * 64);\n          xnxt[h][0] = __ldcg(xv);\n          xnxt[h][1] = __ldcg(xv + 1);\n        }\n      }\n      const int cq = i & 3;\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        const uint4 wv = ring[((i % S) * NTB + n) * 32 + lane];\n        const uint32_t s0 = scale2(scur[n].x, cq);\n        const uint32_t s1 = scale2(scur[n].y, cq);\n        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};\n#pragma unroll\n        for (int ks = 0; ks < 4; ks++) {\n          const uint32_t w = ww[ks];\n          uint32_t a0, a1, a2, a3;\n          if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }\n          else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }\n#pragma unroll\n          for (int h = 0; h < NCH; h++) {\n            const uint4 xs = xcur[h][ks >> 1];\n            if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);\n            else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);\n          }\n        }\n      }\n      if (cq == 3) {\n#pragma unroll\n        for (int n = 0; n < NTB; n++) {\n          scur[n] = snxt[n];\n          if ((i >> 2) + 2 < GPW) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);\n        }\n      }\n#pragma unroll\n      for (int h = 0; h < NCH; h++) {\n        xcur[h][0] = xnxt[h][0];\n        xcur[h][1] = xnxt[h][1];\n      }\n    }\n    cp_wait<0>();\n  }\n\n  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);\n  __syncthreads();  // the ring is dead; the same smem holds the cross-warp reduction\n  wm_epilogue<NTB, NCH, NW>(acc, red, GS[e] * 256.0f, lst, ntok, tbase, MODE, NT, OUT, bx);\n}\n\n#ifndef WM_MINB\n#define WM_MINB 4\n#endif\ntemplate <int MODE, int NT, int NMAX, int NW, int RING, int DBG>\n__global__ void __launch_bounds__(NW * 32, (RING == 3 ? 2 : WM_MINB))\nwm_gemm(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,\n        const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ cnt,\n        const int* __restrict__ lst, const int* __restrict__ act, const int* __restrict__ nact_p,\n        const int* __restrict__ woff, int lmax, int persist) {\n  extern __shared__ uint4 smem[];\n  constexpr int TILES = (MODE == 0 ? 512 : 256) / NT;\n  pdl_wait();  // everything below (items, token lists, activations) comes from earlier kernels\n  pdl_trigger();\n  const int nitems = persist ? __ldcg(nact_p) : 1;\n  for (int u = blockIdx.x;; u += gridDim.x) {\n  int slot, bx;\n  if (persist) {\n    if (u >= nitems * TILES) break;\n    slot = u / TILES; bx = u - slot * TILES;\n  } else { slot = blockIdx.y; bx = blockIdx.x; }\n  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + slot);  // {expert, offset, tokens, valid}\n  if (item.z != 0) {\n  const int e = item.x;\n  const int off = item.y;\n  {\n    const int ntok = item.z;\n    const int* l = lst + e * lmax + off;\n    if (ntok <= 8) {\n      wm_body<MODE, NT, NT, 1, NW, RING, (RING == 2 ? ring2_stages<NT, 1>() : (NT == 8 ? 3 : NT == 4 ? 4 : 6)), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, 0, smem, bx);\n    } else if (NMAX >= 2 && ntok <= 16) {\n#pragma unroll 1\n      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))\n        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 2 ? 2 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 2 ? 2 : 1)>() : RING == 3 ? 4 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);\n    } else if (NMAX >= 4) {\n#pragma unroll 1\n      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))\n        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 4 ? 4 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 4 ? 4 : 1)>() : RING == 3 ? 4 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);\n    }\n  }\n  }\n  if (!persist) break;\n  }\n}\n\n\n\n// ---- row-split variant: the four warps of a CTA own different weight rows but the same activations, which are\n// staged once per CTA through shared memory.  No cross-warp reduction; x traffic from L2 drops 4x. ----\ntemplate <int MODE, int NTB, int NCH, int S, int DBG>\n__device__ __forceinline__ void wm_body_rs(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq,\n                                           const float* __restrict__ GS, const bf16* __restrict__ X,\n                                           bf16* __restrict__ OUT, const int* __restrict__ lst, int e, int ntok,\n                                           uint4* smem, const int bx) {\n  constexpr int RT = MODE == 0 ? RT0 : RT1;\n  constexpr int XW = MODE == 0 ? HD : ID;\n  constexpr int NTC = 4 * NTB;  // tiles per CTA\n  constexpr int XSLOTS = NCH * 2 * 32;\n  constexpr int RPT = (XSLOTS + 127) / 128;\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  uint4* wring = smem;                       // [S][4][NTB][32]\n  uint4* xring = smem + S * 4 * NTB * 32;    // [S][XSLOTS]\n\n  const uint4* wsrc[NTB];\n  const uint2* ssrc[NTB];\n#pragma unroll\n  for (int n = 0; n < NTB; n++) {\n    const int nn = warp * NTB + n;\n    const int tile = MODE == 0 ? ((nn & 1) ? RT0 / 2 + bx * (NTC / 2) + (nn >> 1) : bx * (NTC / 2) + (nn >> 1)) : bx * NTC + nn;\n    wsrc[n] = Wq + ((size_t)e * RT + tile) * KC * 32 + lane;\n    ssrc[n] = Sq + ((size_t)e * RT + tile) * KCG * 32 + lane;\n  }\n  const bf16* xb[RPT];\n  bool xok[RPT];\n#pragma unroll\n  for (int r = 0; r < RPT; r++) {\n    const int sl = tid + 128 * r;\n    xok[r] = false;\n    xb[r] = X;\n    if (sl < XSLOTS) {\n      const int h = sl >> 6, j = (sl >> 5) & 1, L = sl & 31;\n      const int col = h * 8 + (L >> 2);\n      if (col < ntok) {\n        const int p = __ldcg(lst + col);\n        xb[r] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + 16 * (L & 3) + 8 * j;\n        xok[r] = true;\n      }\n    }\n  }\n\n  uint32_t sink = 0;\n  float acc[NTB][NCH][4];\n#pragma unroll\n  for (int n = 0; n < NTB; n++)\n#pragma unroll\n    for (int c = 0; c < NCH; c++)\n#pragma unroll\n      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;\n\n  auto issue = [&](int c) {\n    if (c < KC) {\n#pragma unroll\n      for (int n = 0; n < NTB; n++) { if constexpr (!(DBG & 8)) cp16(&wring[(((c % S) * 4 + warp) * NTB + n) * 32 + lane], wsrc[n] + c * 32); }\n#pragma unroll\n      for (int r = 0; r < RPT; r++)\n        if (tid + 128 * r < XSLOTS && !(DBG & 4)) cp16z(&xring[(c % S) * XSLOTS + tid + 128 * r], xb[r] + c * 64, xok[r] ? 16u : 0u);\n    }\n    cp_commit();\n  };\n#pragma unroll\n  for (int c = 0; c < S - 1; c++) issue(c);\n  uint2 scur[NTB], snxt[NTB];\n#pragma unroll\n  for (int n = 0; n < NTB; n++) {\n    scur[n] = __ldg(ssrc[n]);\n    snxt[n] = __ldg(ssrc[n] + 32);\n  }\n#pragma unroll 1\n  for (int i = 0; i < KC; i++) {\n    cp_wait<S - 2>();\n    __syncthreads();\n    issue(i + S - 1);\n    const int cq = i & 3;\n    uint4 xr[NCH][2];\n#pragma unroll\n    for (int h = 0; h < NCH; h++) {\n      xr[h][0] = xring[(i % S) * XSLOTS + (h * 2) * 32 + lane];\n      xr[h][1] = xring[(i % S) * XSLOTS + (h * 2 + 1) * 32 + lane];\n    }\n#pragma unroll\n    for (int n = 0; n < NTB; n++) {\n      const uint4 wv = wring[(((i % S) * 4 + warp) * NTB + n) * 32 + lane];\n      const uint32_t s0 = scale2(scur[n].x, cq);\n      const uint32_t s1 = scale2(scur[n].y, cq);\n      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};\n#pragma unroll\n      for (int ks = 0; ks < 4; ks++) {\n        const uint32_t w = ww[ks];\n        uint32_t a0, a1, a2, a3;\n        if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }\n        else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }\n#pragma unroll\n        for (int h = 0; h < NCH; h++) {\n          const uint4 xs = xr[h][ks >> 1];\n          if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);\n          else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);\n        }\n      }\n    }\n    if (cq == 3) {\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        scur[n] = snxt[n];\n        if ((i >> 2) + 2 < KCG) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);\n      }\n    }\n  }\n  cp_wait<0>();\n  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);\n\n  const float gs = GS[e] * 256.0f;\n#pragma unroll\n  for (int c = 0; c < NCH; c++)\n#pragma unroll\n    for (int r = 0; r < 4; r++) {\n      const int col = c * 8 + 2 * t + (r & 1);\n      if (col >= ntok) continue;\n      const int p = __ldcg(lst + col);\n      const int row = g + 8 * (r >> 1);\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        if (MODE == 0) {\n          if (n & 1) continue;\n          const float gate = bfr(acc[n][c][r] * gs), up = bfr(acc[n + 1][c][r] * gs);\n          const float sil = gate / (1.0f + __expf(-gate));\n          OUT[(size_t)p * ID + (bx * (NTC / 2) + ((warp * NTB + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);\n        } else {\n          OUT[(size_t)p * HD + (bx * NTC + warp * NTB + n) * 16 + row] = __float2bfloat16(acc[n][c][r] * gs);\n        }\n      }\n    }\n}\n\ntemplate <int MODE, int NTB, int NMAX, int S, int DBG>\n__global__ void __launch_bounds__(128, 3)\nwm_gemm_rs(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,\n           const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst,\n           const int* __restrict__ act, int lmax) {\n  extern __shared__ uint4 smem[];\n  pdl_wait();\n  pdl_trigger();\n  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);\n  if (item.z == 0) return;\n  const int e = item.x, ntok = item.z;\n  const int* l = lst + e * lmax + item.y;\n  if (ntok <= 8) wm_body_rs<MODE, NTB, 1, S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);\n  else if (NMAX >= 2 && ntok <= 16) wm_body_rs<MODE, NTB, (NMAX >= 2 ? 2 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);\n  else if (NMAX >= 4 && (NMAX < 8 || ntok <= 32)) wm_body_rs<MODE, NTB, (NMAX >= 4 ? 4 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);\n  else if (NMAX >= 8) wm_body_rs<MODE, NTB, (NMAX >= 8 ? 8 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);\n}\n\ntemplate <int MODE, int NTB, int NMAX, int S, int DBG>\nstatic void launch_rs(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst,\n                      const int* act, int maxslots, int lmax, int pdl, cudaStream_t st) {\n  dim3 grid((MODE == 0 ? 512 : 256) / (4 * NTB), (unsigned)maxslots);\n  const int dyn = S * (4 * NTB * 512 + NMAX * 2 * 32 * 16);\n  cudaFuncSetAttribute(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);\n  launch_k(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),\n           reinterpret_cast<const uint2*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);\n}\n\n// Post-expert RMSNorm + weighted sum: out[m] = sum_k w[m,k] * rmsnorm_k(y[m,k] * nw), skipping slots routed elsewhere.\n// One CTA per token; warp k owns slot k for the sum of squares, then all threads accumulate their columns.\n__global__ void __launch_bounds__(256)\nwm_finish_kernel(const bf16* __restrict__ Y, const float* __restrict__ TW, const int* __restrict__ lids,\n                 const bf16* __restrict__ NW_, bf16* __restrict__ OUT, float eps) {\n  __shared__ float s_scale[8];\n  pdl_wait();\n  pdl_trigger();\n  const int m = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const bool live = lids[m * 8 + warp] >= 0;\n  float ss = 0.f;\n  if (live) {\n    const uint4* yv = reinterpret_cast<const uint4*>(Y + ((size_t)m * 8 + warp) * HD_ * 1);\n    const uint4* wv = reinterpret_cast<const uint4*>(NW_);\n#pragma unroll\n    for (int i = 0; i < 16; i++) {\n      const uint4 a = __ldcg(yv + i * 32 + lane), w = __ldg(wv + i * 32 + lane);\n      const bf162* ap = reinterpret_cast<const bf162*>(&a);\n      const bf162* wp = reinterpret_cast<const bf162*>(&w);\n#pragma unroll\n      for (int j = 0; j < 4; j++) {\n        const float2 af = __bfloat1622float2(ap[j]), wf = __bfloat1622float2(wp[j]);\n        const float v0 = af.x * wf.x, v1 = af.y * wf.y;\n        ss += v0 * v0 + v1 * v1;\n      }\n    }\n#pragma unroll\n    for (int d = 16; d >= 1; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);\n  }\n  if (lane == 0) s_scale[warp] = live ? TW[m * 8 + warp] * rsqrtf(ss / HD_ + eps) : 0.f;\n  __syncthreads();\n  // each thread: 16 columns = two uint4 of bf16\n  float acc[16];\n#pragma unroll\n  for (int j = 0; j < 16; j++) acc[j] = 0.f;\n  const uint4* wv = reinterpret_cast<const uint4*>(NW_) + tid * 2;\n  const uint4 w0 = __ldg(wv), w1 = __ldg(wv + 1);\n  float wf[16];\n  {\n    const bf162* p0 = reinterpret_cast<const bf162*>(&w0);\n    const bf162* p1 = reinterpret_cast<const bf162*>(&w1);\n#pragma unroll\n    for (int j = 0; j < 4; j++) {\n      const float2 a = __bfloat1622float2(p0[j]), b = __bfloat1622float2(p1[j]);\n      wf[2 * j] = a.x; wf[2 * j + 1] = a.y; wf[8 + 2 * j] = b.x; wf[8 + 2 * j + 1] = b.y;\n    }\n  }\n#pragma unroll\n  for (int k = 0; k < 8; k++) {\n    const float sc = s_scale[k];\n    if (sc != 0.f) {\n      const uint4* yv = reinterpret_cast<const uint4*>(Y + ((size_t)m * 8 + k) * HD_) + tid * 2;\n      const uint4 a0 = __ldcg(yv), a1 = __ldcg(yv + 1);\n      const bf162* p0 = reinterpret_cast<const bf162*>(&a0);\n      const bf162* p1 = reinterpret_cast<const bf162*>(&a1);\n#pragma unroll\n      for (int j = 0; j < 4; j++) {\n        const float2 a = __bfloat1622float2(p0[j]), b = __bfloat1622float2(p1[j]);\n        acc[2 * j] += sc * (a.x * wf[2 * j]);\n        acc[2 * j + 1] += sc * (a.y * wf[2 * j + 1]);\n        acc[8 + 2 * j] += sc * (b.x * wf[8 + 2 * j]);\n        acc[8 + 2 * j + 1] += sc * (b.y * wf[8 + 2 * j + 1]);\n      }\n    }\n  }\n  uint4 o0, o1;\n  bf162* q0 = reinterpret_cast<bf162*>(&o0);\n  bf162* q1 = reinterpret_cast<bf162*>(&o1);\n#pragma unroll\n  for (int j = 0; j < 4; j++) {\n    q0[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);\n    q1[j] = __floats2bfloat162_rn(acc[8 + 2 * j], acc[8 + 2 * j + 1]);\n  }\n  uint4* ov = reinterpret_cast<uint4*>(OUT + (size_t)m * HD_) + tid * 2;\n  ov[0] = o0;\n  ov[1] = o1;\n}\nextern "C" void wm_finish_c(const void* Y, const float* TW, const int* lids, const void* NWT, void* OUT, int M, float eps,\n                            cudaStream_t st) {\n  launch_k(wm_finish_kernel, dim3(M), dim3(256), 0, st, g_pdl != 0, reinterpret_cast<const bf16*>(Y), TW, lids,\n           reinterpret_cast<const bf16*>(NWT), reinterpret_cast<bf16*>(OUT), eps);\n}\n\nextern "C" void wm_route_c(const int* ids, int npair, const int* ls_ptr, int ls_val, int lmax, int* cnt, int* lst,\n                           int* act, int* nact, int* local_ids, int* woff, int grp, int nslots, cudaStream_t st) {\n  const int pdl = g_pdl;\n  launch_k(route_kernel, dim3(1), dim3(1024), 0, st, pdl != 0, ids, npair, ls_ptr, ls_val, lmax, cnt, lst, act, nact, local_ids, woff, grp, nslots);\n}\n\ntemplate <int MODE, int NT, int NMAX, int NW, int RING, int DBG>\nstatic void launch(const void* W, const void* S, const float* GS, const void* X, void* OUT, const int* cnt,\n                   const int* lst, const int* act, const int* nact, const int* woff, int maxslots, int lmax, int pdl, cudaStream_t st) {\n  dim3 grid((MODE == 0 ? 512 : 256) / NT, (unsigned)maxslots);\n  if (g_persist) { const unsigned cap = 132u * (unsigned)g_persist; const unsigned tot = grid.x * grid.y; grid = dim3(tot < cap ? tot : cap, 1); }\n  int dyn = NW * NT * 6 * 512 > 49152 ? 49152 : NW * NT * 6 * 512;\n  if (NT == 8) dyn = NW * 3 * 8 * 512 + 1024;\n  if (RING == 3) { const int d3 = NW * 4 * 4 * 512; dyn = d3 > dyn ? d3 : dyn; if (NW * NT * 4 * 512 > dyn) dyn = NW * NT * 4 * 512; }\n  if (RING == 2) {\n    int per = ring2_stages<NT, 1>() * (NT + 2);\n    if (NMAX >= 2) { const int p2 = ring2_stages<2, 2>() * (2 + 4); per = per > p2 ? per : p2; }\n    if (NMAX >= 4) { const int p4 = ring2_stages<2, 4>() * (2 + 8); per = per > p4 ? per : p4; }\n    dyn = NW * per * 512;\n  }\n  cudaFuncSetAttribute(wm_gemm<MODE, NT, NMAX, NW, RING, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);\n  launch_k(wm_gemm<MODE, NT, NMAX, NW, RING, DBG>, grid, dim3(NW * 32), (size_t)dyn, st, pdl != 0,\n           reinterpret_cast<const uint4*>(W), reinterpret_cast<const uint2*>(S), GS, reinterpret_cast<const bf16*>(X),\n           reinterpret_cast<bf16*>(OUT), cnt, lst, act, nact, woff, lmax, g_persist);\n}\n\n// cfg bits: nmax (1,2,4) | nt (2,4) | nw (4,8) | ring (0/1)\nextern "C" int wm_gemm_c(int mode, int nmax, int nt, int nw, int ring, int dbg, const void* W, const void* S, const float* GS,\n                         const void* X, void* OUT, const int* cnt, const int* lst, const int* act, const int* nact,\n                         const int* woff, int maxslots, int lmax, cudaStream_t st) {\n  const int pdl = g_pdl;\n#define WM_RS(M, T, N, S_) \\\n  if (mode == M && ring == 4 && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }\n  WM_RS(0, 4, 4, 4) WM_RS(1, 4, 4, 4) WM_RS(0, 2, 8, 4) WM_RS(1, 2, 8, 4) WM_RS(0, 4, 8, 4) WM_RS(1, 4, 8, 4)\n#undef WM_RS\n\n#define WM_D(M, T, N, W_, R)                                                                             \\\n  if (mode == M && nt == T && nmax == N && nw == W_ && ring == R && dbg == 0) {                           \\\n    launch<M, T, N, W_, R, 0>(W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax, pdl, st);          \\\n    return 0;                                                                                             \\\n  }\n#define WM_ALL(T, N, W_, R) WM_D(0, T, N, W_, R) WM_D(1, T, N, W_, R)\n  WM_ALL(2, 1, 4, 0) WM_ALL(2, 2, 4, 0) WM_ALL(4, 4, 4, 1)\n#undef WM_ALL\n#undef WM_D\n  return 1;\n}\n\n\n// bit sources of the four code bits (m0, m1, m2, sign) for register j = rh + 2 p, low half (the high half is +16)\n__constant__ int c_src[4][4] = {{6, 7, 8, 15}, {9, 10, 11, 18}, {12, 13, 14, 21}, {4, 0, 1, 3}};\n\n__device__ __forceinline__ int wm_pos(int j, int bit, int h) { return (c_src[j][bit] + 16 * h) & 31; }\n\n// e4m3 scale -> b\' with bf16 bits of S * 2^118 == (b\' << 4) + 0x7600 (clamped at the smallest e4m3 subnormal)\n__device__ __forceinline__ uint8_t s2bp(uint8_t s8) {\n  const int sign_exp = s8 & 0x7F;                       // scales are non-negative\n  float f;\n  {\n    const int e = (sign_exp >> 3) & 15, m = sign_exp & 7;\n    f = e == 0 ? (m * 0.001953125f) : (1.0f + m * 0.125f) * __int_as_float((e - 7 + 127) << 23);\n  }\n  f *= __int_as_float((118 + 127) << 23);               // 2^118\n  int bits = __float_as_uint(f) >> 16;\n  if (bits < 0x7600) bits = 0x7600;\n  return (uint8_t)((bits - 0x7600) >> 4);\n}\n\n// raw: uint8 [E, R, K/2] (low nibble = even k), sf: e4m3 [E, R, K/16]  ->  words int32 [E, RTn, KCn, 32, 4], scales uint8 [E, RTn, KCn/4, 32, 8]\n__global__ void raw_to_wm_w(const uint8_t* __restrict__ raw, uint32_t* __restrict__ out, int R, int K) {\n  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  const int RTn = R / 16, KCn = K / 64;\n  const long total = (long)gridDim.y * RTn * KCn * 128;\n  if (idx >= total) return;\n  const int ks = idx & 3, lane = (idx >> 2) & 31;\n  long r = idx >> 7;\n  const int kc = r % KCn; r /= KCn;\n  const int rt = r % RTn;\n  const int e = blockIdx.y;\n  const int g = lane >> 2, t = lane & 3;\n  uint32_t word = 0;\n#pragma unroll\n  for (int rh = 0; rh < 2; rh++)\n#pragma unroll\n    for (int p = 0; p < 2; p++)\n#pragma unroll\n      for (int h = 0; h < 2; h++) {\n        const int row = rt * 16 + rh * 8 + g;\n        const int k = kc * 64 + t * 16 + ks * 4 + p * 2 + h;\n        const uint8_t byte = raw[((size_t)e * R + row) * (K / 2) + (k >> 1)];\n        const uint32_t code = (k & 1) ? (byte >> 4) : (byte & 15);\n        const int j = rh + 2 * p;\n#pragma unroll\n        for (int bit = 0; bit < 4; bit++) word |= ((code >> bit) & 1u) << wm_pos(j, bit, h);\n      }\n  out[(((size_t)e * RTn + rt) * KCn + kc) * 128 + lane * 4 + ks] = word;\n}\n\n__global__ void raw_to_wm_s(const uint8_t* __restrict__ sf, uint8_t* __restrict__ out, int R, int K) {\n  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  const int RTn = R / 16, KCn = K / 64, KGn = K / 16;\n  const long total = (long)gridDim.y * RTn * (KCn / 4) * 256;\n  if (idx >= total) return;\n  const int b = idx & 7, lane = (idx >> 3) & 31;\n  long r = idx >> 8;\n  const int kcg = r % (KCn / 4); r /= (KCn / 4);\n  const int rt = r % RTn;\n  const int e = blockIdx.y;\n  const int rh = b >> 2, c = b & 3, g = lane >> 2, t = lane & 3;\n  const int row = rt * 16 + rh * 8 + g, grp = kcg * 16 + c * 4 + t;\n  out[(size_t)e * RTn * (KCn / 4) * 256 + ((((size_t)rt * (KCn / 4) + kcg) * 32 + lane) * 8 + b)] = s2bp(sf[((size_t)e * R + row) * KGn + grp]);\n}\n\n// wm -> Marlin: words int32 [E, K/16, N/64*128], scales uint8 [E, K/16, N]\n__global__ void wm_to_marlin_w(const uint32_t* __restrict__ wm, uint32_t* __restrict__ out, int N, int K) {\n  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;      // output word index within the expert\n  const long per = (long)(K / 16) * (N / 64) * 128;\n  if (idx >= per) return;\n  const int RTn = N / 16, KCn = K / 64;\n  const int e = blockIdx.y;\n  const int wq = idx & 3, lane = (idx >> 2) & 31;\n  long r = idx >> 7;\n  const int nt = r % (N / 64); r /= (N / 64);\n  const int kt = (int)r;\n  const int g = lane >> 2, tq = lane & 3;\n  uint32_t res = 0;\n#pragma unroll\n  for (int i = 0; i < 8; i++) {\n    const int kl = i >> 2, hn = (i >> 1) & 1, kh = i & 1;\n    const int n = nt * 64 + wq * 16 + hn * 8 + g;\n    const int k = kt * 16 + kh * 8 + tq * 2 + kl;\n    const int rt = n >> 4, rh = (n >> 3) & 1, gg = n & 7;\n    const int kc = k >> 6, rr = k & 63;\n    const int t = rr >> 4, ks = (rr & 15) >> 2, p = (rr & 3) >> 1, h = rr & 1;\n    const uint32_t w = wm[(((size_t)e * RTn + rt) * KCn + kc) * 128 + (gg * 4 + t) * 4 + ks];\n    const int j = rh + 2 * p;\n    uint32_t code = 0;\n#pragma unroll\n    for (int bit = 0; bit < 4; bit++) code |= ((w >> wm_pos(j, bit, h)) & 1u) << bit;\n    res |= code << (4 * i);\n  }\n  out[(size_t)e * per + idx] = res;\n}\n\n__global__ void wm_to_marlin_s(const uint8_t* __restrict__ wms, uint8_t* __restrict__ out, int N, int K) {\n  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;      // (kg, pos) within the expert\n  const long per = (long)(K / 16) * N;\n  if (idx >= per) return;\n  const int RTn = N / 16, KCn = K / 64;\n  const int e = blockIdx.y;\n  const int kg = idx / N, pos = idx % N;\n  const int blk = pos >> 6, p = pos & 63;\n  const int p0 = (p & ~3) | ((p & 3) == 1 ? 2 : (p & 3) == 2 ? 1 : (p & 3));   // swap of the middle pair of each 4\n  const int nl = (p0 >> 3) + 8 * (p0 & 7);\n  const int n = blk * 64 + nl;\n  const int rt = n >> 4, rh = (n >> 3) & 1, g = n & 7;\n  const int kcg = kg >> 4, c = (kg >> 2) & 3, t = kg & 3;\n  const uint8_t bp = wms[(size_t)e * RTn * (KCn / 4) * 256 + ((((size_t)rt * (KCn / 4) + kcg) * 32 + (g * 4 + t)) * 8 + rh * 4 + c)];\n  out[(size_t)e * per + idx] = (uint8_t)(bp + 104);\n}\n\nextern "C" void conv_raw_to_wm(const void* raw_w, const void* sf, void* out_w, void* out_s, int E, int R, int K, cudaStream_t st) {\n  {\n    const long total = (long)(R / 16) * (K / 64) * 128;\n    dim3 grid((total + 255) / 256, E);\n    raw_to_wm_w<<<grid, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(raw_w), reinterpret_cast<uint32_t*>(out_w), R, K);\n  }\n  {\n    const long total = (long)(R / 16) * (K / 256) * 256;\n    dim3 grid((total + 255) / 256, E);\n    raw_to_wm_s<<<grid, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(sf), reinterpret_cast<uint8_t*>(out_s), R, K);\n  }\n}\n\nextern "C" void conv_wm_to_marlin(const void* wm_w, const void* wm_s, void* out_w, void* out_s, int E, int N, int K, cudaStream_t st) {\n  {\n    const long per = (long)(K / 16) * (N / 64) * 128;\n    dim3 grid((per + 255) / 256, E);\n    wm_to_marlin_w<<<grid, 256, 0, st>>>(reinterpret_cast<const uint32_t*>(wm_w), reinterpret_cast<uint32_t*>(out_w), N, K);\n  }\n  {\n    const long per = (long)(K / 16) * N;\n    dim3 grid((per + 255) / 256, E);\n    wm_to_marlin_s<<<grid, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(wm_s), reinterpret_cast<uint8_t*>(out_s), N, K);\n  }\n}\n\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\nextern "C" void wm_route_c(const int*, int, const int*, int, int, int*, int*, int*, int*, int*, int*, int, int, cudaStream_t);\nextern "C" void wm_set_pdl(int);\nvoid wm_set_pdl_t(int64_t v) { wm_set_pdl((int)v); }\nextern "C" void wm_set_persist(int);\nvoid wm_set_persist_t(int64_t v) { wm_set_persist((int)v); }\nextern "C" void wm_finish_c(const void*, const float*, const int*, const void*, void*, int, float, cudaStream_t);\nvoid wm_finish(torch::Tensor y, torch::Tensor tw, torch::Tensor lids, torch::Tensor nw, torch::Tensor out, double eps) {\n  wm_finish_c(y.data_ptr(), tw.data_ptr<float>(), lids.data_ptr<int>(), nw.data_ptr(), out.data_ptr(), (int)out.size(0), (float)eps,\n              at::cuda::getCurrentCUDAStream());\n}\nextern "C" int wm_gemm_c(int, int, int, int, int, int, const void*, const void*, const float*, const void*, void*, const int*,\n                         const int*, const int*, const int*, const int*, int, int, cudaStream_t);\n\nvoid wm_route(torch::Tensor ids, torch::Tensor ls, int64_t ls_val, int64_t lmax, torch::Tensor cnt, torch::Tensor lst,\n              torch::Tensor act, torch::Tensor nact, torch::Tensor local_ids, torch::Tensor woff, int64_t grp) {\n  wm_route_c(ids.data_ptr<int>(), (int)ids.numel(), ls.numel() ? ls.data_ptr<int>() : nullptr, (int)ls_val, (int)lmax, cnt.data_ptr<int>(),\n             lst.data_ptr<int>(), act.data_ptr<int>(), nact.data_ptr<int>(), local_ids.data_ptr<int>(), woff.data_ptr<int>(), (int)grp,\n             (int)(act.numel() / 4), at::cuda::getCurrentCUDAStream());\n}\n\nvoid wm_gemm_run(int64_t mode, int64_t nmax, int64_t nt, int64_t nw, int64_t ring, int64_t dbg, torch::Tensor W, torch::Tensor S, torch::Tensor GS, torch::Tensor X,\n                 torch::Tensor OUT, torch::Tensor cnt, torch::Tensor lst, torch::Tensor act, torch::Tensor nact,\n                 torch::Tensor woff, int64_t maxslots, int64_t lmax) {\n  int r = wm_gemm_c((int)mode, (int)nmax, (int)nt, (int)nw, (int)ring, (int)dbg, W.data_ptr(), S.data_ptr(), GS.data_ptr<float>(), X.data_ptr(),\n                    OUT.data_ptr(), cnt.data_ptr<int>(), lst.data_ptr<int>(), act.data_ptr<int>(),\n                    nact.data_ptr<int>(), woff.data_ptr<int>(), (int)maxslots, (int)lmax, at::cuda::getCurrentCUDAStream());\n  TORCH_CHECK(r == 0, "unsupported config");\n}\n\nextern "C" void conv_raw_to_wm(const void*, const void*, void*, void*, int, int, int, cudaStream_t);\nextern "C" void conv_wm_to_marlin(const void*, const void*, void*, void*, int, int, int, cudaStream_t);\nvoid conv_raw_to_wm_t(torch::Tensor raw, torch::Tensor sf, torch::Tensor out_w, torch::Tensor out_s) {\n  conv_raw_to_wm(raw.data_ptr(), sf.data_ptr(), out_w.data_ptr(), out_s.data_ptr(), (int)raw.size(0), (int)raw.size(1), (int)raw.size(2) * 2,\n                 at::cuda::getCurrentCUDAStream());\n}\nvoid conv_wm_to_marlin_t(torch::Tensor wm_w, torch::Tensor wm_s, torch::Tensor out_w, torch::Tensor out_s, int64_t N, int64_t K) {\n  conv_wm_to_marlin(wm_w.data_ptr(), wm_s.data_ptr(), out_w.data_ptr(), out_s.data_ptr(), (int)wm_w.size(0), (int)N, (int)K,\n                    at::cuda::getCurrentCUDAStream());\n}\n\'\'\'\n_CPP_SRC = r\'\'\'\nvoid wm_set_pdl_t(int64_t v);\nvoid wm_set_persist_t(int64_t v);\nvoid wm_finish(torch::Tensor y, torch::Tensor tw, torch::Tensor lids, torch::Tensor nw, torch::Tensor out, double eps);\nvoid wm_route(torch::Tensor ids, torch::Tensor ls, int64_t ls_val, int64_t lmax, torch::Tensor cnt, torch::Tensor lst,\n              torch::Tensor act, torch::Tensor nact, torch::Tensor local_ids, torch::Tensor woff, int64_t grp);\nvoid wm_gemm_run(int64_t mode, int64_t nmax, int64_t nt, int64_t nw, int64_t ring, int64_t dbg, torch::Tensor W, torch::Tensor S, torch::Tensor GS, torch::Tensor X,\n                 torch::Tensor OUT, torch::Tensor cnt, torch::Tensor lst, torch::Tensor act, torch::Tensor nact,\n                 torch::Tensor woff, int64_t maxslots, int64_t lmax);\n\nvoid conv_raw_to_wm_t(torch::Tensor raw, torch::Tensor sf, torch::Tensor out_w, torch::Tensor out_s);\nvoid conv_wm_to_marlin_t(torch::Tensor wm_w, torch::Tensor wm_s, torch::Tensor out_w, torch::Tensor out_s, int64_t N, int64_t K);\n\'\'\'\n\n_module = None\n_module_lock = threading.Lock()\n_scratch = None\n_empty_i32 = None\n\n\ndef _get_module():\n    global _module\n    if _module is None:\n        with _module_lock:\n            if _module is None:\n                from torch.utils.cpp_extension import load_inline\n\n                mod = load_inline(\n                    "muse_wm_experts_v1",\n                    _CPP_SRC,\n                    cuda_sources=_CUDA_SRC,\n                    functions=["wm_route", "wm_gemm_run", "wm_set_pdl_t", "conv_raw_to_wm_t", "conv_wm_to_marlin_t"],\n                    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90,code=sm_90"],\n                    verbose=False,\n                )\n                mod.wm_set_pdl_t(1)\n                _module = mod\n    return _module\n\n\n# (max tokens) -> (tiles per CTA, warps per CTA, ring variant, 8-token groups per work item); chosen on the H200 for the kernel benchmark (batch sizes 1..256)\n_CFG = [(8, (2, 4, 0, 1)), (16, (2, 4, 0, 2)), (32, (4, 4, 1, 4)), (10**9, (4, 4, 4, 4))]\n\n\ndef _cfg(m: int):\n    for bound, cfg in _CFG:\n        if m <= bound:\n            return cfg\n    return _CFG[-1][1]\n\n\ndef supports(layer) -> bool:\n    """The wm path is written for this model\'s expert shape under 8-way expert parallelism on SM90."""\n    return (\n        torch.cuda.is_available()\n        and torch.cuda.get_device_capability()[0] == _SM_MAJOR\n        and layer.num_local_experts == _NUM_LOCAL\n        and layer.global_num_experts == 256\n        and layer.hidden_size == _HIDDEN\n        and layer.intermediate_size_per_partition == _INTER\n        and layer.top_k == _TOP_K\n    )\n\n\ndef _process_global_scale(gs: torch.Tensor) -> torch.Tensor:\n    """gs * 2^119 in bf16, as nvfp4_marlin_process_global_scale does for bfloat16 activations."""\n    return (gs.to(torch.bfloat16) * (2.0 ** 119)).contiguous()\n\n\nclass WmExperts:\n    """Weight-major copies of one layer\'s experts plus the small tensors Marlin needs to be fed on demand."""\n\n    def __init__(self, w13, s13, w2, s2, gs13_marlin, gs2_marlin):\n        self.w13, self.s13, self.w2, self.s2 = w13, s13, w2, s2\n        self.gs13_marlin, self.gs2_marlin = gs13_marlin, gs2_marlin    # bf16 [32], as Marlin expects them\n        self.gs13 = (gs13_marlin.float() * (2.0 ** -119)).contiguous()  # exact power-of-two rescale back to the real scale\n        self.gs2 = (gs2_marlin.float() * (2.0 ** -119)).contiguous()\n\n\ndef build_wm_experts(w13_raw, w13_sf, w13_gs, w2_raw, w2_sf, w2_gs) -> WmExperts:\n    """raw HF-layout NVFP4 tensors of one layer\'s local experts -> weight-major layout (does not free the inputs)."""\n    mod = _get_module()\n    dev = w13_raw.device\n    E, R13, K2 = w13_raw.shape\n    out13 = torch.empty((E, R13 // 16, K2 * 2 // 64, 32, 4), dtype=torch.int32, device=dev)\n    s13 = torch.empty((E, R13 // 16, K2 * 2 // 256, 32, 8), dtype=torch.uint8, device=dev)\n    mod.conv_raw_to_wm_t(w13_raw.contiguous(), w13_sf.contiguous().view(torch.uint8), out13, s13)\n    E2, R2, K22 = w2_raw.shape\n    out2 = torch.empty((E2, R2 // 16, K22 * 2 // 64, 32, 4), dtype=torch.int32, device=dev)\n    s2 = torch.empty((E2, R2 // 16, K22 * 2 // 256, 32, 8), dtype=torch.uint8, device=dev)\n    mod.conv_raw_to_wm_t(w2_raw.contiguous(), w2_sf.contiguous().view(torch.uint8), out2, s2)\n    return WmExperts(out13, s13, out2, s2, _process_global_scale(w13_gs), _process_global_scale(w2_gs))\n\n\nclass _Scratch:\n    """One Marlin-layout copy of an expert bank, shared by every layer (layers run one after the other on a stream)."""\n\n    def __init__(self, device):\n        k16 = _HIDDEN // 16\n        self.w13 = torch.empty((_NUM_LOCAL, k16, 2 * _INTER // 64 * 128), dtype=torch.int32, device=device)\n        self.s13 = torch.empty((_NUM_LOCAL, k16, 2 * _INTER), dtype=torch.float8_e4m3fn, device=device)\n        self.w2 = torch.empty((_NUM_LOCAL, _INTER // 16, _HIDDEN // 64 * 128), dtype=torch.int32, device=device)\n        self.s2 = torch.empty((_NUM_LOCAL, _INTER // 16, _HIDDEN), dtype=torch.float8_e4m3fn, device=device)\n\n\ndef reserve_scratch(device) -> None:\n    global _scratch\n    if _scratch is None:\n        _scratch = _Scratch(device)\n\n\ndef marlin_view(wm: WmExperts):\n    """Convert one layer\'s weights into the shared Marlin scratch; returns (w13, s13, gs13, w2, s2, gs2)."""\n    mod = _get_module()\n    reserve_scratch(wm.w13.device)\n    sc = _scratch\n    mod.conv_wm_to_marlin_t(wm.w13, wm.s13, sc.w13, sc.s13.view(torch.uint8), 2 * _INTER, _HIDDEN)\n    mod.conv_wm_to_marlin_t(wm.w2, wm.s2, sc.w2, sc.s2.view(torch.uint8), _HIDDEN, _INTER)\n    return sc.w13, sc.s13, wm.gs13_marlin, sc.w2, sc.s2, wm.gs2_marlin\n\n\ndef expert_outputs(wm: WmExperts, hidden_states: torch.Tensor, topk_ids: torch.Tensor, local_start: int):\n    """Routed experts for up to WM_MAX_TOKENS tokens.  Returns (y [m, top_k, H] bf16 with unspecified rows for routes\n    outside this GPU, local expert id per route (-1 outside) [m, top_k] int32)."""\n    global _empty_i32\n    mod = _get_module()\n    x = hidden_states.contiguous()\n    m = x.shape[0]\n    dev = x.device\n    if _empty_i32 is None or _empty_i32.device != dev:\n        _empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)\n    nt, nw, ring, nch = _cfg(m)\n    lmax = (m + 7) // 8 * 8\n    ids = topk_ids.to(torch.int32).contiguous()\n    cnt = torch.empty(_NUM_LOCAL, dtype=torch.int32, device=dev)\n    lst = torch.empty(_NUM_LOCAL * lmax, dtype=torch.int32, device=dev)\n    act = torch.empty(4 * (_NUM_LOCAL + (m * _TOP_K) // (nch * 8) + 8), dtype=torch.int32, device=dev)\n    woff = torch.empty(128, dtype=torch.int32, device=dev)\n    nact = torch.empty(1, dtype=torch.int32, device=dev)\n    lids = torch.empty(m * _TOP_K, dtype=torch.int32, device=dev)\n    mod.wm_route(ids, _empty_i32, local_start, lmax, cnt, lst, act, nact, lids, woff, nch * 8)\n    maxslots = min(m * _TOP_K, _NUM_LOCAL + (m * _TOP_K) // (nch * 8))\n    inter = torch.empty((m * _TOP_K, _INTER), dtype=torch.bfloat16, device=dev)\n    y = torch.empty((m * _TOP_K, _HIDDEN), dtype=torch.bfloat16, device=dev)\n    mod.wm_gemm_run(0, nch, nt, nw, ring, 0, wm.w13, wm.s13, wm.gs13, x, inter, cnt, lst, act, nact, woff, maxslots, lmax)\n    mod.wm_gemm_run(1, nch, nt, nw, ring, 0, wm.w2, wm.s2, wm.gs2, inter, y, cnt, lst, act, nact, woff, maxslots, lmax)\n    return y.view(m, _TOP_K, _HIDDEN), lids.view(m, _TOP_K)\n'


def _conv_lab(state, weights):
    import types
    wmx = types.ModuleType("wmx_lab")
    exec(compile(_WMX_SRC, "wmx_lab", "exec"), wmx.__dict__)
    dev = weights["w13"].device
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    t0 = time.time()
    wm = wmx.build_wm_experts(weights["w13"][:32].contiguous(), weights["w13_sf"][:32].contiguous(), weights["w13_gs"][:32].contiguous(),
                              weights["w2"][:32].contiguous(), weights["w2_sf"][:32].contiguous(), weights["w2_gs"][:32].contiguous())
    torch.cuda.synchronize()
    print("LABV module compile+build %.1fs" % (time.time() - t0), flush=True)
    wmx.reserve_scratch(dev)
    # exactness against the real Marlin tensors from the baseline repack
    bank = state["banks"][(0, 32)]
    w13, s13, g13, w2, s2, g2 = wmx.marlin_view(wm)
    torch.cuda.synchronize()
    print("LABV marlin w13 equal", bool(torch.equal(w13, bank["w13"])), "w2 equal", bool(torch.equal(w2, bank["w2"])),
          "s13 equal", bool(torch.equal(s13.view(torch.uint8), bank["w13_s"].view(torch.uint8))), "s2 equal", bool(torch.equal(s2.view(torch.uint8), bank["w2_s"].view(torch.uint8))),
          "gs13 equal", bool(torch.equal(g13, bank["w13_gs"])), "gs2 equal", bool(torch.equal(g2, bank["w2_gs"])), flush=True)
    t = _cv_time(lambda: wmx.marlin_view(wm), flush)
    print("LABV marlin_view (both matrices) us=%.1f" % t, flush=True)
    # wm path vs (conversion + stock Marlin) at growing batch sizes
    for m in (512, 1024, 2048, 4096, 8192, 16384):
        x, ids, tw = _lab_in(m, dev, 100 + m)
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
                "config": {"bs": m, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0, "local_count": 32, "top_k": 8, "sharding": "ep8"}}
        def wm_path():
            y, lids = wmx.expert_outputs(wm, x, ids, 0)
            return _finish_experts(y, tw, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=lids)
        ref = _marlin_kernel_fn(data, state).float() if m <= 8192 else None
        t_marlin = _cv_time(lambda: _marlin_kernel_fn(data, state), flush) if m <= 8192 else -1
        row = []
        saved = list(wmx._CFG)
        for name, cfg in (("t4n4", (4, 4, 4, 4)), ("t2n8", (2, 4, 4, 8)), ("t4n8", (4, 4, 4, 8))):
            wmx._CFG[:] = [(10**9, cfg)]
            try:
                o2 = wm_path().float()
                e2 = ((o2 - ref).norm() / ref.norm()).item() if m <= 8192 else 0.0
                row.append("%s=%.1f(%.0e)" % (name, _cv_time(wm_path, flush), e2))
            except Exception as ex:
                row.append("%s=ERR" % name)
        wmx._CFG[:] = saved
        print("LABV m=%d marlin=%.1f %s" % (m, t_marlin, " ".join(row)), flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        wmx.marlin_view(wm)
        torch.cuda.synchronize()
    for ev in prof.key_averages():
        if "wm_to_marlin" in ev.key or "raw_to_wm" in ev.key:
            print("LABV kernel", ev.key[:60], "us=%.1f" % ev.device_time_total, flush=True)


def _lab_in(m, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, 4096), generator=g, device=device).to(torch.bfloat16)
    scores = torch.rand((m, 256), generator=g, device=device)
    z = torch.randn((256,), generator=g, device=device)
    u = torch.rand((256,), generator=g, device=device).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return x, ids.to(torch.int32), (routed.values / routed.values.sum(-1, keepdim=True)).float()


def _cv_time(fn, flush, reps=8):
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


import time


def prepare(weights):
    state = _prepare_marlin(weights)
    try:
        _conv_lab(state, weights)
    except Exception:
        import traceback
        print("LABV FAILED", traceback.format_exc(), flush=True)
    return state


def kernel_fn(data, state):
    return _marlin_kernel_fn(data, state)
