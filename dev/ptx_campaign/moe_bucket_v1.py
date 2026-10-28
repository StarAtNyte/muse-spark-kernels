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


# ---- weight-major NVFP4 path (CUDA) ----
_WM_CUDA = '\n#include <cuda_bf16.h>\n#include <cuda_runtime.h>\n#include <stdint.h>\n\ntypedef __nv_bfloat16 bf16;\ntypedef __nv_bfloat162 bf162;\n\n#define HD 4096\n#define HD_ 4096\n#define ID 4096\n#define KC 64            // k64 chunks along K = 4096\n#define KCG 16           // groups of 4 chunks (scale words)\n#define RT0 512          // 16-row tiles of w13 (gate 256 + up 256)\n#define RT1 256          // 16-row tiles of w2\n\nconstexpr uint32_t MASK = 0x81C081C0u;\n\n// Weight word -> bf16x2 holding the E2M1 value times 2^-126 (denormal-linear encoding).\ntemplate <int J>\n__device__ __forceinline__ uint32_t dec(uint32_t w) {\n  if constexpr (J == 0) {\n    return w & MASK;\n  } else if constexpr (J < 3) {\n    return __funnelshift_r(w, w, 3 * J) & MASK;\n  } else {\n    return ((w & 0x001B001Bu) * 0x1084u) & MASK;  // sources m0@4 m1@0 m2@1 sign@3 (+16 for the high half)\n  }\n}\n\n__device__ __forceinline__ uint32_t scale2(uint32_t q, int c) {\n  // byte c of q holds b\'; scale = bf16 bits (b\'<<4) + 0x7600 = S * 2^118\n  uint32_t bits = ((q >> (8 * c)) & 0xFFu) << 4;\n  return (bits + 0x7600u) * 0x10001u;\n}\n\n__device__ __forceinline__ uint32_t hmul(uint32_t a, uint32_t b) {\n  uint32_t r;\n  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));\n  return r;\n}\n\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }\n\n__device__ __forceinline__ void pdl_wait() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void pdl_trigger() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n#endif\n}\n\nstatic int g_pdl = 0;\nstatic int g_persist = 0;\nextern "C" void wm_set_persist(int v) { g_persist = v; }\nextern "C" void wm_set_pdl(int v) { g_pdl = v; }\n\ntemplate <typename... KArgs, typename... Args>\nstatic void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, bool pdl,\n                     Args... args) {\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = grid;\n  cfg.blockDim = block;\n  cfg.dynamicSmemBytes = smem;\n  cfg.stream = st;\n  cudaLaunchAttribute attr[1];\n  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n  attr[0].val.programmaticStreamSerializationAllowed = 1;\n  cfg.attrs = attr;\n  cfg.numAttrs = pdl ? 1 : 0;\n  cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);\n}\n\n__global__ void route_kernel(const int* __restrict__ ids, int npair, const int* __restrict__ ls_ptr, int ls_val, int lmax,\n                             int* __restrict__ cnt, int* __restrict__ lst, int* __restrict__ act,\n                             int* __restrict__ nact, int* __restrict__ local_ids, int* __restrict__ woff, int grp, int nslots) {\n  __shared__ int s_cnt[32];\n  pdl_wait();\n  const int ls = ls_ptr ? *ls_ptr : ls_val;\n  pdl_trigger();\n  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;\n  for (int i = threadIdx.x; i < nslots; i += blockDim.x) reinterpret_cast<int4*>(act)[i] = make_int4(0, 0, 0, 0);  // work items {expert, offset, tokens, 0}\n  __syncthreads();\n  for (int p = threadIdx.x; p < npair; p += blockDim.x) {\n    int e = (ids[p] - ls) & 255;  // experts are ordered mod 256 starting at this GPU\'s first expert\n    bool ok = e < 32;\n    local_ids[p] = ok ? e : -1;\n    if (ok) {\n      int pos = atomicAdd(&s_cnt[e], 1);\n      lst[e * lmax + pos] = p;\n    }\n  }\n  __syncthreads();\n  if (threadIdx.x < 32) {\n    int lane = threadIdx.x;\n    int c = s_cnt[lane];\n    cnt[lane] = c;\n    // work items: one per group of `grp` tokens of an expert, so a hot expert spreads over several CTAs\n    const int groups = (c + grp - 1) / grp;\n    // longest-processing-time-first: experts with the most tokens (largest NCH) get the earliest work items\n    int start = 0, total = 0;\n#pragma unroll\n    for (int j = 0; j < 32; j++) {\n      const int cj = __shfl_sync(0xffffffffu, c, j), gj = __shfl_sync(0xffffffffu, groups, j);\n      if (cj > c || (cj == c && j < lane)) start += gj;\n      total += gj;\n    }\n    for (int j = 0; j < groups; j++)\n      reinterpret_cast<int4*>(act)[start + j] = make_int4(lane, j * grp, min(c - j * grp, grp), 1);\n    if (lane == 31) *nact = total;\n  }\n  // The extended work buffer is used only by the bucketed caller. Existing\n  // callers retain their original 128-item allocation and one-element nact.\n  if(nslots>=512){\n    __syncthreads();\n    if(threadIdx.x<32){\n      int lane=threadIdx.x,total=*nact,carry[3]={0,0,0};\n#pragma unroll\n      for(int q=0;q<4;q++){\n        int index=q*32+lane;\n        int4 item=index<total?__ldcg(reinterpret_cast<const int4*>(act)+index):make_int4(0,0,0,0);\n        int tier=item.z<=8?0:(item.z<=16?1:2);\n#pragma unroll\n        for(int b=0;b<3;b++){\n          unsigned mask=__ballot_sync(0xffffffff,item.z>0&&tier==b);\n          int rank=__popc(mask&((1u<<lane)-1));\n          if(item.z>0&&tier==b)reinterpret_cast<int4*>(act+512*(b+1))[carry[b]+rank]=item;\n          carry[b]+=__popc(mask);\n        }\n      }\n      if(lane==0)for(int b=0;b<3;b++)nact[b+1]=carry[b];\n    }\n  }\n\n}\n\n__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),\n               "l"(gsrc));\n}\n__device__ __forceinline__ void cp16z(void* sdst, const void* gsrc, uint32_t nbytes) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),\n               "l"(gsrc), "r"(nbytes));\n}\ntemplate <int NTB_, int NCH_>\n__host__ __device__ constexpr int ring2_stages() {\n  return (14336 / (512 * (NTB_ + 2 * NCH_))) > 6 ? 6 : ((14336 / (512 * (NTB_ + 2 * NCH_))) < 3 ? 3 : (14336 / (512 * (NTB_ + 2 * NCH_))));\n}\n__device__ __forceinline__ void cp8(void* sdst, const void* gsrc) {\n  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(sdst));\n  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\\n" ::"r"(d), "l"(gsrc));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N>\n__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n\ntemplate <int NTB, int NCH, int NW>\n__device__ __forceinline__ void wm_epilogue(float (&acc)[NTB][NCH][4], float* red, const float gs, const int* lst,\n                                            int ntok, int tbase, int mode, int nt, bf16* OUT, const int bx) {\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n#pragma unroll\n  for (int n = 0; n < NTB; n++)\n#pragma unroll\n    for (int c = 0; c < NCH; c++)\n#pragma unroll\n      for (int r = 0; r < 4; r++)\n        red[(((warp * NTB + n) * NCH + c) * 4 + r) * 32 + lane] = acc[n][c][r];\n  __syncthreads();\n  for (int item = threadIdx.x; item < NCH * 4 * 32; item += NW * 32) {\n    const int li = item & 31, r = (item >> 5) & 3, c = item >> 7;\n    const int row = (li >> 2) + 8 * (r >> 1);\n    const int col = c * 8 + 2 * (li & 3) + (r & 1);\n    if (col >= ntok) continue;\n    const int p = __ldcg(lst + col);\n#pragma unroll\n    for (int n = 0; n < NTB; n++) {\n      if (mode == 0 && (n & 1)) continue;\n      float s = 0.f;\n#pragma unroll\n      for (int w = 0; w < NW; w++) s += red[(((w * NTB + n) * NCH + c) * 4 + r) * 32 + li];\n      if (mode == 0) {\n        float s2 = 0.f;\n#pragma unroll\n        for (int w = 0; w < NW; w++) s2 += red[(((w * NTB + n + 1) * NCH + c) * 4 + r) * 32 + li];\n        const float gate = bfr(bfr(s * 256.0f) * gs), up = bfr(bfr(s2 * 256.0f) * gs);\n        const float sil = gate / (1.0f + __expf(-gate));\n        OUT[(size_t)p * ID + (bx * (nt / 2) + ((tbase + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);\n      } else {\n        OUT[(size_t)p * HD + (bx * nt + tbase + n) * 16 + row] = __float2bfloat16(bfr(s * 256.0f) * gs);\n      }\n    }\n  }\n  __syncthreads();\n}\n\n// RING == false: every lane issues its weight loads up front, group by group (best for small problems).\n// RING == true : weights stream through a per-warp cp.async ring in shared memory (lane-private slots).\ntemplate <int MODE, int NT, int NTB, int NCH, int NW, int RING, int S, int DBG>\n__device__ __forceinline__ void wm_body(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq,\n                                        const float* __restrict__ GS, const bf16* __restrict__ X,\n                                        bf16* __restrict__ OUT, const int* __restrict__ lst, int e, int ntok,\n                                        int tbase, uint4* smem, const int bx) {\n  constexpr int RT = MODE == 0 ? RT0 : RT1;\n  constexpr int XW = MODE == 0 ? HD : ID;\n  constexpr int CPW = 64 / NW;  // chunks of k64 per warp\n  constexpr int GPW = CPW / 4;  // scale groups per warp\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  float* red = reinterpret_cast<float*>(smem);\n\n  int tile[NTB];\n#pragma unroll\n  for (int n = 0; n < NTB; n++) {\n    const int nn = tbase + n;\n    if (MODE == 0) tile[n] = (nn & 1) ? RT0 / 2 + bx * (NT / 2) + (nn >> 1) : bx * (NT / 2) + (nn >> 1);\n    else tile[n] = bx * NT + nn;\n  }\n\n  const bf16* xp[NCH];\n#pragma unroll\n  for (int c = 0; c < NCH; c++) {\n    int col = c * 8 + g;\n    xp[c] = nullptr;\n    if (col < ntok) {\n      int p = __ldcg(lst + col);\n      xp[c] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + 16 * t + warp * CPW * 64;\n    }\n  }\n\n  float acc[NTB][NCH][4];\n#pragma unroll\n  for (int n = 0; n < NTB; n++)\n#pragma unroll\n    for (int c = 0; c < NCH; c++)\n#pragma unroll\n      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;\n\n  uint32_t sink = 0;\n  const uint4* wsrc[NTB];\n  const uint2* ssrc[NTB];\n#pragma unroll\n  for (int n = 0; n < NTB; n++) {\n    wsrc[n] = Wq + ((size_t)e * RT + tile[n]) * KC * 32 + warp * CPW * 32 + lane;\n    ssrc[n] = Sq + ((size_t)e * RT + tile[n]) * KCG * 32 + warp * GPW * 32 + lane;\n  }\n\n  if (RING == 0) {\n#pragma unroll\n    for (int q = 0; q < GPW; q++) {\n      uint2 sc[NTB];\n      uint4 wv[NTB][4];\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        sc[n] = __ldg(ssrc[n] + q * 32);\n#pragma unroll\n        for (int c = 0; c < 4; c++) {\n          if constexpr (DBG & 8) wv[n][c] = make_uint4(lane * 7 + c, lane * 13 + q, lane + n, lane * 3);\n          else wv[n][c] = __ldg(wsrc[n] + (q * 4 + c) * 32);\n        }\n      }\n#pragma unroll\n      for (int c = 0; c < 4; c++) {\n        uint4 xa[NCH], xb[NCH];\n#pragma unroll\n        for (int h = 0; h < NCH; h++) {\n          if (xp[h] && !(DBG & 4)) {\n            const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (q * 4 + c) * 64);\n            xa[h] = __ldcg(xv);\n            xb[h] = __ldcg(xv + 1);\n          } else {\n            xa[h] = make_uint4(0, 0, 0, 0);\n            xb[h] = make_uint4(0, 0, 0, 0);\n          }\n        }\n#pragma unroll\n        for (int n = 0; n < NTB; n++) {\n          const uint32_t s0 = scale2(sc[n].x, c);\n          const uint32_t s1 = scale2(sc[n].y, c);\n          const uint32_t ww[4] = {wv[n][c].x, wv[n][c].y, wv[n][c].z, wv[n][c].w};\n#pragma unroll\n          for (int ks = 0; ks < 4; ks++) {\n            const uint32_t w = ww[ks];\n            uint32_t a0, a1, a2, a3;\n            if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }\n            else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }\n#pragma unroll\n            for (int h = 0; h < NCH; h++) {\n              const uint4 xs = ks < 2 ? xa[h] : xb[h];\n              if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);\n              else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);\n            }\n          }\n        }\n      }\n    }\n  } else if (RING == 2) {\n    constexpr int XS = NCH * 2;\n    constexpr int STG = NTB + XS;\n    uint4* ring = smem + warp * (S * STG * 32);\n    auto issue = [&](int c) {\n      if (c < CPW && !(DBG & 8)) {\n#pragma unroll\n        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * STG + n) * 32 + lane], wsrc[n] + c * 32);\n      }\n      if (c < CPW) {\n#pragma unroll\n        for (int h = 0; h < NCH; h++) {\n          const bool ok = xp[h] != nullptr && !(DBG & 4);\n          const bf16* src = ok ? xp[h] + c * 64 : X;\n          cp16z(&ring[((c % S) * STG + NTB + 2 * h) * 32 + lane], src, ok ? 16u : 0u);\n          cp16z(&ring[((c % S) * STG + NTB + 2 * h + 1) * 32 + lane], src + 8, ok ? 16u : 0u);\n        }\n      }\n      cp_commit();\n    };\n#pragma unroll\n    for (int c = 0; c < S - 1; c++) issue(c);\n    uint2 scur[NTB], snxt[NTB];\n#pragma unroll\n    for (int n = 0; n < NTB; n++) {\n      scur[n] = __ldg(ssrc[n]);\n      snxt[n] = GPW > 1 ? __ldg(ssrc[n] + 32) : scur[n];\n    }\n#pragma unroll 1\n    for (int i = 0; i < CPW; i++) {\n      cp_wait<S - 2>();\n      issue(i + S - 1);\n      const int cq = i & 3;\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        const uint4 wv = ring[((i % S) * STG + n) * 32 + lane];\n        const uint32_t s0 = scale2(scur[n].x, cq);\n        const uint32_t s1 = scale2(scur[n].y, cq);\n        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};\n#pragma unroll\n        for (int ks = 0; ks < 4; ks++) {\n          const uint32_t w = ww[ks];\n          uint32_t a0, a1, a2, a3;\n          if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }\n          else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }\n#pragma unroll\n          for (int h = 0; h < NCH; h++) {\n            const uint4 xs = ring[((i % S) * STG + NTB + 2 * h + (ks >> 1)) * 32 + lane];\n            if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);\n            else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);\n          }\n        }\n      }\n      if (cq == 3) {\n#pragma unroll\n        for (int n = 0; n < NTB; n++) {\n          scur[n] = snxt[n];\n          if ((i >> 2) + 2 < GPW) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);\n        }\n      }\n    }\n    cp_wait<0>();\n  } else {\n    uint4* ring = smem + warp * (S * NTB * 32);\n    auto issue = [&](int c) {\n      if (c < CPW && !(DBG & 8)) {\n#pragma unroll\n        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * NTB + n) * 32 + lane], wsrc[n] + c * 32);\n      }\n      cp_commit();\n    };\n#pragma unroll\n    for (int c = 0; c < S - 1; c++) issue(c);\n\n    uint2 scur[NTB], snxt[NTB];\n#pragma unroll\n    for (int n = 0; n < NTB; n++) {\n      scur[n] = __ldg(ssrc[n]);\n      snxt[n] = GPW > 1 ? __ldg(ssrc[n] + 32) : scur[n];\n    }\n    uint4 xcur[NCH][2], xnxt[NCH][2];\n#pragma unroll\n    for (int h = 0; h < NCH; h++) {\n      if (xp[h] && !(DBG & 4)) {\n        const uint4* xv = reinterpret_cast<const uint4*>(xp[h]);\n        xcur[h][0] = __ldcg(xv);\n        xcur[h][1] = __ldcg(xv + 1);\n      } else {\n        xcur[h][0] = make_uint4(0, 0, 0, 0);\n        xcur[h][1] = make_uint4(0, 0, 0, 0);\n      }\n      xnxt[h][0] = xcur[h][0];\n      xnxt[h][1] = xcur[h][1];\n    }\n\n#pragma unroll 1\n    for (int i = 0; i < CPW; i++) {\n      cp_wait<S - 2>();\n      issue(i + S - 1);\n#pragma unroll\n      for (int h = 0; h < NCH; h++) {\n        if (xp[h] && i + 1 < CPW && !(DBG & 4)) {\n          const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (i + 1) * 64);\n          xnxt[h][0] = __ldcg(xv);\n          xnxt[h][1] = __ldcg(xv + 1);\n        }\n      }\n      const int cq = i & 3;\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        const uint4 wv = ring[((i % S) * NTB + n) * 32 + lane];\n        const uint32_t s0 = scale2(scur[n].x, cq);\n        const uint32_t s1 = scale2(scur[n].y, cq);\n        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};\n#pragma unroll\n        for (int ks = 0; ks < 4; ks++) {\n          const uint32_t w = ww[ks];\n          uint32_t a0, a1, a2, a3;\n          if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }\n          else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }\n#pragma unroll\n          for (int h = 0; h < NCH; h++) {\n            const uint4 xs = xcur[h][ks >> 1];\n            if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);\n            else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);\n          }\n        }\n      }\n      if (cq == 3) {\n#pragma unroll\n        for (int n = 0; n < NTB; n++) {\n          scur[n] = snxt[n];\n          if ((i >> 2) + 2 < GPW) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);\n        }\n      }\n#pragma unroll\n      for (int h = 0; h < NCH; h++) {\n        xcur[h][0] = xnxt[h][0];\n        xcur[h][1] = xnxt[h][1];\n      }\n    }\n    cp_wait<0>();\n  }\n\n  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);\n  __syncthreads();  // the ring is dead; the same smem holds the cross-warp reduction\n  wm_epilogue<NTB, NCH, NW>(acc, red, GS[e], lst, ntok, tbase, MODE, NT, OUT, bx);\n}\n\n#ifndef WM_MINB\n#define WM_MINB 4\n#endif\ntemplate <int MODE, int NT, int NMAX, int NW, int RING, int DBG>\n__global__ void __launch_bounds__(NW * 32, (RING == 3 ? 2 : WM_MINB))\nwm_gemm(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,\n        const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ cnt,\n        const int* __restrict__ lst, const int* __restrict__ act, const int* __restrict__ nact_p,\n        const int* __restrict__ woff, int lmax, int persist) {\n  extern __shared__ uint4 smem[];\n  constexpr int TILES = (MODE == 0 ? 512 : 256) / NT;\n  pdl_wait();  // everything below (items, token lists, activations) comes from earlier kernels\n  pdl_trigger();\n  const int nitems = persist ? __ldcg(nact_p) : 1;\n  for (int u = blockIdx.x;; u += gridDim.x) {\n  int slot, bx;\n  if (persist) {\n    if (u >= nitems * TILES) break;\n    slot = u / TILES; bx = u - slot * TILES;\n  } else { slot = blockIdx.y; bx = blockIdx.x; }\n  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + slot);  // {expert, offset, tokens, valid}\n  if (item.z != 0) {\n  const int e = item.x;\n  const int off = item.y;\n  {\n    const int ntok = item.z;\n    const int* l = lst + e * lmax + off;\n    if (ntok <= 8) {\n      wm_body<MODE, NT, NT, 1, NW, RING, (RING == 2 ? ring2_stages<NT, 1>() : (NT == 8 ? 3 : NT == 4 ? 4 : 6)), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, 0, smem, bx);\n    } else if (NMAX >= 2 && ntok <= 16) {\n#pragma unroll 1\n      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))\n        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 2 ? 2 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 2 ? 2 : 1)>() : RING == 3 ? 4 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);\n    } else if (NMAX >= 4) {\n#pragma unroll 1\n      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))\n        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 4 ? 4 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 4 ? 4 : 1)>() : RING == 3 ? 4 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);\n    }\n  }\n  }\n  if (!persist) break;\n  }\n}\n\n\n\n// ---- row-split variant: the four warps of a CTA own different weight rows but the same activations, which are\n// staged once per CTA through shared memory.  No cross-warp reduction; x traffic from L2 drops 4x. ----\ntemplate <int MODE, int NTB, int NCH, int S, int DBG>\n__device__ __forceinline__ void wm_body_rs(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq,\n                                           const float* __restrict__ GS, const bf16* __restrict__ X,\n                                           bf16* __restrict__ OUT, const int* __restrict__ lst, int e, int ntok,\n                                           uint4* smem, const int bx) {\n  constexpr int RT = MODE == 0 ? RT0 : RT1;\n  constexpr int XW = MODE == 0 ? HD : ID;\n  constexpr int NTC = 4 * NTB;  // tiles per CTA\n  constexpr int XSLOTS = NCH * 2 * 32;\n  constexpr int RPT = (XSLOTS + 127) / 128;\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  constexpr int SOFF = 3 * 4 * NTB * 16;     // scale ring: [3][4][NTB][32] uint2, in uint4 units\n  uint2* sring = reinterpret_cast<uint2*>(smem);\n  uint4* wring = smem + SOFF;                // [S][4][NTB][32]\n  uint4* xring = wring + S * 4 * NTB * 32;   // [S][XSLOTS]\n\n  const uint4* wsrc[NTB];\n  const uint2* ssrc[NTB];\n#pragma unroll\n  for (int n = 0; n < NTB; n++) {\n    const int nn = warp * NTB + n;\n    const int tile = MODE == 0 ? ((nn & 1) ? RT0 / 2 + bx * (NTC / 2) + (nn >> 1) : bx * (NTC / 2) + (nn >> 1)) : bx * NTC + nn;\n    wsrc[n] = Wq + ((size_t)e * RT + tile) * KC * 32 + lane;\n    ssrc[n] = Sq + ((size_t)e * RT + tile) * KCG * 32 + lane;\n  }\n  const bf16* xb[RPT];\n  bool xok[RPT];\n#pragma unroll\n  for (int r = 0; r < RPT; r++) {\n    const int sl = tid + 128 * r;\n    xok[r] = false;\n    xb[r] = X;\n    if (sl < XSLOTS) {\n      const int h = sl >> 6, j = (sl >> 5) & 1, L = sl & 31;\n      const int col = h * 8 + (L >> 2);\n      if (col < ntok) {\n        const int p = __ldcg(lst + col);\n        xb[r] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + 16 * (L & 3) + 8 * j;\n        xok[r] = true;\n      }\n    }\n  }\n\n  uint32_t sink = 0;\n  float acc[NTB][NCH][4];\n#pragma unroll\n  for (int n = 0; n < NTB; n++)\n#pragma unroll\n    for (int c = 0; c < NCH; c++)\n#pragma unroll\n      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;\n\n  auto issue_sc = [&](int g) {\n    if (g < KCG && !(DBG & 32)) {\n#pragma unroll\n      for (int n = 0; n < NTB; n++) cp8(&sring[(((g % 3) * 4 + warp) * NTB + n) * 32 + lane], ssrc[n] + g * 32);\n    }\n  };\n  auto issue = [&](int c) {\n    if ((c & 3) == 0) issue_sc((c >> 2) + 1);\n    if (c < KC) {\n#pragma unroll\n      for (int n = 0; n < NTB; n++) { if constexpr (!(DBG & 8)) cp16(&wring[(((c % S) * 4 + warp) * NTB + n) * 32 + lane], wsrc[n] + c * 32); }\n#pragma unroll\n      for (int r = 0; r < RPT; r++)\n        if (tid + 128 * r < XSLOTS && !(DBG & 4)) cp16z(&xring[(c % S) * XSLOTS + tid + 128 * r], xb[r] + c * 64, xok[r] ? 16u : 0u);\n    }\n    cp_commit();\n  };\n  issue_sc(0);\n#pragma unroll\n  for (int c = 0; c < S - 1; c++) issue(c);\n#pragma unroll 1\n  for (int i = 0; i < KC; i++) {\n    cp_wait<S - 2>();\n    __syncthreads();\n    issue(i + S - 1);\n    const int cq = i & 3;\n    uint4 xr[NCH][2];\n#pragma unroll\n    for (int h = 0; h < NCH; h++) {\n      xr[h][0] = xring[(i % S) * XSLOTS + (h * 2) * 32 + lane];\n      xr[h][1] = xring[(i % S) * XSLOTS + (h * 2 + 1) * 32 + lane];\n    }\n#pragma unroll\n    for (int n = 0; n < NTB; n++) {\n      const uint4 wv = wring[(((i % S) * 4 + warp) * NTB + n) * 32 + lane];\n      const uint2 sc = (DBG & 32) ? make_uint2(0x38383838u, 0x38383838u) : sring[((((i >> 2) % 3) * 4 + warp) * NTB + n) * 32 + lane];\n      const uint32_t s0 = scale2(sc.x, cq);\n      const uint32_t s1 = scale2(sc.y, cq);\n      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};\n#pragma unroll\n      for (int ks = 0; ks < 4; ks++) {\n        const uint32_t w = ww[ks];\n        uint32_t a0, a1, a2, a3;\n        if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }\n        else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }\n#pragma unroll\n        for (int h = 0; h < NCH; h++) {\n          const uint4 xs = xr[h][ks >> 1];\n          if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);\n          else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);\n        }\n      }\n    }\n  }\n  cp_wait<0>();\n  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);\n\n  const float gs = GS[e];\n#pragma unroll\n  for (int c = 0; c < NCH; c++)\n#pragma unroll\n    for (int r = 0; r < 4; r++) {\n      const int col = c * 8 + 2 * t + (r & 1);\n      if (col >= ntok || (DBG & 16)) continue;\n      const int p = (DBG & 64) ? col * 7 : __ldcg(lst + col);\n      const int row = g + 8 * (r >> 1);\n#pragma unroll\n      for (int n = 0; n < NTB; n++) {\n        if (MODE == 0) {\n          if (n & 1) continue;\n          const float gate = bfr(bfr(acc[n][c][r] * 256.0f) * gs), up = bfr(bfr(acc[n + 1][c][r] * 256.0f) * gs);\n          const float sil = gate / (1.0f + __expf(-gate));\n          OUT[(size_t)p * ID + (bx * (NTC / 2) + ((warp * NTB + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);\n        } else {\n          OUT[(size_t)p * HD + (bx * NTC + warp * NTB + n) * 16 + row] = __float2bfloat16(bfr(acc[n][c][r] * 256.0f) * gs);\n        }\n      }\n    }\n}\n\ntemplate <int MODE, int NTB, int NMAX, int S, int DBG>\n__global__ void __launch_bounds__(128, (S <= 3 ? 4 : 3))\nwm_gemm_rs(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,\n           const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst,\n           const int* __restrict__ act, int lmax) {\n  extern __shared__ uint4 smem[];\n  pdl_wait();\n  pdl_trigger();\n  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);\n  if (item.z == 0) return;\n  const int e = item.x, ntok = item.z;\n  const int* l = lst + e * lmax + item.y;\n  if (ntok <= 8) wm_body_rs<MODE, NTB, 1, S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);\n  else if (NMAX >= 2 && ntok <= 16) wm_body_rs<MODE, NTB, (NMAX >= 2 ? 2 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);\n  else if (NMAX >= 4 && (NMAX < 8 || ntok <= 32)) wm_body_rs<MODE, NTB, (NMAX >= 4 ? 4 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);\n  else if (NMAX >= 8) wm_body_rs<MODE, NTB, (NMAX >= 8 ? 8 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);\n}\n\ntemplate <int MODE, int NTB, int NMAX, int S, int DBG>\nstatic void launch_rs(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst,\n                      const int* act, int maxslots, int lmax, int pdl, cudaStream_t st) {\n  dim3 grid((MODE == 0 ? 512 : 256) / (4 * NTB), (unsigned)maxslots);\n  const int dyn = S * (4 * NTB * 512 + NMAX * 2 * 32 * 16) + 3 * 4 * NTB * 256;\n  cudaFuncSetAttribute(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);\n  launch_k(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),\n           reinterpret_cast<const uint2*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);\n}\n\n\n// ---- wgmma (Hopper) building blocks ----\n__device__ __forceinline__ uint64_t wg_desc(const void* p, uint32_t lbo, uint32_t sbo) {\n  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);\n  return (uint64_t)((a >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);\n}\n__device__ __forceinline__ void wg_fence() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void wg_commit_wait() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.commit_group.sync.aligned;\\nwgmma.wait_group.sync.aligned 0;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void proxy_fence() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n#endif\n}\ntemplate <int N> struct WgN;\n\ntemplate <> struct WgN<32> {\n  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])\n                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n  }\n};\ntemplate <> struct WgN<64> {\n  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n  }\n};\ntemplate <> struct WgN<128> {\n  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n  }\n};\n// position of natural k inside its 64-chunk after the \'fragment order\' permutation the wgmma path reads its activations in\n__device__ __forceinline__ int wg_perm_k(int k) {\n  const int c = k >> 6, kl = k & 63;\n  const int t = kl >> 4, ks = (kl >> 2) & 3, p = (kl >> 1) & 1, h = kl & 1;\n  return c * 64 + (ks * 2 + p) * 8 + t * 2 + h;\n}\n__global__ void wm_permx_kernel(const bf16* __restrict__ X, bf16* __restrict__ Xp, int ld) {\n  pdl_wait();\n  pdl_trigger();\n  const size_t base = (size_t)blockIdx.x * ld;\n  for (int k = threadIdx.x; k < ld; k += blockDim.x) Xp[base + wg_perm_k(k)] = X[base + k];\n}\nextern "C" void wm_permx_c(const void* X, void* Xp, int M, cudaStream_t st) {\n  launch_k(wm_permx_kernel, dim3(M), dim3(256), 0, st, g_pdl != 0, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(Xp), 4096);\n}\n\n// ---- wgmma row-split variant: a warpgroup owns 64 weight rows (one 16-row tile per warp, decoded into registers) and up to\n// 128 tokens; B (activations) lives in shared memory in core-matrix layout ----\ntemplate <int MODE, int NCH, int S>\n__device__ __forceinline__ void wm_body_wg(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,\n                                           const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst, int e,\n                                           int ntok, uint8_t* smem_raw, const int bx) {\n  constexpr int RT = MODE == 0 ? RT0 : RT1;\n  constexpr int XW = MODE == 0 ? HD : ID;\n  constexpr int N = NCH * 8;\n  constexpr int NG = MODE == 0 ? 2 : 1;                  // weight tiles per warp: gate + up (MODE 0)\n  constexpr int XST = N * 128;                           // bytes of one x stage (N tokens x 64 k bf16)\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  uint8_t* xring = smem_raw;                             // [S][N tokens in core layout]\n  uint4* wring = reinterpret_cast<uint4*>(smem_raw + S * XST);   // [S][4 warps][NG][32 lanes]: one uint4 (4 ks words) per lane\n  const uint4* wsrc[NG];\n  const uint2* ssrc[NG];\n#pragma unroll\n  for (int q = 0; q < NG; q++) {\n    // MODE 0: gate tile bx*4+warp, up tile RT0/2 + same; MODE 1: tile bx*4+warp\n    const int tile = MODE == 0 ? (q == 0 ? bx * 4 + warp : RT0 / 2 + bx * 4 + warp) : bx * 4 + warp;\n    wsrc[q] = Wq + ((size_t)e * RT + tile) * KC * 32 + lane;\n    ssrc[q] = Sq + ((size_t)e * RT + tile) * KCG * 32 + lane;\n  }\n  // x staging: granule id = tid + 128 * u : token = id >> 3, k chunk = id & 7\n  const bf16* xb[NCH / 2];\n  bool xok[NCH / 2];\n#pragma unroll\n  for (int u = 0; u < NCH / 2; u++) {\n    const int id = tid + 128 * u;\n    const int col = id >> 3;\n    xok[u] = col < ntok;\n    xb[u] = X;\n    if (xok[u]) {\n      const int p = __ldcg(lst + col);\n      xb[u] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + (id & 7) * 8;\n    }\n  }\n  auto issue = [&](int c) {\n    if (c < KC) {\n#pragma unroll\n      for (int q = 0; q < NG; q++) cp16(&wring[(((c % S) * 4 + warp) * NG + q) * 32 + lane], wsrc[q] + c * 32);\n#pragma unroll\n      for (int u = 0; u < NCH / 2; u++) {\n        const int id = tid + 128 * u;\n        cp16z(xring + (c % S) * XST + (id >> 6) * 1024 + (id & 7) * 128 + ((id >> 3) & 7) * 16, xb[u] + c * 64, xok[u] ? 16u : 0u);\n      }\n    }\n    cp_commit();\n  };\n  float acc[NG][N / 2];\n#pragma unroll\n  for (int q = 0; q < NG; q++)\n#pragma unroll\n    for (int i = 0; i < N / 2; i++) acc[q][i] = 0.f;\n#pragma unroll\n  for (int c = 0; c < S - 1; c++) issue(c);\n  uint2 scur[NG], snxt[NG];\n#pragma unroll\n  for (int q = 0; q < NG; q++) {\n    scur[q] = __ldg(ssrc[q]);\n    snxt[q] = __ldg(ssrc[q] + 32);\n  }\n#pragma unroll 1\n  for (int i = 0; i < KC; i++) {\n    cp_wait<S - 2>();\n    proxy_fence();\n    __syncthreads();\n    issue(i + S - 1);\n    const int cq = i & 3;\n    uint32_t a[NG][4][4];\n#pragma unroll\n    for (int q = 0; q < NG; q++) {\n      const uint32_t s0 = scale2(scur[q].x, cq);\n      const uint32_t s1 = scale2(scur[q].y, cq);\n      const uint4 wv = wring[(((i % S) * 4 + warp) * NG + q) * 32 + lane];\n      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};\n#pragma unroll\n      for (int ks = 0; ks < 4; ks++) {\n        a[q][ks][0] = hmul(dec<0>(ww[ks]), s0);\n        a[q][ks][1] = hmul(dec<1>(ww[ks]), s1);\n        a[q][ks][2] = hmul(dec<2>(ww[ks]), s0);\n        a[q][ks][3] = hmul(dec<3>(ww[ks]), s1);\n      }\n    }\n    wg_fence();\n#pragma unroll\n    for (int ks = 0; ks < 4; ks++)\n#pragma unroll\n      for (int q = 0; q < NG; q++) WgN<N>::mma(acc[q], a[q][ks][0], a[q][ks][1], a[q][ks][2], a[q][ks][3], wg_desc(xring + (i % S) * XST + ks * 256, 128, 1024));\n    wg_commit_wait();\n    if (cq == 3) {\n#pragma unroll\n      for (int q = 0; q < NG; q++) {\n        scur[q] = snxt[q];\n        if ((i >> 2) + 2 < KCG) snxt[q] = __ldg(ssrc[q] + ((i >> 2) + 2) * 32);\n      }\n    }\n  }\n  cp_wait<0>();\n  const float gs = GS[e];\n#pragma unroll\n  for (int jb = 0; jb < N / 8; jb++)\n#pragma unroll\n    for (int r = 0; r < 4; r++) {\n      const int col = jb * 8 + 2 * t + (r & 1);\n      if (col >= ntok) continue;\n      const int p = __ldcg(lst + col);\n      const int row = warp * 16 + g + 8 * (r >> 1);\n      if (MODE == 0) {\n        const float gate = bfr(bfr(acc[0][jb * 4 + r] * 256.0f) * gs), up = bfr(bfr(acc[1][jb * 4 + r] * 256.0f) * gs);\n        const float sil = gate / (1.0f + __expf(-gate));\n        OUT[(size_t)p * ID + wg_perm_k(bx * 64 + row)] = __float2bfloat16(sil * up);\n      } else {\n        OUT[(size_t)p * HD + bx * 64 + row] = __float2bfloat16(bfr(acc[0][jb * 4 + r] * 256.0f) * gs);\n      }\n    }\n}\n\ntemplate <int MODE, int NMAX, int S>\n__global__ void __launch_bounds__(128, 2)\nwm_gemm_wg(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS, const bf16* __restrict__ X,\n           bf16* __restrict__ OUT, const int* __restrict__ lst, const int* __restrict__ act, int lmax) {\n  extern __shared__ __align__(16) uint8_t smem_wg[];\n  pdl_wait();\n  pdl_trigger();\n  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);\n  if (item.z == 0) return;\n  const int e = item.x, ntok = item.z;\n  const int* l = lst + e * lmax + item.y;\n  if (ntok <= 32) wm_body_wg<MODE, 4, S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);\n  else if (NMAX >= 8 && ntok <= 64) wm_body_wg<MODE, (NMAX >= 8 ? 8 : 4), S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);\n  else if (NMAX >= 16) wm_body_wg<MODE, (NMAX >= 16 ? 16 : 4), S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);\n}\n\ntemplate <int MODE, int NMAX, int S>\nstatic void launch_wg(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst, const int* act,\n                      int maxslots, int lmax, int pdl, cudaStream_t st) {\n  dim3 grid(64, (unsigned)maxslots);\n  constexpr int NG = MODE == 0 ? 2 : 1;\n  const int dyn = S * (NMAX * 8 * 128 + 4 * NG * 512);\n  cudaFuncSetAttribute(wm_gemm_wg<MODE, NMAX, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);\n  launch_k(wm_gemm_wg<MODE, NMAX, S>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),\n           reinterpret_cast<const uint2*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);\n}\n\n// Post-expert RMSNorm + weighted sum: out[m] = sum_k w[m,k] * rmsnorm_k(y[m,k] * nw), skipping slots routed elsewhere.\n// One CTA per token; warp k owns slot k for the sum of squares, then all threads accumulate their columns.\n__global__ void __launch_bounds__(256)\nwm_finish_kernel(const bf16* __restrict__ Y, const float* __restrict__ TW, const int* __restrict__ lids,\n                 const bf16* __restrict__ NW_, bf16* __restrict__ OUT, float eps) {\n  __shared__ float s_scale[8];\n  pdl_wait();\n  pdl_trigger();\n  const int m = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const bool live = lids[m * 8 + warp] >= 0;\n  float ss = 0.f;\n  if (live) {\n    const uint4* yv = reinterpret_cast<const uint4*>(Y + ((size_t)m * 8 + warp) * HD_ * 1);\n    const uint4* wv = reinterpret_cast<const uint4*>(NW_);\n#pragma unroll\n    for (int i = 0; i < 16; i++) {\n      const uint4 a = __ldcg(yv + i * 32 + lane), w = __ldg(wv + i * 32 + lane);\n      const bf162* ap = reinterpret_cast<const bf162*>(&a);\n      const bf162* wp = reinterpret_cast<const bf162*>(&w);\n#pragma unroll\n      for (int j = 0; j < 4; j++) {\n        const float2 af = __bfloat1622float2(ap[j]), wf = __bfloat1622float2(wp[j]);\n        const float v0 = af.x * wf.x, v1 = af.y * wf.y;\n        ss += v0 * v0 + v1 * v1;\n      }\n    }\n#pragma unroll\n    for (int d = 16; d >= 1; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);\n  }\n  if (lane == 0) s_scale[warp] = live ? TW[m * 8 + warp] * rsqrtf(ss / HD_ + eps) : 0.f;\n  __syncthreads();\n  // each thread: 16 columns = two uint4 of bf16\n  float acc[16];\n#pragma unroll\n  for (int j = 0; j < 16; j++) acc[j] = 0.f;\n  const uint4* wv = reinterpret_cast<const uint4*>(NW_) + tid * 2;\n  const uint4 w0 = __ldg(wv), w1 = __ldg(wv + 1);\n  float wf[16];\n  {\n    const bf162* p0 = reinterpret_cast<const bf162*>(&w0);\n    const bf162* p1 = reinterpret_cast<const bf162*>(&w1);\n#pragma unroll\n    for (int j = 0; j < 4; j++) {\n      const float2 a = __bfloat1622float2(p0[j]), b = __bfloat1622float2(p1[j]);\n      wf[2 * j] = a.x; wf[2 * j + 1] = a.y; wf[8 + 2 * j] = b.x; wf[8 + 2 * j + 1] = b.y;\n    }\n  }\n#pragma unroll\n  for (int k = 0; k < 8; k++) {\n    const float sc = s_scale[k];\n    if (sc != 0.f) {\n      const uint4* yv = reinterpret_cast<const uint4*>(Y + ((size_t)m * 8 + k) * HD_) + tid * 2;\n      const uint4 a0 = __ldcg(yv), a1 = __ldcg(yv + 1);\n      const bf162* p0 = reinterpret_cast<const bf162*>(&a0);\n      const bf162* p1 = reinterpret_cast<const bf162*>(&a1);\n#pragma unroll\n      for (int j = 0; j < 4; j++) {\n        const float2 a = __bfloat1622float2(p0[j]), b = __bfloat1622float2(p1[j]);\n        acc[2 * j] += sc * (a.x * wf[2 * j]);\n        acc[2 * j + 1] += sc * (a.y * wf[2 * j + 1]);\n        acc[8 + 2 * j] += sc * (b.x * wf[8 + 2 * j]);\n        acc[8 + 2 * j + 1] += sc * (b.y * wf[8 + 2 * j + 1]);\n      }\n    }\n  }\n  uint4 o0, o1;\n  bf162* q0 = reinterpret_cast<bf162*>(&o0);\n  bf162* q1 = reinterpret_cast<bf162*>(&o1);\n#pragma unroll\n  for (int j = 0; j < 4; j++) {\n    q0[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);\n    q1[j] = __floats2bfloat162_rn(acc[8 + 2 * j], acc[8 + 2 * j + 1]);\n  }\n  uint4* ov = reinterpret_cast<uint4*>(OUT + (size_t)m * HD_) + tid * 2;\n  ov[0] = o0;\n  ov[1] = o1;\n}\nextern "C" void wm_finish_c(const void* Y, const float* TW, const int* lids, const void* NWT, void* OUT, int M, float eps,\n                            cudaStream_t st) {\n  launch_k(wm_finish_kernel, dim3(M), dim3(256), 0, st, g_pdl != 0, reinterpret_cast<const bf16*>(Y), TW, lids,\n           reinterpret_cast<const bf16*>(NWT), reinterpret_cast<bf16*>(OUT), eps);\n}\n\nextern "C" void wm_route_c(const int* ids, int npair, const int* ls_ptr, int ls_val, int lmax, int* cnt, int* lst,\n                           int* act, int* nact, int* local_ids, int* woff, int grp, int nslots, cudaStream_t st) {\n  const int pdl = g_pdl;\n  launch_k(route_kernel, dim3(1), dim3(1024), 0, st, pdl != 0, ids, npair, ls_ptr, ls_val, lmax, cnt, lst, act, nact, local_ids, woff, grp, nslots);\n}\n\ntemplate <int MODE, int NT, int NMAX, int NW, int RING, int DBG>\nstatic void launch(const void* W, const void* S, const float* GS, const void* X, void* OUT, const int* cnt,\n                   const int* lst, const int* act, const int* nact, const int* woff, int maxslots, int lmax, int pdl, cudaStream_t st) {\n  dim3 grid((MODE == 0 ? 512 : 256) / NT, (unsigned)maxslots);\n  if (g_persist) { const unsigned cap = 132u * (unsigned)g_persist; const unsigned tot = grid.x * grid.y; grid = dim3(tot < cap ? tot : cap, 1); }\n  int dyn = NW * NT * 6 * 512 > 49152 ? 49152 : NW * NT * 6 * 512;\n  if (NT == 8) dyn = NW * 3 * 8 * 512 + 1024;\n  if (RING == 3) { const int d3 = NW * 4 * 4 * 512; dyn = d3 > dyn ? d3 : dyn; if (NW * NT * 4 * 512 > dyn) dyn = NW * NT * 4 * 512; }\n  if (RING == 2) {\n    int per = ring2_stages<NT, 1>() * (NT + 2);\n    if (NMAX >= 2) { const int p2 = ring2_stages<2, 2>() * (2 + 4); per = per > p2 ? per : p2; }\n    if (NMAX >= 4) { const int p4 = ring2_stages<2, 4>() * (2 + 8); per = per > p4 ? per : p4; }\n    dyn = NW * per * 512;\n  }\n  cudaFuncSetAttribute(wm_gemm<MODE, NT, NMAX, NW, RING, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);\n  launch_k(wm_gemm<MODE, NT, NMAX, NW, RING, DBG>, grid, dim3(NW * 32), (size_t)dyn, st, pdl != 0,\n           reinterpret_cast<const uint4*>(W), reinterpret_cast<const uint2*>(S), GS, reinterpret_cast<const bf16*>(X),\n           reinterpret_cast<bf16*>(OUT), cnt, lst, act, nact, woff, lmax, g_persist);\n}\n\n\n// ---- L2 prefetch of the down-projection weights of the active experts (issued while the gate/up GEMM streams) ----\n__device__ __forceinline__ uint64_t pol_evict_last() {\n  uint64_t p;\n  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));\n  return p;\n}\n__device__ __forceinline__ void pf_l2(const void* ptr, uint32_t bytes, uint64_t pol) {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(ptr), "r"(bytes), "l"(pol) : "memory");\n#endif\n}\n// W2 codes of one expert: RT1 * KC * 512 B = 8 MiB; scales RT1 * KCG * 256 B = 1 MiB. Slab = 8 KiB.\n__global__ void __launch_bounds__(128)\nwm_pf_kernel(const uint8_t* __restrict__ W2, const uint8_t* __restrict__ S2, const int* __restrict__ act,\n             const int* __restrict__ nact_p, int max_items, int slabs_code, int slabs_scale, size_t code_stride, size_t scale_stride) {\n  pdl_wait();\n  pdl_trigger();\n  const int nitems = min(__ldcg(nact_p), max_items);\n  const uint64_t pol = pol_evict_last();\n  const int per = slabs_code + slabs_scale;\n  const long total = (long)nitems * per;\n  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {\n    const int it = (int)(i / per), r = (int)(i % per);\n    const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + it);\n    if (item.z == 0 || item.y != 0) continue;   // one prefetch per expert (its first work item)\n    const size_t e = item.x;\n    if (r < slabs_code) pf_l2(W2 + e * code_stride + (size_t)r * 8192, 8192, pol);\n    else pf_l2(S2 + e * scale_stride + (size_t)(r - slabs_code) * 8192, 8192, pol);\n  }\n}\nextern "C" void wm_pf_c(const void* W2, const void* S2, const int* act, const int* nact, int max_items, int pdl, cudaStream_t st) {\n  constexpr size_t CODE = (size_t)RT1 * KC * 512, SCALE = (size_t)RT1 * KCG * 256;\n  launch_k(wm_pf_kernel, dim3(132), dim3(128), 0, st, pdl != 0, reinterpret_cast<const uint8_t*>(W2),\n           reinterpret_cast<const uint8_t*>(S2), act, nact, max_items, (int)(CODE / 8192), (int)((SCALE + 8191) / 8192), CODE, SCALE);\n}\n\n// cfg bits: nmax (1,2,4) | nt (2,4) | nw (4,8) | ring (0/1)\n\ntemplate<int MODE,int NCH>\n__global__ void __launch_bounds__(128,(NCH<=2?4:3))\nwm_bucket(const uint4* W,const uint2* SF,const float* GS,const bf16* X,bf16* OUT,\n const int* lst,const int* act,const int* nact,int lmax){\n extern __shared__ uint4 smem[];\n pdl_wait();pdl_trigger();\n constexpr int tier=NCH==1?0:(NCH==2?1:2);\n constexpr int tiles=MODE==0?32:16;\n int count=__ldcg(nact+1+tier);\n if constexpr(NCH==4){\n   int slot=blockIdx.y;if(slot>=count)return;\n   int4 item=__ldcg(reinterpret_cast<const int4*>(act+512*(tier+1))+slot);\n   wm_body_rs<MODE,4,NCH,4,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,smem,blockIdx.x);\n }else for(int job=blockIdx.x;job<count*tiles;job+=gridDim.x){\n   int slot=job/tiles,bx=job%tiles;\n   int4 item=__ldcg(reinterpret_cast<const int4*>(act+512*(tier+1))+slot);\n   wm_body_rs<MODE,4,NCH,4,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,smem,bx);\n   __syncthreads();\n }\n}\ntemplate<int MODE,int NCH>\nstatic void bucket_launch(const void* W,const void* SF,const float* GS,const void* X,void* OUT,\n const int* lst,const int* act,const int* nact,int lmax,int cap,cudaStream_t stream){\n constexpr int dyn=45056+4096*NCH;\n cudaFuncSetAttribute(wm_bucket<MODE,NCH>,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);\n cudaFuncSetAttribute(wm_bucket<MODE,NCH>,cudaFuncAttributePreferredSharedMemoryCarveout,100);\n dim3 grid=NCH==4?dim3(MODE==0?32:16,cap):dim3(cap);\n launch_k(wm_bucket<MODE,NCH>,grid,dim3(128),dyn,stream,g_pdl!=0,\n (const uint4*)W,(const uint2*)SF,GS,(const bf16*)X,(bf16*)OUT,lst,act,nact,lmax);\n}\nextern "C" int wm_bucket_c(int mode,int tier,int cap,const void* W,const void* SF,const float* GS,const void* X,void* OUT,\n const int* lst,const int* act,const int* nact,int lmax,cudaStream_t stream){\n#define BKT(M,T,N) if(mode==M&&tier==T){bucket_launch<M,N>(W,SF,GS,X,OUT,lst,act,nact,lmax,cap,stream);return cudaGetLastError();}\n BKT(0,0,1) BKT(0,1,2) BKT(0,2,4) BKT(1,0,1) BKT(1,1,2) BKT(1,2,4)\n#undef BKT\n return cudaErrorInvalidValue;\n}\nextern "C" int wm_gemm_c(int mode, int nmax, int nt, int nw, int ring, int dbg, const void* W, const void* S, const float* GS,\n                         const void* X, void* OUT, const int* cnt, const int* lst, const int* act, const int* nact,\n                         const int* woff, int maxslots, int lmax, cudaStream_t st) {\n  const int pdl = g_pdl;\n#define WM_WG(M, N_, S_) if (mode == M && ring == 7 && nmax == N_ && nt == S_ && dbg == 0) { launch_wg<M, N_, S_>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }\n#undef WM_WG\n#define WM_RS(M, T, N, S_) \\\n  if (mode == M && ring == 4 && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }\n  WM_RS(0, 4, 4, 4) WM_RS(1, 4, 4, 4) WM_RS(0, 4, 2, 4) WM_RS(1, 4, 2, 4) WM_RS(0, 4, 1, 4) WM_RS(1, 4, 1, 4)\n  WM_RS(0, 2, 4, 4) WM_RS(1, 2, 4, 4) WM_RS(0, 2, 2, 4) WM_RS(1, 2, 2, 4) WM_RS(0, 2, 1, 4) WM_RS(1, 2, 1, 4)\n  WM_RS(0, 2, 8, 4) WM_RS(1, 2, 8, 4) WM_RS(0, 2, 8, 3) WM_RS(1, 2, 8, 3) WM_RS(0, 4, 8, 3) WM_RS(1, 4, 8, 3)\n#undef WM_RS\n#define WM_RS(M, T, N, S_, R_) \\\n  if (mode == M && ring == R_ && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }\n  WM_RS(0, 4, 4, 6, 5) WM_RS(1, 4, 4, 6, 5) WM_RS(0, 2, 4, 6, 5) WM_RS(1, 2, 4, 6, 5) WM_RS(0, 4, 4, 8, 6) WM_RS(1, 4, 4, 8, 6) WM_RS(0, 2, 4, 8, 6) WM_RS(1, 2, 4, 8, 6)\n  WM_RS(0, 4, 2, 6, 5) WM_RS(1, 4, 2, 6, 5) WM_RS(0, 4, 1, 6, 5) WM_RS(1, 4, 1, 6, 5)\n  WM_RS(0, 4, 4, 3, 8) WM_RS(1, 4, 4, 3, 8) WM_RS(0, 2, 4, 3, 8) WM_RS(1, 2, 4, 3, 8) WM_RS(0, 2, 8, 3, 9) WM_RS(1, 2, 8, 3, 9) WM_RS(0, 2, 4, 3, 9) WM_RS(1, 2, 4, 3, 9)\n#undef WM_RS\n#define WM_RSD(M, D) if (mode == M && ring == 4 && nt == 4 && nmax == 4 && dbg == D) { launch_rs<M, 4, 4, 4, D>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }\n  WM_RSD(0, 1) WM_RSD(1, 1) WM_RSD(0, 2) WM_RSD(1, 2) WM_RSD(0, 8) WM_RSD(1, 8) WM_RSD(0, 4) WM_RSD(1, 4) WM_RSD(0, 15) WM_RSD(1, 15) WM_RSD(0, 3) WM_RSD(1, 3) WM_RSD(0, 16) WM_RSD(1, 16) WM_RSD(0, 32) WM_RSD(1, 32) WM_RSD(0, 48) WM_RSD(1, 48) WM_RSD(0, 63) WM_RSD(1, 63) WM_RSD(0, 31) WM_RSD(1, 31) WM_RSD(0, 64) WM_RSD(1, 64)\n#undef WM_RSD\n#define WM_D(M, T, N, W_, R)                                                                             \\\n  if (mode == M && nt == T && nmax == N && nw == W_ && ring == R && dbg == 0) {                           \\\n    launch<M, T, N, W_, R, 0>(W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax, pdl, st);          \\\n    return 0;                                                                                             \\\n  }\n#define WM_DBG(M, T, N, W_, R, D)                                                                         \\\n  if (mode == M && nt == T && nmax == N && nw == W_ && ring == R && dbg == D) {                           \\\n    launch<M, T, N, W_, R, D>(W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax, pdl, st);          \\\n    return 0;                                                                                             \\\n  }\n#define WM_DBGS(M, T, N, W_, R) WM_DBG(M, T, N, W_, R, 1) WM_DBG(M, T, N, W_, R, 2) WM_DBG(M, T, N, W_, R, 8) WM_DBG(M, T, N, W_, R, 3) WM_DBG(M, T, N, W_, R, 11) WM_DBG(M, T, N, W_, R, 4) WM_DBG(M, T, N, W_, R, 9)\n#define WM_ALL(T, N, W_, R) WM_D(0, T, N, W_, R) WM_D(1, T, N, W_, R)\n  WM_ALL(2, 1, 4, 0) WM_ALL(2, 2, 4, 0) WM_ALL(2, 1, 8, 0) WM_ALL(2, 2, 8, 0) WM_ALL(4, 1, 4, 0) WM_ALL(4, 2, 4, 0)\n  WM_ALL(2, 1, 4, 1) WM_ALL(2, 2, 4, 1) WM_ALL(2, 4, 4, 1) WM_ALL(4, 1, 4, 1) WM_ALL(4, 2, 4, 1) WM_ALL(4, 4, 4, 1)\n  WM_ALL(2, 1, 8, 1) WM_ALL(2, 2, 8, 1) WM_ALL(2, 4, 8, 1)\n  WM_ALL(8, 1, 4, 1) WM_ALL(8, 2, 4, 1) WM_ALL(8, 4, 4, 1)\n  WM_ALL(4, 4, 4, 3) WM_ALL(4, 2, 4, 3) WM_ALL(4, 1, 4, 3) WM_ALL(8, 4, 4, 3) WM_ALL(8, 2, 4, 3)\n  WM_ALL(4, 1, 2, 1) WM_ALL(4, 2, 2, 1) WM_ALL(4, 4, 2, 1) WM_ALL(2, 1, 2, 0) WM_ALL(2, 2, 2, 0) WM_ALL(2, 1, 2, 1)\n  WM_ALL(2, 1, 4, 2) WM_ALL(2, 2, 4, 2) WM_ALL(2, 4, 4, 2) WM_ALL(4, 1, 4, 2) WM_ALL(4, 2, 4, 2) WM_ALL(4, 4, 4, 2)\n  WM_ALL(2, 1, 8, 2) WM_ALL(2, 2, 8, 2) WM_ALL(2, 4, 8, 2)\n#ifdef WM_DBG_BUILD\n  WM_DBGS(0, 2, 1, 4, 0) WM_DBGS(1, 2, 1, 4, 0) WM_DBGS(0, 2, 1, 4, 2) WM_DBGS(1, 2, 1, 4, 2) WM_DBGS(0, 2, 2, 4, 2) WM_DBGS(1, 2, 2, 4, 2)\n  WM_DBGS(0, 4, 4, 4, 2) WM_DBGS(1, 4, 4, 4, 2) WM_DBGS(0, 2, 4, 4, 2) WM_DBGS(1, 2, 4, 4, 2)\n  WM_DBGS(0, 4, 4, 4, 1) WM_DBGS(1, 4, 4, 4, 1)\n#endif\n#undef WM_ALL\n#undef WM_D\n  return 1;\n}\n\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\nextern "C" void wm_route_c(const int*, int, const int*, int, int, int*, int*, int*, int*, int*, int*, int, int, cudaStream_t);\nextern "C" void wm_set_pdl(int);\nvoid wm_set_pdl_t(int64_t v) { wm_set_pdl((int)v); }\nextern "C" void wm_set_persist(int);\nvoid wm_set_persist_t(int64_t v) { wm_set_persist((int)v); }\nextern "C" void wm_finish_c(const void*, const float*, const int*, const void*, void*, int, float, cudaStream_t);\nvoid wm_finish(torch::Tensor y, torch::Tensor tw, torch::Tensor lids, torch::Tensor nw, torch::Tensor out, double eps) {\n  wm_finish_c(y.data_ptr(), tw.data_ptr<float>(), lids.data_ptr<int>(), nw.data_ptr(), out.data_ptr(), (int)out.size(0), (float)eps,\n              at::cuda::getCurrentCUDAStream());\n}\nextern "C" void wm_permx_c(const void*, void*, int, cudaStream_t);\nvoid wm_permx(torch::Tensor x, torch::Tensor xp) { wm_permx_c(x.data_ptr(), xp.data_ptr(), (int)x.size(0), at::cuda::getCurrentCUDAStream()); }\nextern "C" int wm_gemm_c(int, int, int, int, int, int, const void*, const void*, const float*, const void*, void*, const int*,\n                         const int*, const int*, const int*, const int*, int, int, cudaStream_t);\nextern "C" void wm_pf_c(const void*, const void*, const int*, const int*, int, int, cudaStream_t);\nvoid wm_pf(torch::Tensor w2, torch::Tensor s2, torch::Tensor act, torch::Tensor nact, int64_t max_items, int64_t pdl) {\n  wm_pf_c(w2.data_ptr(), s2.data_ptr(), act.data_ptr<int>(), nact.data_ptr<int>(), (int)max_items, (int)pdl, at::cuda::getCurrentCUDAStream());\n}\n\nvoid wm_route(torch::Tensor ids, torch::Tensor ls, int64_t ls_val, int64_t lmax, torch::Tensor cnt, torch::Tensor lst,\n              torch::Tensor act, torch::Tensor nact, torch::Tensor local_ids, torch::Tensor woff, int64_t grp) {\n  wm_route_c(ids.data_ptr<int>(), (int)ids.numel(), ls.numel() ? ls.data_ptr<int>() : nullptr, (int)ls_val, (int)lmax, cnt.data_ptr<int>(),\n             lst.data_ptr<int>(), act.data_ptr<int>(), nact.data_ptr<int>(), local_ids.data_ptr<int>(), woff.data_ptr<int>(), (int)grp,\n             (int)(act.numel() / 4), at::cuda::getCurrentCUDAStream());\n}\n\nvoid wm_gemm_run(int64_t mode, int64_t nmax, int64_t nt, int64_t nw, int64_t ring, int64_t dbg, torch::Tensor W, torch::Tensor S, torch::Tensor GS, torch::Tensor X,\n                 torch::Tensor OUT, torch::Tensor cnt, torch::Tensor lst, torch::Tensor act, torch::Tensor nact,\n                 torch::Tensor woff, int64_t maxslots, int64_t lmax) {\n  int r = wm_gemm_c((int)mode, (int)nmax, (int)nt, (int)nw, (int)ring, (int)dbg, W.data_ptr(), S.data_ptr(), GS.data_ptr<float>(), X.data_ptr(),\n                    OUT.data_ptr(), cnt.data_ptr<int>(), lst.data_ptr<int>(), act.data_ptr<int>(),\n                    nact.data_ptr<int>(), woff.data_ptr<int>(), (int)maxslots, (int)lmax, at::cuda::getCurrentCUDAStream());\n  TORCH_CHECK(r == 0, "unsupported config");\n}\n\nextern "C" int wm_bucket_c(int,int,int,const void*,const void*,const float*,const void*,void*,const int*,const int*,const int*,int,cudaStream_t);\nvoid wm_bucket_run(int64_t mode,int64_t tier,int64_t cap,torch::Tensor W,torch::Tensor SF,torch::Tensor GS,torch::Tensor X,torch::Tensor OUT,torch::Tensor lst,torch::Tensor act,torch::Tensor nact,int64_t lmax){\n int e=wm_bucket_c(mode,tier,cap,W.data_ptr(),SF.data_ptr(),GS.data_ptr<float>(),X.data_ptr(),OUT.data_ptr(),lst.data_ptr<int>(),act.data_ptr<int>(),nact.data_ptr<int>(),lmax,at::cuda::getCurrentCUDAStream());\n TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));\n}\n'
_WM_CPP = '\nvoid wm_set_pdl_t(int64_t v);\nvoid wm_set_persist_t(int64_t v);\nvoid wm_pf(torch::Tensor w2, torch::Tensor s2, torch::Tensor act, torch::Tensor nact, int64_t max_items, int64_t pdl);\nvoid wm_permx(torch::Tensor x, torch::Tensor xp);\nvoid wm_finish(torch::Tensor y, torch::Tensor tw, torch::Tensor lids, torch::Tensor nw, torch::Tensor out, double eps);\nvoid wm_route(torch::Tensor ids, torch::Tensor ls, int64_t ls_val, int64_t lmax, torch::Tensor cnt, torch::Tensor lst,\n              torch::Tensor act, torch::Tensor nact, torch::Tensor local_ids, torch::Tensor woff, int64_t grp);\nvoid wm_gemm_run(int64_t mode, int64_t nmax, int64_t nt, int64_t nw, int64_t ring, int64_t dbg, torch::Tensor W, torch::Tensor S, torch::Tensor GS, torch::Tensor X,\n                 torch::Tensor OUT, torch::Tensor cnt, torch::Tensor lst, torch::Tensor act, torch::Tensor nact,\n                 torch::Tensor woff, int64_t maxslots, int64_t lmax);\n\nvoid wm_bucket_run(int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);\n'

def _wm_repack(w, sf):
    """w: uint8 [E, R, K/2] (low nibble first), sf: float8_e4m3fn [E, R, K/16].
    Returns (words int32 [E, R/16, K/64, 32, 4], scales int32-viewable uint8 [E, R/16, K/256, 32, 8])."""
    E, R, K2 = w.shape
    K = K2 * 2
    RT, KCn = R // 16, K // 64
    out_w = torch.empty((E, RT, KCn, 32, 4), dtype=torch.int32, device=w.device)
    out_s = torch.empty((E, RT, KCn // 4, 32, 8), dtype=torch.uint8, device=w.device)
    # per-reg source bit positions for (magnitude bits m0..m2, sign), low half; high half is +16
    src = {
        0: (6, 7, 8, 15),
        1: (9, 10, 11, 18),
        2: (12, 13, 14, 21),
        3: (4, 0, 1, 3),
    }
    for e in range(E):
        t = w[e]
        codes = torch.stack((t & 15, t >> 4), dim=-1).reshape(R, K).to(torch.int64)
        # [rt, rh, g, kc, t, ks, p, h]
        c8 = codes.view(RT, 2, 8, KCn, 4, 4, 2, 2)
        word = torch.zeros((RT, 8, KCn, 4, 4), dtype=torch.int64, device=w.device)  # [rt,g,kc,t,ks]
        for rh in range(2):
            for p in range(2):
                j = rh + 2 * p
                for h in range(2):
                    c = c8[:, rh, :, :, :, :, p, h]  # [rt, g, kc, t, ks]
                    for bit in range(4):
                        pos = (src[j][bit] + 16 * h) % 32
                        word |= ((c >> bit) & 1) << pos
        word = word.permute(0, 2, 1, 3, 4)  # [rt, kc, g, t, ks]
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        out_w[e] = word.reshape(RT, KCn, 32, 4)
        # scales: raw e4m3 -> b' with S*2^118 = bf16 bits ((b'<<4) + 0x7600)
        s = sf[e].float() * (2.0 ** 118)
        bits = (s.view(torch.int32) >> 16).clamp(min=0x7600)
        bp = ((bits - 0x7600) >> 4).to(torch.uint8)  # [R, K/16]
        bp = bp.view(RT, 2, 8, KCn // 4, 4, 4)  # [rt, rh, g, kcg, c, t]; group = kcg*16 + c*4 + t
        out_s[e] = bp.permute(0, 3, 2, 5, 1, 4).reshape(RT, KCn // 4, 32, 8)  # [rt, kcg, (g,t), (rh,c)]
    return out_w, out_s


_WM_MOD = None
_WM_DBG = 0
_WM_PDL = True
_WM_FIN = False
_WM_NOLS = None
_WM_EXTRA = []


def _wm_module():
    global _WM_MOD
    if _WM_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _WM_MOD = load_inline('muse_wm_bucket', _WM_CPP, cuda_sources=_WM_CUDA, functions=['wm_route', 'wm_gemm_run', 'wm_set_pdl_t', 'wm_set_persist_t', 'wm_finish', 'wm_pf', 'wm_bucket_run'], extra_cuda_cflags=['-O3', '--use_fast_math'] + _WM_EXTRA, verbose=False)
        _WM_MOD.wm_set_pdl_t(int(_WM_PDL and torch.cuda.get_device_capability()[0] >= 9))
    return _WM_MOD


def prepare(weights):
    global _WM_NOLS
    state = _prepare_marlin(weights)
    mod = _wm_module()
    _WM_NOLS = torch.empty(0, dtype=torch.int32, device=weights['w13'].device)
    w13, s13 = _wm_repack(weights["w13"][:32], weights["w13_sf"][:32])
    w2, s2 = _wm_repack(weights["w2"][:32], weights["w2_sf"][:32])
    state["wm"] = {"mod": mod, "w13": w13, "s13": s13, "gs13": weights["w13_gs"][:32].contiguous(),
                   "w2": w2, "s2": s2, "gs2": weights["w2_gs"][:32].contiguous()}
    torch.cuda.synchronize()
    return state


WM_MAX_TOKENS = 256
_WM_PF = 0          # experts whose down-projection weights are prefetched into L2 during the gate/up GEMM
_WM_PF_MINM = 1
# (m upper bound) -> (tiles per warp, warps per CTA, cp.async ring)
_WM_CFG = [(1,(2,4,0)),(8,(2,2,0)),(16,(2,8,0)),(32,(4,4,1)),(10**9,(4,4,4))]


def _wm_cfg(m):
    for bound, cfg in _WM_CFG:
        if m <= bound:
            return cfg
    return _WM_CFG[-1][1]


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    if (m > WM_MAX_TOKENS or cfg["local_start"] != 0 or cfg["local_count"] != 32 or cfg["d_hidden"] != 4096
            or cfg["d_expert"] != 4096):
        return _marlin_kernel_fn(data, state)
    wm = state["wm"]
    mod = wm["mod"]
    x = x.contiguous()
    nch = 1 if m <= 8 else 2 if m <= 16 else 4
    lmax = (m + 7) // 8 * 8
    _c = _wm_cfg(m)
    nt, nw, ring = _c[:3]
    nt1 = _c[3] if len(_c) > 3 else nt
    dev = x.device
    ids = data["topk_ids"].contiguous()
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(512, dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    mod.wm_route(ids, _WM_NOLS, 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
    if _WM_PF > 0 and m >= _WM_PF_MINM:
        mod.wm_pf(wm["w2"], wm["s2"], act, nact, _WM_PF, 1)
    maxslots = min(m * TOP_K, 32 + (m * TOP_K) // (nch * 8))
    inter = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)
    y = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)
    mod.wm_gemm_run(0, nch, nt, nw, ring, _WM_DBG, wm["w13"], wm["s13"], wm["gs13"], x, inter, cnt, lst, act, nact, woff, maxslots, lmax)
    mod.wm_gemm_run(1, nch, nt1, nw, ring, _WM_DBG, wm["w2"], wm["s2"], wm["gs2"], inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    if _WM_FIN:
        out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
        mod.wm_finish(y, topk_weights, lids, state["norm_w"], out, POST_NORM_EPS)
        return out
    return _finish_experts(
        y.view(m, TOP_K, 4096), topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=lids.view(m, TOP_K))


def _bucket_forward(data,state):
    x=data['hidden_states'].contiguous();m=x.shape[0];dev=x.device
    wm=state['wm'];mod=wm['mod'];lmax=(m+7)//8*8
    ids=data['topk_ids'].contiguous()
    cnt=torch.empty(32,dtype=torch.int32,device=dev)
    lst=torch.empty(32*lmax,dtype=torch.int32,device=dev)
    act=torch.empty(2048,dtype=torch.int32,device=dev)
    nact=torch.empty(4,dtype=torch.int32,device=dev)
    lids=torch.empty(m*TOP_K,dtype=torch.int32,device=dev)
    woff=torch.empty(128,dtype=torch.int32,device=dev)
    mod.wm_route(ids,_WM_NOLS,0,lmax,cnt,lst,act,nact,lids,woff,32)
    inter=torch.empty((m*TOP_K,4096),dtype=torch.bfloat16,device=dev)
    y=torch.empty_like(inter)
    main=torch.cuda.current_stream();parallel=state.get('_bucket_parallel',True)
    def run(tier):
        cap=state['_bucket_caps'][tier]
        if tier==2:cap=min(cap,32+(m*TOP_K)//32)
        mod.wm_bucket_run(0,tier,cap,wm['w13'],wm['s13'],wm['gs13'],x,inter,lst,act,nact,lmax)
        mod.wm_bucket_run(1,tier,cap,wm['w2'],wm['s2'],wm['gs2'],inter,y,lst,act,nact,lmax)
    if parallel:
        for tier in (2,1,0):
            stream=state['_bucket_streams'][tier];stream.wait_stream(main)
            with torch.cuda.stream(stream):run(tier)
        for stream in state['_bucket_streams']:main.wait_stream(stream)
    else:
        for tier in (2,1,0):run(tier)
    tw=data['topk_weights'].to(torch.float32).contiguous()
    return _finish_experts(y.view(m,TOP_K,4096),tw,norm_weight=state['norm_w'],eps=POST_NORM_EPS,weight_first=True,slot_expert_ids=lids.view(m,TOP_K))

_before_bucket_prepare=prepare
def prepare(weights):
    state=_before_bucket_prepare(weights)
    state['_bucket_streams']=[torch.cuda.Stream() for _ in range(3)]
    sms=torch.cuda.get_device_properties(weights['w13'].device).multi_processor_count
    state['_bucket_caps']=(sms*4,sms*2,sms*2)
    return state
_before_bucket_kernel_fn=kernel_fn
def kernel_fn(data,state):
    c=data['config'];m=data['hidden_states'].shape[0]
    if (128<=m<=256 and c['local_start']==0 and c['local_count']==32 and c['d_hidden']==4096 and c['d_expert']==4096):
        return _bucket_forward(data,state)
    return _before_bucket_kernel_fn(data,state)
