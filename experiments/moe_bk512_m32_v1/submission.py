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
    if (256<=m<=256 and c['local_start']==0 and c['local_count']==32 and c['d_hidden']==4096 and c['d_expert']==4096):
        return _bucket_forward(data,state)
    return _before_bucket_kernel_fn(data,state)

# Component: MIT team bigboateng, operatorstack/open-kernel-challenge-solutions
# Commit 6d9b77656b6c295e421be9e1424b0f8f99299b0a. Namespace-only adaptation.
# MIT License
# 
# Copyright (c) 2026 bigboateng
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# Open Kernel Challenge solutions (team bigboateng)
# Copyright (c) 2026 bigboateng. Licensed under the MIT License (see LICENSE),
# except for the portions listed below.
# 
# This repository uses and reproduces code from SGLang
# (https://github.com/sgl-project/sglang), Copyright the SGLang authors,
# licensed under the Apache License, Version 2.0
# (https://www.apache.org/licenses/LICENSE-2.0). The SGLang-derived portions
# listed below remain under the Apache License 2.0. The solutions also import
# SGLang (sglang, sgl_kernel) at run time; those packages are not included here.
# 
# muse_moe_w4a16/submission.py
#   - Imports _post_expert_norm_weighted_sum_kernel from
#     sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py.
#   - _wsum_ks reproduces that kernel's body (same math, order and rsqrt), with
#     one change: it first sums FP32 split-K partials and rounds them to BF16.
# 
# muse_bf16_proj/submission.py
#   - Imports _bf16_gemm_dispatch_impl from
#     sglang/srt/layers/quantization/unquant.py (the reference path for every
#     shape this file does not serve with its own kernels).
#   - The router's hi/lo split with FP32 accumulation follows
#     MuseSpark1xRouter.forward_split2.
# 
# muse_decode_layer/submission.py
#   - _post_expert_ws_pdl_kernel reproduces the body of SGLang's
#     _post_expert_norm_weighted_sum_kernel
#     (sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py),
#     with programmatic dependent launch added, and with an optional sum of FP32
#     split-K partials before the norm.
#   - _router_hilo uses the same hi/lo split with FP32 accumulation as
#     MuseSpark1xRouter.forward_split2.
#   - _rms_norm_local_ids_kernel reimplements SGLang's single-CTA
#     moe_align_small_numel algorithm (same "+1 offset" buckets) inside the
#     pre-expert norm kernel.
#   - Imports SGLang modules and kernels: _post_expert_norm_weighted_sum_kernel,
#     silu_and_mul, moe_wna16_marlin_gemm, FusedMuseSparkRouting,
#     FusedRMSNormGate, FusedPostNormResidualPreNorm, fused_norm_residual,
#     moe_align_block_size, MuseSpark1xNormPreservingResidualConnection,
#     MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale,
#     marlin_make_workspace, prepare_moe_nvfp4_layer_for_marlin,
#     _bf16_gemm_dispatch_impl, and sgl_kernel's flash_attn_with_kvcache and
#     scalar_types.
# 
# muse_attention_decode/submission.py
#   - Imports sgl_kernel's flash_attn_with_kvcache and SGLang's rms_norm_gate
#     (sglang/kernels/ops/muse_spark_v12/norm_gate.py).

"""Routed NVFP4 (W4A16) MoE experts for one EP8 rank, on grouped Gluon kernels. From 128 tokens (verify) both GEMMs
run on the register path with a per-item token tile: W13 on work items of up to 32 rows (one weight decode for up to
32 token rows, 8/16/32-row tile per item) and W2 on items of up to 16 rows (8/16-row tile per item).

Pipeline (4 launches; 3 at decode 1, where each top-k slot is one item and the routing pass is skipped):
  1. _route (Triton, one CTA): maps global expert ids to this rank's experts, groups the (token, slot) pairs by expert
     (sorted, so pair order within an expert is deterministic) and writes a work list of (expert, first row, end row)
     items of at most BM rows.
  2. _w13_silu (Gluon): for each work item and 64-row N tile, streams the expert's gate and up weight rows once,
     multiplies them with the item's token rows and writes silu(gate) * up (BF16) per pair.
  3. _w2 (Gluon): the same for W2 on the intermediate rows, writing BF16 expert outputs per pair.
  4. SGLang's fused post-expert-norm weighted-sum Triton kernel: RMSNorm with (1 + w), gate weight, sum over slots.

GEMM data path (per CTA: 4 warps, 64 weight rows, K in blocks of 256):
- Weights stay NVFP4. prepare() only permutes the bits inside each 32-bit word of packed E2M1 codes (lossless), so
  that one rotate and one mask place a pair of codes in BF16 bit positions (value * 2^-126); a bf16x2 multiply by
  the E4M3 block scale (times 2^119, exact) completes the decode. The remaining 2^7 and the FP32 global scale are
  applied to the FP32 accumulator.
- Weight words are loaded from global memory straight into the registers of the mma_v2 A operand
  (k_width-2 operand layout; logical K k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 maps to physical K 64 q + 8 w + 4 b3 +
  2 b2 + b0, so each thread's K values are 32 contiguous bytes); the decode happens in those registers, with no
  shared memory for weights.
- Activations are the B operand, loaded once per CTA at every token count. They go through an XOR-swizzled shared
  buffer read by each warp in the B-operand layout (mma_v2), at every token count (VERIFY_SMB = False). With
  VERIFY_SMB, from SMB_MIN_M tokens (verify 16/32) each K block is instead stored as a 128-byte-swizzled [BM, BK]
  tile in the logical K order; on sm_90 (H200) the GEMMs multiply it with warpgroup_mma (wgmma), decoding the
  weights in K chunks of 64 just before each chunk's wgmma (register A operand); elsewhere mma_v2 reads the tile.
- Token tiles: 8 rows for M <= 32 tokens, 16 rows otherwise (shape-based choice).
- W13's register allocation is capped at 96 from SMB_MIN_M tokens so more CTAs fit per SM (H200 measurement).
- At M <= PF_MAX_M tokens (decode 1) each GEMM double-buffers its weight and scale loads in registers: the next K
  block is requested before the current one is decoded, so a CTA has two blocks in flight.
BF16 roundings follow SGLang's path: h rounded to BF16 before SiLU, the SiLU result and the product rounded to BF16,
W2 outputs in BF16.
"""
from typing import TypedDict as _om_TypedDict
import torch as _om_torch
import triton as _om_triton
import triton.language as _om_tl
from triton.experimental import gluon as _om_gluon
import triton.experimental.gluon.language as _om_gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2 as _om_mma_v2
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared as _om_fence_async_shared, warpgroup_mma as _om_warpgroup_mma
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import _post_expert_norm_weighted_sum_kernel as _om__post_expert_norm_weighted_sum_kernel

class _om_Weights(_om_TypedDict):
    w13: _om_torch.Tensor
    w13_sf: _om_torch.Tensor
    w13_gs: _om_torch.Tensor
    w2: _om_torch.Tensor
    w2_sf: _om_torch.Tensor
    w2_gs: _om_torch.Tensor
    post_norm_w: _om_torch.Tensor

class _om_Config(_om_TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class _om_Inputs(_om_TypedDict):
    hidden_states: _om_torch.Tensor
    topk_ids: _om_torch.Tensor
    topk_weights: _om_torch.Tensor
    config: _om_Config
_om_TOP_K = 8
_om_POST_NORM_EPS = 1e-08
_om_SCALE_BIAS = _om_gl.constexpr(2.0 ** 119)
_om_ACC_BIAS = _om_gl.constexpr(2.0 ** 7)
_om_BM = 16
_om_BN = 64
_om_BK = 256
_om_NUM_WARPS = 4
_om_XS_MIN_M = 1
_om_W13_MAXNREG = {}
_om_W13_MAXNREG_BIG = 96
_om_D1_W13_SPLIT = 2
_om_D1_W2_KS = 4
_om_PF_MAX_M = 1
_om_IL_MIN_M = 128
_om_IL_M = (8,)
_om_GU_M = (16, 32)
_om_SMB_MIN_M = 128
_om_VERIFY_SMB = False
_om_MIX_MIN_M = 128
_om_W2_VERIFY_C175 = False
_om_BM_W13 = 32
_om_W13_MAXNREG_MIX = 128

@_om_triton.jit
def _om__route(ids_ptr, local_ptr, sorted_ptr, work_ptr, nwork_ptr, P, local_start, LOCAL: _om_tl.constexpr, _om_BM: _om_tl.constexpr, P_BLOCK: _om_tl.constexpr, W_MAX: _om_tl.constexpr):
    offs = _om_tl.arange(0, P_BLOCK)
    pm = offs < P
    gid = _om_tl.load(ids_ptr + offs, mask=pm, other=-1)
    lid = gid - local_start
    live = pm & (lid >= 0) & (lid < LOCAL)
    lid = _om_tl.where(live, lid, -1)
    _om_tl.store(local_ptr + offs, lid, mask=pm)
    key = _om_tl.where(live, lid * P_BLOCK + offs, LOCAL * P_BLOCK + offs)
    skey = _om_tl.sort(key)
    _om_tl.store(sorted_ptr + offs, skey % P_BLOCK, mask=pm)
    counts = _om_tl.histogram(_om_tl.maximum(lid, 0), LOCAL, mask=live)
    starts = _om_tl.cumsum(counts, 0) - counts
    chunks = (counts + _om_BM - 1) // _om_BM
    coff = _om_tl.cumsum(chunks, 0) - chunks
    total = _om_tl.sum(chunks, 0)
    e_ar = _om_tl.arange(0, LOCAL)
    w = _om_tl.arange(0, W_MAX)
    sel = (coff[None, :] <= w[:, None]) & ((coff + chunks)[None, :] > w[:, None])
    owner = _om_tl.sum(_om_tl.where(sel, e_ar[None, :], 0), 1)
    c_first = _om_tl.sum(_om_tl.where(sel, coff[None, :], 0), 1)
    e_start = _om_tl.sum(_om_tl.where(sel, starts[None, :], 0), 1)
    e_cnt = _om_tl.sum(_om_tl.where(sel, counts[None, :], 0), 1)
    r0 = e_start + (w - c_first) * _om_BM
    r1 = _om_tl.minimum(r0 + _om_BM, e_start + e_cnt)
    wm = w < total
    _om_tl.store(work_ptr + w * 3 + 0, owner, mask=wm)
    _om_tl.store(work_ptr + w * 3 + 1, r0, mask=wm)
    _om_tl.store(work_ptr + w * 3 + 2, r1, mask=wm)
    _om_tl.store(nwork_ptr, total)

@_om_triton.jit
def _om__route2(ids_ptr, local_ptr, sorted_ptr, work_ptr, nwork_ptr, work2_ptr, nwork2_ptr, P, local_start, LOCAL: _om_tl.constexpr, _om_BM: _om_tl.constexpr, BM2: _om_tl.constexpr, P_BLOCK: _om_tl.constexpr, W_MAX: _om_tl.constexpr, W_MAX2: _om_tl.constexpr):
    """_route, plus a second work list over the same sorted pairs with items of at most BM2 rows (W13's)."""
    offs = _om_tl.arange(0, P_BLOCK)
    pm = offs < P
    gid = _om_tl.load(ids_ptr + offs, mask=pm, other=-1)
    lid = gid - local_start
    live = pm & (lid >= 0) & (lid < LOCAL)
    lid = _om_tl.where(live, lid, -1)
    _om_tl.store(local_ptr + offs, lid, mask=pm)
    key = _om_tl.where(live, lid * P_BLOCK + offs, LOCAL * P_BLOCK + offs)
    skey = _om_tl.sort(key)
    _om_tl.store(sorted_ptr + offs, skey % P_BLOCK, mask=pm)
    counts = _om_tl.histogram(_om_tl.maximum(lid, 0), LOCAL, mask=live)
    starts = _om_tl.cumsum(counts, 0) - counts
    e_ar = _om_tl.arange(0, LOCAL)
    chunks = (counts + _om_BM - 1) // _om_BM
    coff = _om_tl.cumsum(chunks, 0) - chunks
    total = _om_tl.sum(chunks, 0)
    w = _om_tl.arange(0, W_MAX)
    sel = (coff[None, :] <= w[:, None]) & ((coff + chunks)[None, :] > w[:, None])
    owner = _om_tl.sum(_om_tl.where(sel, e_ar[None, :], 0), 1)
    c_first = _om_tl.sum(_om_tl.where(sel, coff[None, :], 0), 1)
    e_start = _om_tl.sum(_om_tl.where(sel, starts[None, :], 0), 1)
    e_cnt = _om_tl.sum(_om_tl.where(sel, counts[None, :], 0), 1)
    r0 = e_start + (w - c_first) * _om_BM
    r1 = _om_tl.minimum(r0 + _om_BM, e_start + e_cnt)
    wm = w < total
    _om_tl.store(work_ptr + w * 3 + 0, owner, mask=wm)
    _om_tl.store(work_ptr + w * 3 + 1, r0, mask=wm)
    _om_tl.store(work_ptr + w * 3 + 2, r1, mask=wm)
    _om_tl.store(nwork_ptr, total)
    chunks2 = (counts + BM2 - 1) // BM2
    coff2 = _om_tl.cumsum(chunks2, 0) - chunks2
    total2 = _om_tl.sum(chunks2, 0)
    w2 = _om_tl.arange(0, W_MAX2)
    sel2 = (coff2[None, :] <= w2[:, None]) & ((coff2 + chunks2)[None, :] > w2[:, None])
    owner2 = _om_tl.sum(_om_tl.where(sel2, e_ar[None, :], 0), 1)
    c_first2 = _om_tl.sum(_om_tl.where(sel2, coff2[None, :], 0), 1)
    e_start2 = _om_tl.sum(_om_tl.where(sel2, starts[None, :], 0), 1)
    e_cnt2 = _om_tl.sum(_om_tl.where(sel2, counts[None, :], 0), 1)
    q0 = e_start2 + (w2 - c_first2) * BM2
    q1 = _om_tl.minimum(q0 + BM2, e_start2 + e_cnt2)
    wm2 = w2 < total2
    _om_tl.store(work2_ptr + w2 * 3 + 0, owner2, mask=wm2)
    _om_tl.store(work2_ptr + w2 * 3 + 1, q0, mask=wm2)
    _om_tl.store(work2_ptr + w2 * 3 + 2, q1, mask=wm2)
    _om_tl.store(nwork2_ptr, total2)
_om__DECODE_ASM = _om_gl.constexpr('\n{\n.reg .b32 t, u;\nand.b32 t, $4, 0x81C081C0;\nmul.rn.bf16x2 $0, t, $6;\nshf.l.wrap.b32 t, $4, $4, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $1, t, $6;\nshf.l.wrap.b32 t, $4, $4, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $2, t, $6;\nshf.l.wrap.b32 t, $4, $4, 6;\nshf.l.wrap.b32 u, $4, $4, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $3, t, $6;\n}\n')
_om__DECODE_ASM16 = _om_gl.constexpr('\n{\n.reg .b32 t, u;\nand.b32 t, $8, 0x81C081C0;\nmul.rn.bf16x2 $0, t, $24;\nand.b32 t, $10, 0x81C081C0;\nmul.rn.bf16x2 $1, t, $25;\nshf.l.wrap.b32 t, $8, $8, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $2, t, $24;\nshf.l.wrap.b32 t, $10, $10, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $3, t, $25;\nshf.l.wrap.b32 t, $8, $8, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $4, t, $24;\nshf.l.wrap.b32 t, $10, $10, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $5, t, $25;\nshf.l.wrap.b32 t, $8, $8, 6;\nshf.l.wrap.b32 u, $8, $8, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $6, t, $24;\nshf.l.wrap.b32 t, $10, $10, 6;\nshf.l.wrap.b32 u, $10, $10, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $7, t, $25;\n}\n')
_om_MMA_L = _om_gl.constexpr(_om_gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]))
_om_A_L = _om_gl.constexpr(_om_gl.DotOperandLayout(operand_index=0, parent=_om_MMA_L.value, k_width=2))
_om_B_L = _om_gl.constexpr(_om_gl.DotOperandLayout(operand_index=1, parent=_om_MMA_L.value, k_width=2))
_om_L3 = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 8]))
_om_LW16 = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 4, 0, 0, 0]], lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]], warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 8, 2, 2, 2]))
_om_LW16C = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]], lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]], warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 2, 2, 2, 2]))
_om_LS = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 4]))
_om_LX = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[16, 4, 64]))
_om_LX2 = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[64, 64]))
_om_LX32 = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0], [16, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[32, 4, 64]))
_om_LX2_32 = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[128, 64]))
_om_LX8 = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[8, 4, 64]))
_om_LX2_8 = _om_gl.constexpr(_om_gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[32, 64]))
_om_LXG = _om_gl.constexpr(_om_gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0]))
_om_X_SMEM = _om_gl.constexpr(_om_gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]))
_om_XB_SMEM = _om_gl.constexpr(_om_gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2))
_om_MMA3_8 = _om_gl.constexpr(_om_gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 8, 16]))
_om_MMA3_16 = _om_gl.constexpr(_om_gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 16, 16]))
_om_A3_8 = _om_gl.constexpr(_om_gl.DotOperandLayout(operand_index=0, parent=_om_MMA3_8.value, k_width=2))
_om_A3_16 = _om_gl.constexpr(_om_gl.DotOperandLayout(operand_index=0, parent=_om_MMA3_16.value, k_width=2))

@_om_gluon.jit
def _om__idx3(D0: _om_gl.constexpr, D1: _om_gl.constexpr, D2: _om_gl.constexpr, L: _om_gl.constexpr):
    a = _om_gl.arange(0, D0, layout=_om_gl.SliceLayout(1, _om_gl.SliceLayout(2, L)))
    b = _om_gl.arange(0, D1, layout=_om_gl.SliceLayout(0, _om_gl.SliceLayout(2, L)))
    c = _om_gl.arange(0, D2, layout=_om_gl.SliceLayout(0, _om_gl.SliceLayout(1, L)))
    return (_om_gl.expand_dims(_om_gl.expand_dims(a, 1), 2), _om_gl.expand_dims(_om_gl.expand_dims(b, 0), 2), _om_gl.expand_dims(_om_gl.expand_dims(c, 0), 1))

@_om_gluon.jit
def _om__x8(v):
    v2 = _om_gl.join(v, v)
    v4 = _om_gl.join(v2, v2)
    return _om_gl.join(v4, v4)

@_om_gluon.jit
def _om__decode_tile(w3, sb, _om_BN: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_L3: _om_gl.constexpr, _om_A_L: _om_gl.constexpr):
    """Packed words [BN, 4, 8] + scale bytes [BN, 4, 4] -> decoded A operand [BN, BK] (logical K
    k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 <-> physical K 64 q + 8 w + 4 b3 + 2 b2 + b0)."""
    s = (sb.to(_om_gl.float8e4nv, bitcast=True).to(_om_gl.float32) * _om_SCALE_BIAS).to(_om_gl.bfloat16)
    s8 = _om_gl.reshape(_om_gl.join(s, s), [_om_BN, 4, 8])
    w16 = _om_gl.convert_layout(_om__x8(w3), _om_LW16)
    s16 = _om_gl.convert_layout(_om__x8(s8), _om_LW16)
    t = _om_gl.inline_asm_elementwise(_om__DECODE_ASM16, '=r,=r,=r,=r,=r,=r,=r,=r,' + ','.join(['r'] * 24), [w16, s16], dtype=_om_gl.bfloat16, is_pure=True, pack=16)
    t = _om_gl.reshape(_om_gl.permute(t, (0, 2, 5, 4, 1, 3)), [_om_BN, _om_BK])
    return _om_gl.convert_layout(t, _om_A_L, assert_trivial=True)

@_om_gluon.jit
def _om__wsplit4(w3, _om_BN: _om_gl.constexpr):
    """Packed words [BN, 4, 8] (word w = 4 c1 + 2 c0 + wl) -> the four K chunks c = 2 c1 + c0, each [BN, 4, 2]."""
    lo, hi = _om_gl.split(_om_gl.permute(_om_gl.reshape(w3, [_om_BN, 4, 2, 2, 2]), (0, 1, 4, 2, 3)))
    c0, c2 = _om_gl.split(lo)
    c1, c3 = _om_gl.split(hi)
    return (c0, c1, c2, c3)

@_om_gluon.jit
def _om__ssplit4(sb, _om_BN: _om_gl.constexpr):
    """Scale bytes [BN, 4, 4] (group g = chunk c = 2 c1 + c0) -> four [BN, 4, 1]."""
    lo, hi = _om_gl.split(_om_gl.reshape(sb, [_om_BN, 4, 2, 2]))
    c0, c2 = _om_gl.split(lo)
    c1, c3 = _om_gl.split(hi)
    return (_om_gl.expand_dims(c0, 2), _om_gl.expand_dims(c1, 2), _om_gl.expand_dims(c2, 2), _om_gl.expand_dims(c3, 2))

@_om_gluon.jit
def _om__mma_chunk(wc, sc, xb_s, C: _om_gl.constexpr, acc, _om_BN: _om_gl.constexpr, a_l: _om_gl.constexpr, HOP: _om_gl.constexpr):
    """Decode K chunk C (64 logical K) and multiply it with the matching 64 columns of the shared activation tile."""
    s = (sc.to(_om_gl.float8e4nv, bitcast=True).to(_om_gl.float32) * _om_SCALE_BIAS).to(_om_gl.bfloat16)
    s2 = _om_gl.reshape(_om_gl.join(s, s), [_om_BN, 4, 2])
    w16 = _om_gl.convert_layout(_om__x8(wc), _om_LW16C)
    s16 = _om_gl.convert_layout(_om__x8(s2), _om_LW16C)
    t = _om_gl.inline_asm_elementwise(_om__DECODE_ASM16, '=r,=r,=r,=r,=r,=r,=r,=r,' + ','.join(['r'] * 24), [w16, s16], dtype=_om_gl.bfloat16, is_pure=True, pack=16)
    a = _om_gl.convert_layout(_om_gl.reshape(_om_gl.permute(t, (0, 2, 5, 4, 1, 3)), [_om_BN, 64]), a_l)
    bv = xb_s.slice(64 * C, 64, dim=1).permute((1, 0))
    if HOP:
        acc = _om_warpgroup_mma(a, bv, acc)
    else:
        acc = _om_mma_v2(a, bv.load(_om_B_L), acc)
    return acc

@_om_gluon.jit
def _om__xg_offsets(sorted_ptr, r0, r1, K: _om_gl.constexpr, BY_TOKEN: _om_gl.constexpr, _om_BM: _om_gl.constexpr):
    """Offsets/mask [4 BM, 64] in LXG for the cooperative load: row 4n + q, column c -> x[row_n, 64q + c]."""
    rr = _om_gl.arange(0, 4 * _om_BM, layout=_om_gl.SliceLayout(1, _om_LXG))
    cc = _om_gl.arange(0, 64, layout=_om_gl.SliceLayout(0, _om_LXG))
    rv = r0 + rr // 4
    pair = _om_gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    row = pair // 8 if BY_TOKEN else pair
    off = _om_gl.expand_dims(row * K + rr % 4 * 64, 1) + _om_gl.expand_dims(cc, 0)
    msk = _om_gl.expand_dims(rv < r1, 1) & _om_gl.expand_dims(cc >= 0, 0)
    return (off, msk)

@_om_gluon.jit
def _om__x_shared(x_ptr, off, msk, k0, x_s, _om_BM: _om_gl.constexpr, _om_BK: _om_gl.constexpr):
    xg = _om_gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    _om_gl.barrier()
    x_s.store(xg)
    _om_gl.barrier()
    if _om_BM == 8:
        xv = _om_gl.reshape(x_s.load(_om_LX2_8), [_om_BM, 4, 64])
    elif _om_BM == 32:
        xv = _om_gl.reshape(x_s.load(_om_LX2_32), [_om_BM, 4, 64])
    else:
        xv = _om_gl.reshape(x_s.load(_om_LX2), [_om_BM, 4, 64])
    return _om__x_tile(xv, _om_BM, _om_BK, _om_B_L)

@_om_gluon.jit
def _om__xb_store(x_ptr, off, msk, k0, xb_s, _om_BM: _om_gl.constexpr, _om_BK: _om_gl.constexpr):
    """Cooperative load of one K block of activations, stored to xb_s as [n, k' = 32 w + 16 b3 + 8 b2 + 2 q + b0]."""
    xg = _om_gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    xl = _om_gl.reshape(_om_gl.permute(_om_gl.reshape(xg, [_om_BM, 4, 8, 2, 2, 2]), (0, 2, 3, 4, 1, 5)), [_om_BM, _om_BK])
    _om_gl.barrier()
    xb_s.store(xl)
    _om_fence_async_shared()
    _om_gl.barrier()

@_om_gluon.jit
def _om__x_tile(xv, _om_BM: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_B_L: _om_gl.constexpr):
    xb = _om_gl.reshape(_om_gl.permute(_om_gl.reshape(xv, [_om_BM, 4, 8, 2, 2, 2]), (2, 3, 4, 1, 5, 0)), [_om_BK, _om_BM])
    return _om_gl.convert_layout(xb, _om_B_L, assert_trivial=_om_BM != 32)

@_om_gluon.jit
def _om__w13_silu(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, K: _om_gl.constexpr, I: _om_gl.constexpr, _om_BN: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_BM: _om_gl.constexpr, XS: _om_gl.constexpr, SMB: _om_gl.constexpr=False, HOP: _om_gl.constexpr=False, CHK: _om_gl.constexpr=False, PF: _om_gl.constexpr=False, DIRECT: _om_gl.constexpr=False, NLOC: _om_gl.constexpr=32, IL: _om_gl.constexpr=False, SPLIT: _om_gl.constexpr=1):
    mma: _om_gl.constexpr = (_om_MMA3_8 if _om_BM == 8 else _om_MMA3_16) if HOP else _om_MMA_L
    a_l: _om_gl.constexpr = (_om_A3_8 if _om_BM == 8 else _om_A3_16) if HOP else _om_A_L
    pid_n = _om_gl.program_id(0)
    pid_w = _om_gl.program_id(1)
    KS: _om_gl.constexpr = K // SPLIT
    pid_k = _om_gl.program_id(2) if SPLIT > 1 else 0
    kb = pid_k * KS
    if DIRECT:
        e = _om_gl.load(work_ptr + pid_w) - _om_gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(_om_gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(_om_gl.int32)
        if pid_n + pid_k == 0:
            _om_gl.store(nwork_ptr + 1 + pid_w, e + active.to(_om_gl.int32) - 1)
    else:
        active = pid_w < _om_gl.load(nwork_ptr)
        e = _om_gl.load(work_ptr + pid_w * 3 + 0)
        r0 = _om_gl.load(work_ptr + pid_w * 3 + 1)
        r1 = _om_gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * _om_BN
        rw, qw, jw = _om__idx3(_om_BN, 4, 8, _om_L3)
        KB: _om_gl.constexpr = K // _om_BK
        BLKW: _om_gl.constexpr = _om_BN * 36 if IL else _om_BN * 32
        BLKB: _om_gl.constexpr = _om_BN * 144 if IL else _om_BN * 16
        S0: _om_gl.constexpr = _om_BN * 128 if IL else 0
        w_off = (e * (2 * I // _om_BN) + pid_n) * KB * BLKW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _om__idx3(_om_BN, 4, 4, _om_LS)
        s_off = (e * (2 * I // _om_BN) + pid_n) * KB * BLKB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _om__idx3(_om_BM, 4, 64, _om_LX8 if _om_BM == 8 else _om_LX32 if _om_BM == 32 else _om_LX)
        rv = r0 + mx
        pair = _om_gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = pair // 8 * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _om__xg_offsets(sorted_ptr, r0, r1, K, True, _om_BM)
        x_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [4 * _om_BM, 64], _om_X_SMEM)
        xb_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [_om_BM, _om_BK], _om_XB_SMEM)
        wp = w_ptr.to(_om_gl.pointer_type(_om_gl.int32))
        acc_g = _om_gl.zeros([_om_BN, _om_BM], _om_gl.float32, layout=mma)
        acc_u = _om_gl.zeros([_om_BN, _om_BM], _om_gl.float32, layout=mma)
        if PF:
            wg_n = _om_gl.load(wp + w_off + kb // _om_BK * BLKW)
            sg_n = _om_gl.load(sf_ptr + s_off + kb // _om_BK * BLKB)
            wu_n = _om_gl.load(wp + w_off + I // _om_BN * KB * BLKW + kb // _om_BK * BLKW)
            su_n = _om_gl.load(sf_ptr + s_off + I // _om_BN * KB * BLKB + kb // _om_BK * BLKB)
        for k0 in range(kb, kb + KS, _om_BK):
            if PF:
                wg = wg_n
                sg = sg_n
                wu = wu_n
                su = su_n
                kn = k0 + _om_BK
                wm = (rw >= 0) & (kn < kb + KS)
                sm = (rs >= 0) & (kn < kb + KS)
                wg_n = _om_gl.load(wp + w_off + kn // _om_BK * BLKW, mask=wm, other=0)
                sg_n = _om_gl.load(sf_ptr + s_off + kn // _om_BK * BLKB, mask=sm)
                wu_n = _om_gl.load(wp + w_off + I // _om_BN * KB * BLKW + kn // _om_BK * BLKW, mask=wm, other=0)
                su_n = _om_gl.load(sf_ptr + s_off + I // _om_BN * KB * BLKB + kn // _om_BK * BLKB, mask=sm)
            else:
                wg = _om_gl.load(wp + w_off + k0 // _om_BK * BLKW)
                sg = _om_gl.load(sf_ptr + s_off + k0 // _om_BK * BLKB)
                wu = _om_gl.load(wp + w_off + I // _om_BN * KB * BLKW + k0 // _om_BK * BLKW)
                su = _om_gl.load(sf_ptr + s_off + I // _om_BN * KB * BLKB + k0 // _om_BK * BLKB)
            if SMB:
                _om__xb_store(x_ptr, xoff, xmsk, k0, xb_s, _om_BM, _om_BK)
                if CHK:
                    g0, g1, g2, g3 = _om__wsplit4(wg, _om_BN)
                    h0, h1, h2, h3 = _om__ssplit4(sg, _om_BN)
                    acc_g = _om__mma_chunk(g0, h0, xb_s, 0, acc_g, _om_BN, a_l, HOP)
                    acc_g = _om__mma_chunk(g1, h1, xb_s, 1, acc_g, _om_BN, a_l, HOP)
                    acc_g = _om__mma_chunk(g2, h2, xb_s, 2, acc_g, _om_BN, a_l, HOP)
                    acc_g = _om__mma_chunk(g3, h3, xb_s, 3, acc_g, _om_BN, a_l, HOP)
                    u0, u1, u2, u3 = _om__wsplit4(wu, _om_BN)
                    v0, v1, v2, v3 = _om__ssplit4(su, _om_BN)
                    acc_u = _om__mma_chunk(u0, v0, xb_s, 0, acc_u, _om_BN, a_l, HOP)
                    acc_u = _om__mma_chunk(u1, v1, xb_s, 1, acc_u, _om_BN, a_l, HOP)
                    acc_u = _om__mma_chunk(u2, v2, xb_s, 2, acc_u, _om_BN, a_l, HOP)
                    acc_u = _om__mma_chunk(u3, v3, xb_s, 3, acc_u, _om_BN, a_l, HOP)
                elif HOP:
                    acc_g = _om_warpgroup_mma(_om__decode_tile(wg, sg, _om_BN, _om_BK, _om_L3, a_l), xb_s.permute((1, 0)), acc_g)
                    acc_u = _om_warpgroup_mma(_om__decode_tile(wu, su, _om_BN, _om_BK, _om_L3, a_l), xb_s.permute((1, 0)), acc_u)
                else:
                    xb = xb_s.permute((1, 0)).load(_om_B_L)
                    acc_g = _om_mma_v2(_om__decode_tile(wg, sg, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc_g)
                    acc_u = _om_mma_v2(_om__decode_tile(wu, su, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc_u)
            else:
                if XS:
                    xb = _om__x_shared(x_ptr, xoff, xmsk, k0, x_s, _om_BM, _om_BK)
                else:
                    xb = _om__x_tile(_om_gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), _om_BM, _om_BK, _om_B_L)
                acc_g = _om_mma_v2(_om__decode_tile(wg, sg, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc_g)
                acc_u = _om_mma_v2(_om__decode_tile(wu, su, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc_u)
        n_o = _om_gl.arange(0, _om_BN, layout=_om_gl.SliceLayout(1, mma))
        m_o = _om_gl.arange(0, _om_BM, layout=_om_gl.SliceLayout(0, mma))
        ro = r0 + m_o
        pair_o = _om_gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        omask = _om_gl.expand_dims(ro < r1, 0) & _om_gl.expand_dims(n_o >= 0, 1)
        if SPLIT > 1:
            pptr = out_ptr + pid_k * (8 * 2 * I) + _om_gl.expand_dims(pair_o, 0) * (2 * I) + _om_gl.expand_dims(n0 + n_o, 1)
            _om_gl.store(pptr, acc_g, mask=omask)
            _om_gl.store(pptr + I, acc_u, mask=omask)
        else:
            g = _om_gl.load(gs_ptr + e) * _om_ACC_BIAS
            hg = (acc_g * g).to(_om_gl.bfloat16).to(_om_gl.float32)
            hu = (acc_u * g).to(_om_gl.bfloat16).to(_om_gl.float32)
            inter = (hg / (1.0 + _om_gl.exp(-hg))).to(_om_gl.bfloat16).to(_om_gl.float32) * hu
            optr = out_ptr + _om_gl.expand_dims(pair_o, 0) * I + _om_gl.expand_dims(n0 + n_o, 1)
            _om_gl.store(optr, inter.to(_om_gl.bfloat16), mask=omask)

@_om_gluon.jit
def _om__gu_split(t, _om_BN: _om_gl.constexpr, _om_BM: _om_gl.constexpr):
    """[BN, BM] with rows 16 s + 8 g + i -> (g = 0 part, g = 1 part), each [BN / 16, 8, BM] (row 8 apart: one thread)."""
    return _om_gl.split(_om_gl.permute(_om_gl.reshape(t, [_om_BN // 16, 2, 8, _om_BM]), (0, 2, 3, 1)))

@_om_gluon.jit
def _om__w2(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, K: _om_gl.constexpr, N: _om_gl.constexpr, _om_BN: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_BM: _om_gl.constexpr, XS: _om_gl.constexpr, SMB: _om_gl.constexpr=False, HOP: _om_gl.constexpr=False, CHK: _om_gl.constexpr=False, PF: _om_gl.constexpr=False, DIRECT: _om_gl.constexpr=False, NLOC: _om_gl.constexpr=32, GU: _om_gl.constexpr=False, IL: _om_gl.constexpr=False):
    mma: _om_gl.constexpr = (_om_MMA3_8 if _om_BM == 8 else _om_MMA3_16) if HOP else _om_MMA_L
    a_l: _om_gl.constexpr = (_om_A3_8 if _om_BM == 8 else _om_A3_16) if HOP else _om_A_L
    pid_n = _om_gl.program_id(0)
    pid_w = _om_gl.program_id(1)
    if DIRECT:
        e = _om_gl.load(work_ptr + pid_w) - _om_gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(_om_gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(_om_gl.int32)
        if GU:
            if pid_n == 0:
                _om_gl.store(nwork_ptr + 1 + pid_w, e + active.to(_om_gl.int32) - 1)
    else:
        active = pid_w < _om_gl.load(nwork_ptr)
        e = _om_gl.load(work_ptr + pid_w * 3 + 0)
        r0 = _om_gl.load(work_ptr + pid_w * 3 + 1)
        r1 = _om_gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * _om_BN
        rw, qw, jw = _om__idx3(_om_BN, 4, 8, _om_L3)
        KB: _om_gl.constexpr = K // _om_BK
        BLKW: _om_gl.constexpr = _om_BN * 36 if IL else _om_BN * 32
        BLKB: _om_gl.constexpr = _om_BN * 144 if IL else _om_BN * 16
        S0: _om_gl.constexpr = _om_BN * 128 if IL else 0
        w_off = (e * (N // _om_BN) + pid_n) * KB * BLKW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _om__idx3(_om_BN, 4, 4, _om_LS)
        s_off = (e * (N // _om_BN) + pid_n) * KB * BLKB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _om__idx3(_om_BM, 4, 64, _om_LX8 if _om_BM == 8 else _om_LX32 if _om_BM == 32 else _om_LX)
        rv = r0 + mx
        pair = _om_gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = (pair // 8 if GU else pair) * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _om__xg_offsets(sorted_ptr, r0, r1, K, GU, _om_BM)
        x_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [4 * _om_BM, 64], _om_X_SMEM)
        xb_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [_om_BM, _om_BK], _om_XB_SMEM)
        wp = w_ptr.to(_om_gl.pointer_type(_om_gl.int32))
        acc = _om_gl.zeros([_om_BN, _om_BM], _om_gl.float32, layout=mma)
        if PF:
            wq_n = _om_gl.load(wp + w_off)
            sq_n = _om_gl.load(sf_ptr + s_off)
        for k0 in range(0, K, _om_BK):
            if PF:
                wq = wq_n
                sq = sq_n
                kn = k0 + _om_BK
                wq_n = _om_gl.load(wp + w_off + kn // _om_BK * BLKW, mask=(rw >= 0) & (kn < K), other=0)
                sq_n = _om_gl.load(sf_ptr + s_off + kn // _om_BK * BLKB, mask=(rs >= 0) & (kn < K))
            else:
                wq = _om_gl.load(wp + w_off + k0 // _om_BK * BLKW)
                sq = _om_gl.load(sf_ptr + s_off + k0 // _om_BK * BLKB)
            if SMB:
                _om__xb_store(a_ptr, xoff, xmsk, k0, xb_s, _om_BM, _om_BK)
                if CHK:
                    q0, q1, q2, q3 = _om__wsplit4(wq, _om_BN)
                    p0, p1, p2, p3 = _om__ssplit4(sq, _om_BN)
                    acc = _om__mma_chunk(q0, p0, xb_s, 0, acc, _om_BN, a_l, HOP)
                    acc = _om__mma_chunk(q1, p1, xb_s, 1, acc, _om_BN, a_l, HOP)
                    acc = _om__mma_chunk(q2, p2, xb_s, 2, acc, _om_BN, a_l, HOP)
                    acc = _om__mma_chunk(q3, p3, xb_s, 3, acc, _om_BN, a_l, HOP)
                elif HOP:
                    acc = _om_warpgroup_mma(_om__decode_tile(wq, sq, _om_BN, _om_BK, _om_L3, a_l), xb_s.permute((1, 0)), acc)
                else:
                    acc = _om_mma_v2(_om__decode_tile(wq, sq, _om_BN, _om_BK, _om_L3, _om_A_L), xb_s.permute((1, 0)).load(_om_B_L), acc)
            else:
                if XS:
                    xb = _om__x_shared(a_ptr, xoff, xmsk, k0, x_s, _om_BM, _om_BK)
                else:
                    xb = _om__x_tile(_om_gl.load(a_ptr + x_off + k0, mask=x_mask, other=0.0), _om_BM, _om_BK, _om_B_L)
                acc = _om_mma_v2(_om__decode_tile(wq, sq, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc)
        y = acc * (_om_gl.load(gs_ptr + e) * _om_ACC_BIAS)
        n_o = _om_gl.arange(0, _om_BN, layout=_om_gl.SliceLayout(1, mma))
        m_o = _om_gl.arange(0, _om_BM, layout=_om_gl.SliceLayout(0, mma))
        ro = r0 + m_o
        pair_o = _om_gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        if GU:
            rows2 = _om_gl.expand_dims(n_o, 1) + _om_gl.expand_dims(m_o * 0, 0)
            pair2 = _om_gl.expand_dims(pair_o, 0) + _om_gl.expand_dims(n_o * 0, 1)
            msk2 = _om_gl.expand_dims(ro < r1, 0) & _om_gl.expand_dims(n_o >= 0, 1)
            hg, hu = _om__gu_split(y, _om_BN, _om_BM)
            rg, _ru = _om__gu_split(rows2, _om_BN, _om_BM)
            pg, _pu = _om__gu_split(pair2, _om_BN, _om_BM)
            mg, _mu = _om__gu_split(msk2.to(_om_gl.int32), _om_BN, _om_BM)
            hg = hg.to(_om_gl.bfloat16).to(_om_gl.float32)
            hu = hu.to(_om_gl.bfloat16).to(_om_gl.float32)
            v = (hg / (1.0 + _om_gl.exp(-hg))).to(_om_gl.bfloat16).to(_om_gl.float32) * hu
            col = pid_n * (_om_BN // 2) + rg // 16 * 8 + rg % 8
            _om_gl.store(out_ptr + pg * (N // 2) + col, v.to(_om_gl.bfloat16), mask=mg != 0)
        else:
            optr = out_ptr + _om_gl.expand_dims(pair_o, 0) * N + _om_gl.expand_dims(n0 + n_o, 1)
            _om_gl.store(optr, y.to(_om_gl.bfloat16), mask=_om_gl.expand_dims(ro < r1, 0) & _om_gl.expand_dims(n_o >= 0, 1))
    if DIRECT and (not GU):
        if r1 == r0:
            zl: _om_gl.constexpr = _om_gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
            zo = _om_gl.arange(0, _om_BN, layout=zl)
            _om_gl.store(out_ptr + pid_w * N + pid_n * _om_BN + zo, _om_gl.zeros([_om_BN], _om_gl.bfloat16, layout=zl))

@_om_gluon.jit
def _om__w2_ks(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, K: _om_gl.constexpr, N: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_BM: _om_gl.constexpr, KS: _om_gl.constexpr, PF: _om_gl.constexpr, NLOC: _om_gl.constexpr=32):
    _om_BN: _om_gl.constexpr = 64
    pid_n = _om_gl.program_id(0)
    pid_w = _om_gl.program_id(1)
    pid_k = _om_gl.program_id(2)
    e = _om_gl.load(work_ptr + pid_w) - _om_gl.load(nwork_ptr)
    active = (e >= 0) & (e < NLOC)
    e = e * active.to(_om_gl.int32)
    r0 = pid_w
    r1 = pid_w + active.to(_om_gl.int32)
    if active:
        KB: _om_gl.constexpr = K // _om_BK
        KSB: _om_gl.constexpr = KB // KS
        kb0 = pid_k * KSB
        rw, qw, jw = _om__idx3(_om_BN, 4, 8, _om_L3)
        w_off = (e * (N // _om_BN) + pid_n) * KB * (_om_BN * 32) + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _om__idx3(_om_BN, 4, 4, _om_LS)
        s_off = (e * (N // _om_BN) + pid_n) * KB * (_om_BN * 16) + rs * 16 + qs * 4 + gs_
        xoff, xmsk = _om__xg_offsets(sorted_ptr, r0, r1, K, False, _om_BM)
        x_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [4 * _om_BM, 64], _om_X_SMEM)
        wp = w_ptr.to(_om_gl.pointer_type(_om_gl.int32))
        acc = _om_gl.zeros([_om_BN, _om_BM], _om_gl.float32, layout=_om_MMA_L)
        if PF:
            wq_n = _om_gl.load(wp + w_off + kb0 * (_om_BN * 32))
            sq_n = _om_gl.load(sf_ptr + s_off + kb0 * (_om_BN * 16))
        for j in range(0, KSB):
            kb = kb0 + j
            if PF:
                wq = wq_n
                sq = sq_n
                more = j + 1 < KSB
                wq_n = _om_gl.load(wp + w_off + (kb + 1) * (_om_BN * 32), mask=(rw >= 0) & more, other=0)
                sq_n = _om_gl.load(sf_ptr + s_off + (kb + 1) * (_om_BN * 16), mask=(rs >= 0) & more)
            else:
                wq = _om_gl.load(wp + w_off + kb * (_om_BN * 32))
                sq = _om_gl.load(sf_ptr + s_off + kb * (_om_BN * 16))
            xb = _om__x_shared(a_ptr, xoff, xmsk, kb * _om_BK, x_s, _om_BM, _om_BK)
            acc = _om_mma_v2(_om__decode_tile(wq, sq, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc)
        y = acc * (_om_gl.load(gs_ptr + e) * _om_ACC_BIAS)
        n_o = _om_gl.arange(0, _om_BN, layout=_om_gl.SliceLayout(1, _om_MMA_L))
        m_o = _om_gl.arange(0, _om_BM, layout=_om_gl.SliceLayout(0, _om_MMA_L))
        ro = r0 + m_o
        pair_o = _om_gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        optr = out_ptr + pid_k * (8 * N) + _om_gl.expand_dims(pair_o, 0) * N + _om_gl.expand_dims(pid_n * _om_BN + n_o, 1)
        _om_gl.store(optr, y, mask=_om_gl.expand_dims(ro < r1, 0) & _om_gl.expand_dims(n_o >= 0, 1))

@_om_triton.jit
def _om__wsum_ks(Y, NW, TW, OUT, IDS, eps, H: _om_tl.constexpr, KS: _om_tl.constexpr, TOPK: _om_tl.constexpr, FP32: _om_tl.constexpr):
    """SGLang's _post_expert_norm_weighted_sum_kernel (APPLY_NORM, HAS_WEIGHT, WEIGHT_FIRST, HAS_IDS; same math,
    order and rsqrt), with one change: FP32 means Y holds KS fp32 partial rows per slot, summed and rounded to BF16
    (as W2's BF16 output is) before the unchanged norm."""
    t = _om_tl.program_id(0)
    offs = _om_tl.arange(0, H)
    w = _om_tl.load(NW + offs).to(_om_tl.float32)
    acc = _om_tl.zeros([H], dtype=_om_tl.float32)
    for k in _om_tl.static_range(TOPK):
        live = _om_tl.load(IDS + t * TOPK + k) >= 0
        if live:
            if FP32:
                x = _om_tl.zeros([H], _om_tl.float32)
                for s in _om_tl.static_range(KS):
                    x += _om_tl.load(Y + s * (TOPK * H) + (t * TOPK + k) * H + offs)
                x = x.to(_om_tl.bfloat16).to(_om_tl.float32)
            else:
                x = _om_tl.load(Y + (t * TOPK + k) * H + offs).to(_om_tl.float32)
            y = x * w
            var = _om_tl.sum(y * y, axis=0) / H
            y = y * _om_tl.rsqrt(var + eps)
            acc += y * _om_tl.load(TW + t * TOPK + k).to(_om_tl.float32)
    _om_tl.store(OUT + t * H + offs, acc.to(OUT.dtype.element_ty))

@_om_triton.jit
def _om__d1_reduce(P, GS, IDS, INTER, I: _om_tl.constexpr, S: _om_tl.constexpr, BLOCK: _om_tl.constexpr, NLOC: _om_tl.constexpr):
    """M = 1: inter[slot] = SiLU(gate) * up from W13's S fp32 partials [S, 8, 2I] of each local slot, with
    W13's own scale and roundings. Program (slot, column block); non-local slots return."""
    slot = _om_tl.program_id(0)
    lid = _om_tl.load(IDS + slot)
    if (lid >= 0) & (lid < NLOC):
        c = _om_tl.program_id(1) * BLOCK + _om_tl.arange(0, BLOCK)
        ga = _om_tl.zeros([BLOCK], _om_tl.float32)
        ua = _om_tl.zeros([BLOCK], _om_tl.float32)
        for s in _om_tl.static_range(S):
            ga += _om_tl.load(P + s * (8 * 2 * I) + slot * (2 * I) + c)
            ua += _om_tl.load(P + s * (8 * 2 * I) + slot * (2 * I) + I + c)
        g = _om_tl.load(GS + lid) * 128.0
        hg = (ga * g).to(_om_tl.bfloat16).to(_om_tl.float32)
        hu = (ua * g).to(_om_tl.bfloat16).to(_om_tl.float32)
        v = (hg / (1.0 + _om_tl.exp(-hg))).to(_om_tl.bfloat16).to(_om_tl.float32) * hu
        _om_tl.store(INTER + slot * I + c, v.to(_om_tl.bfloat16))

@_om_gluon.jit
def _om__w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n, K: _om_gl.constexpr, I: _om_gl.constexpr, _om_BN: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_BM: _om_gl.constexpr, XS: _om_gl.constexpr, SMB: _om_gl.constexpr, HOP: _om_gl.constexpr, CHK: _om_gl.constexpr, PF: _om_gl.constexpr):
    mma: _om_gl.constexpr = (_om_MMA3_8 if _om_BM == 8 else _om_MMA3_16) if HOP else _om_MMA_L
    a_l: _om_gl.constexpr = (_om_A3_8 if _om_BM == 8 else _om_A3_16) if HOP else _om_A_L
    n0 = pid_n * _om_BN
    rw, qw, jw = _om__idx3(_om_BN, 4, 8, _om_L3)
    KB: _om_gl.constexpr = K // _om_BK
    w_off = (e * (2 * I // _om_BN) + pid_n) * KB * (_om_BN * 32) + rw * 32 + qw * 8 + jw
    rs, qs, gs_ = _om__idx3(_om_BN, 4, 4, _om_LS)
    s_off = (e * (2 * I // _om_BN) + pid_n) * KB * (_om_BN * 16) + rs * 16 + qs * 4 + gs_
    mx, qx, cx = _om__idx3(_om_BM, 4, 64, _om_LX8 if _om_BM == 8 else _om_LX32 if _om_BM == 32 else _om_LX)
    rv = r0 + mx
    pair = _om_gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    x_off = pair // 8 * K + qx * 64 + cx
    x_mask = (rv < r1) & (cx >= 0)
    xoff, xmsk = _om__xg_offsets(sorted_ptr, r0, r1, K, True, _om_BM)
    x_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [4 * _om_BM, 64], _om_X_SMEM)
    xb_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [_om_BM, _om_BK], _om_XB_SMEM)
    wp = w_ptr.to(_om_gl.pointer_type(_om_gl.int32))
    acc_g = _om_gl.zeros([_om_BN, _om_BM], _om_gl.float32, layout=mma)
    acc_u = _om_gl.zeros([_om_BN, _om_BM], _om_gl.float32, layout=mma)
    if PF:
        wg_n = _om_gl.load(wp + w_off)
        sg_n = _om_gl.load(sf_ptr + s_off)
        wu_n = _om_gl.load(wp + w_off + I * (K // 8))
        su_n = _om_gl.load(sf_ptr + s_off + I * (K // 16))
    for k0 in range(0, K, _om_BK):
        if PF:
            wg = wg_n
            sg = sg_n
            wu = wu_n
            su = su_n
            kn = k0 + _om_BK
            wm = (rw >= 0) & (kn < K)
            sm = (rs >= 0) & (kn < K)
            wg_n = _om_gl.load(wp + w_off + kn // _om_BK * (_om_BN * 32), mask=wm, other=0)
            sg_n = _om_gl.load(sf_ptr + s_off + kn // _om_BK * (_om_BN * 16), mask=sm)
            wu_n = _om_gl.load(wp + w_off + I * (K // 8) + kn // _om_BK * (_om_BN * 32), mask=wm, other=0)
            su_n = _om_gl.load(sf_ptr + s_off + I * (K // 16) + kn // _om_BK * (_om_BN * 16), mask=sm)
        else:
            wg = _om_gl.load(wp + w_off + k0 // _om_BK * (_om_BN * 32))
            sg = _om_gl.load(sf_ptr + s_off + k0 // _om_BK * (_om_BN * 16))
            wu = _om_gl.load(wp + w_off + I * (K // 8) + k0 // _om_BK * (_om_BN * 32))
            su = _om_gl.load(sf_ptr + s_off + I * (K // 16) + k0 // _om_BK * (_om_BN * 16))
        if SMB:
            _om__xb_store(x_ptr, xoff, xmsk, k0, xb_s, _om_BM, _om_BK)
            if CHK:
                g0, g1, g2, g3 = _om__wsplit4(wg, _om_BN)
                h0, h1, h2, h3 = _om__ssplit4(sg, _om_BN)
                acc_g = _om__mma_chunk(g0, h0, xb_s, 0, acc_g, _om_BN, a_l, HOP)
                acc_g = _om__mma_chunk(g1, h1, xb_s, 1, acc_g, _om_BN, a_l, HOP)
                acc_g = _om__mma_chunk(g2, h2, xb_s, 2, acc_g, _om_BN, a_l, HOP)
                acc_g = _om__mma_chunk(g3, h3, xb_s, 3, acc_g, _om_BN, a_l, HOP)
                u0, u1, u2, u3 = _om__wsplit4(wu, _om_BN)
                v0, v1, v2, v3 = _om__ssplit4(su, _om_BN)
                acc_u = _om__mma_chunk(u0, v0, xb_s, 0, acc_u, _om_BN, a_l, HOP)
                acc_u = _om__mma_chunk(u1, v1, xb_s, 1, acc_u, _om_BN, a_l, HOP)
                acc_u = _om__mma_chunk(u2, v2, xb_s, 2, acc_u, _om_BN, a_l, HOP)
                acc_u = _om__mma_chunk(u3, v3, xb_s, 3, acc_u, _om_BN, a_l, HOP)
            elif HOP:
                acc_g = _om_warpgroup_mma(_om__decode_tile(wg, sg, _om_BN, _om_BK, _om_L3, a_l), xb_s.permute((1, 0)), acc_g)
                acc_u = _om_warpgroup_mma(_om__decode_tile(wu, su, _om_BN, _om_BK, _om_L3, a_l), xb_s.permute((1, 0)), acc_u)
            else:
                xb = xb_s.permute((1, 0)).load(_om_B_L)
                acc_g = _om_mma_v2(_om__decode_tile(wg, sg, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc_g)
                acc_u = _om_mma_v2(_om__decode_tile(wu, su, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc_u)
        else:
            if XS:
                xb = _om__x_shared(x_ptr, xoff, xmsk, k0, x_s, _om_BM, _om_BK)
            else:
                xb = _om__x_tile(_om_gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), _om_BM, _om_BK, _om_B_L)
            acc_g = _om_mma_v2(_om__decode_tile(wg, sg, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc_g)
            acc_u = _om_mma_v2(_om__decode_tile(wu, su, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc_u)
    g = _om_gl.load(gs_ptr + e) * _om_ACC_BIAS
    hg = (acc_g * g).to(_om_gl.bfloat16).to(_om_gl.float32)
    hu = (acc_u * g).to(_om_gl.bfloat16).to(_om_gl.float32)
    inter = (hg / (1.0 + _om_gl.exp(-hg))).to(_om_gl.bfloat16).to(_om_gl.float32) * hu
    n_o = _om_gl.arange(0, _om_BN, layout=_om_gl.SliceLayout(1, mma))
    m_o = _om_gl.arange(0, _om_BM, layout=_om_gl.SliceLayout(0, mma))
    ro = r0 + m_o
    pair_o = _om_gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
    optr = out_ptr + _om_gl.expand_dims(pair_o, 0) * I + _om_gl.expand_dims(n0 + n_o, 1)
    _om_gl.store(optr, inter.to(_om_gl.bfloat16), mask=_om_gl.expand_dims(ro < r1, 0) & _om_gl.expand_dims(n_o >= 0, 1))

@_om_gluon.jit
def _om__w13_mix(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, K: _om_gl.constexpr, I: _om_gl.constexpr, _om_BN: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_BM: _om_gl.constexpr, XS: _om_gl.constexpr, SMB: _om_gl.constexpr=False, HOP: _om_gl.constexpr=False, CHK: _om_gl.constexpr=False, PF: _om_gl.constexpr=False):
    pid_n = _om_gl.program_id(0)
    pid_w = _om_gl.program_id(1)
    active = pid_w < _om_gl.load(nwork_ptr)
    e = _om_gl.load(work_ptr + pid_w * 3 + 0)
    r0 = _om_gl.load(work_ptr + pid_w * 3 + 1)
    r1 = _om_gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        if r1 - r0 <= 8:
            _om__w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n, K=K, I=I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=8, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        elif _om_BM == 32:
            if r1 - r0 <= 16:
                _om__w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n, K=K, I=I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=16, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
            else:
                _om__w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n, K=K, I=I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=32, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        else:
            _om__w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n, K=K, I=I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=_om_BM, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)

@_om_gluon.jit
def _om__w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n, K: _om_gl.constexpr, N: _om_gl.constexpr, _om_BN: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_BM: _om_gl.constexpr, XS: _om_gl.constexpr, SMB: _om_gl.constexpr, HOP: _om_gl.constexpr, CHK: _om_gl.constexpr, PF: _om_gl.constexpr):
    mma: _om_gl.constexpr = (_om_MMA3_8 if _om_BM == 8 else _om_MMA3_16) if HOP else _om_MMA_L
    a_l: _om_gl.constexpr = (_om_A3_8 if _om_BM == 8 else _om_A3_16) if HOP else _om_A_L
    n0 = pid_n * _om_BN
    rw, qw, jw = _om__idx3(_om_BN, 4, 8, _om_L3)
    KB: _om_gl.constexpr = K // _om_BK
    w_off = (e * (N // _om_BN) + pid_n) * KB * (_om_BN * 32) + rw * 32 + qw * 8 + jw
    rs, qs, gs_ = _om__idx3(_om_BN, 4, 4, _om_LS)
    s_off = (e * (N // _om_BN) + pid_n) * KB * (_om_BN * 16) + rs * 16 + qs * 4 + gs_
    mx, qx, cx = _om__idx3(_om_BM, 4, 64, _om_LX8 if _om_BM == 8 else _om_LX32 if _om_BM == 32 else _om_LX)
    rv = r0 + mx
    pair = _om_gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    x_off = pair * K + qx * 64 + cx
    x_mask = (rv < r1) & (cx >= 0)
    xoff, xmsk = _om__xg_offsets(sorted_ptr, r0, r1, K, False, _om_BM)
    x_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [4 * _om_BM, 64], _om_X_SMEM)
    xb_s = _om_gl.allocate_shared_memory(_om_gl.bfloat16, [_om_BM, _om_BK], _om_XB_SMEM)
    wp = w_ptr.to(_om_gl.pointer_type(_om_gl.int32))
    acc = _om_gl.zeros([_om_BN, _om_BM], _om_gl.float32, layout=mma)
    if PF:
        wq_n = _om_gl.load(wp + w_off)
        sq_n = _om_gl.load(sf_ptr + s_off)
    for k0 in range(0, K, _om_BK):
        if PF:
            wq = wq_n
            sq = sq_n
            kn = k0 + _om_BK
            wq_n = _om_gl.load(wp + w_off + kn // _om_BK * (_om_BN * 32), mask=(rw >= 0) & (kn < K), other=0)
            sq_n = _om_gl.load(sf_ptr + s_off + kn // _om_BK * (_om_BN * 16), mask=(rs >= 0) & (kn < K))
        else:
            wq = _om_gl.load(wp + w_off + k0 // _om_BK * (_om_BN * 32))
            sq = _om_gl.load(sf_ptr + s_off + k0 // _om_BK * (_om_BN * 16))
        if SMB:
            _om__xb_store(a_ptr, xoff, xmsk, k0, xb_s, _om_BM, _om_BK)
            if CHK:
                q0, q1, q2, q3 = _om__wsplit4(wq, _om_BN)
                p0, p1, p2, p3 = _om__ssplit4(sq, _om_BN)
                acc = _om__mma_chunk(q0, p0, xb_s, 0, acc, _om_BN, a_l, HOP)
                acc = _om__mma_chunk(q1, p1, xb_s, 1, acc, _om_BN, a_l, HOP)
                acc = _om__mma_chunk(q2, p2, xb_s, 2, acc, _om_BN, a_l, HOP)
                acc = _om__mma_chunk(q3, p3, xb_s, 3, acc, _om_BN, a_l, HOP)
            elif HOP:
                acc = _om_warpgroup_mma(_om__decode_tile(wq, sq, _om_BN, _om_BK, _om_L3, a_l), xb_s.permute((1, 0)), acc)
            else:
                acc = _om_mma_v2(_om__decode_tile(wq, sq, _om_BN, _om_BK, _om_L3, _om_A_L), xb_s.permute((1, 0)).load(_om_B_L), acc)
        else:
            if XS:
                xb = _om__x_shared(a_ptr, xoff, xmsk, k0, x_s, _om_BM, _om_BK)
            else:
                xb = _om__x_tile(_om_gl.load(a_ptr + x_off + k0, mask=x_mask, other=0.0), _om_BM, _om_BK, _om_B_L)
            acc = _om_mma_v2(_om__decode_tile(wq, sq, _om_BN, _om_BK, _om_L3, _om_A_L), xb, acc)
    y = acc * (_om_gl.load(gs_ptr + e) * _om_ACC_BIAS)
    n_o = _om_gl.arange(0, _om_BN, layout=_om_gl.SliceLayout(1, mma))
    m_o = _om_gl.arange(0, _om_BM, layout=_om_gl.SliceLayout(0, mma))
    ro = r0 + m_o
    pair_o = _om_gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
    optr = out_ptr + _om_gl.expand_dims(pair_o, 0) * N + _om_gl.expand_dims(n0 + n_o, 1)
    _om_gl.store(optr, y.to(_om_gl.bfloat16), mask=_om_gl.expand_dims(ro < r1, 0) & _om_gl.expand_dims(n_o >= 0, 1))

@_om_gluon.jit
def _om__w2_mix(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, K: _om_gl.constexpr, N: _om_gl.constexpr, _om_BN: _om_gl.constexpr, _om_BK: _om_gl.constexpr, _om_BM: _om_gl.constexpr, XS: _om_gl.constexpr, SMB: _om_gl.constexpr=False, HOP: _om_gl.constexpr=False, CHK: _om_gl.constexpr=False, PF: _om_gl.constexpr=False):
    pid_n = _om_gl.program_id(0)
    pid_w = _om_gl.program_id(1)
    active = pid_w < _om_gl.load(nwork_ptr)
    e = _om_gl.load(work_ptr + pid_w * 3 + 0)
    r0 = _om_gl.load(work_ptr + pid_w * 3 + 1)
    r1 = _om_gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        if r1 - r0 <= 8:
            _om__w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n, K=K, N=N, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=8, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        else:
            _om__w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n, K=K, N=N, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=_om_BM, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)

def _om__src_bit(j, t):
    """Source bit in the stored word of target bit t of decoded output j (see _DECODE_ASM)."""
    if j == 0:
        return t
    if j == 1:
        return (t - 4) % 32
    if j == 2:
        return (t - 10) % 32
    return (t - 6) % 32 if 2160099520 >> t & 1 else (t - 14) % 32

def _om__permute_bits(packed):
    """uint8 [..., K/2] (two E2M1 codes per byte, low nibble first) -> the same codes with each 32-bit word's bits
    permuted (lossless): natural code c of the word goes to decoded output c // 2, half c % 2."""
    words = packed.contiguous().view(_om_torch.int32)
    out = _om_torch.zeros_like(words)
    for c in range(8):
        j, h = (c // 2, c % 2)
        for bit, tgt in enumerate((6, 7, 8, 15)):
            out |= (words >> 4 * c + bit & 1) << _om__src_bit(j, tgt + 16 * h)
    return out.view(_om_torch.uint8).view(packed.shape)

def _om__tile_major(codes, sf):
    """Codes uint8 [L, rows, K/2] and fp8 scales [L, rows, K/16] -> the same bytes ordered tile-major, so that
    each (expert, BN-row tile, BK block) is one contiguous BN x 128 B run of codes and one BN x 16 B run of scales
    (a lossless permutation of bytes; contiguous blocks stream from DRAM better than BN rows 2 KB apart)."""
    n_l, rows, half_k = codes.shape
    kb = 2 * half_k // _om_BK
    c = codes.view(n_l, rows // _om_BN, _om_BN, kb, _om_BK // 2).permute(0, 1, 3, 2, 4).contiguous()
    s = sf.contiguous().view(n_l, rows // _om_BN, _om_BN, kb, _om_BK // 16).permute(0, 1, 3, 2, 4).contiguous()
    return (c, s)

def _om__gu_rows(I):
    """Row order of the gate/up-paired W13 copy. New row 64 t + 16 s + 8 g + i holds gate (g = 0) or up (g = 1)
    row of output column 32 t + 8 s + i (a lossless row permutation)."""
    r = _om_torch.arange(2 * I)
    t, s, g, i = (r // 64, r % 64 // 16, r % 16 // 8, r % 8)
    return g * I + 32 * t + 8 * s + i

def _om__interleaved(c, s):
    """Tile-major codes and scales merged per block (codes then scales, 9216 B), a second lossless copy
    used from IL_MIN_M tokens and at IL_M, where it streams better."""
    n_l, nt, kb = c.shape[:3]
    blk = _om_torch.cat([c.reshape(n_l, nt, kb, -1), s.view(_om_torch.uint8).reshape(n_l, nt, kb, -1)], dim=3).contiguous()
    return (blk, blk.view(_om_torch.float8_e4m3fn))

def _om_prepare(weights):
    L = 32
    w13, w13_sf = _om__tile_major(_om_torch.stack([_om__permute_bits(weights['w13'][e]) for e in range(L)]), weights['w13_sf'][:L])
    gu = _om__gu_rows(weights['w13'].shape[1] // 2).to(weights['w13'].device)
    w13g, w13g_sf = _om__tile_major(_om_torch.stack([_om__permute_bits(weights['w13'][e][gu]) for e in range(L)]), weights['w13_sf'][:L][:, gu])
    w2, w2_sf = _om__tile_major(_om_torch.stack([_om__permute_bits(weights['w2'][e]) for e in range(L)]), weights['w2_sf'][:L])
    state = {'w13': w13, 'w13_sf': w13_sf, 'w13_gs': weights['w13_gs'][:L].float().contiguous(), 'w2': w2, 'w2_sf': w2_sf, 'w2_gs': weights['w2_gs'][:L].float().contiguous(), 'w13g': w13g, 'w13g_sf': w13g_sf, 'il13': _om__interleaved(w13, w13_sf), 'il2': _om__interleaved(w2, w2_sf), 'norm_w': (weights['post_norm_w'].float() + 1.0).to(_om_torch.bfloat16), 'local': (0, L), 'hopper': _om_torch.cuda.get_device_capability()[0] == 9, 'local_start_t': _om_torch.full((1 + _om_TOP_K,), -1, dtype=_om_torch.int32, device=weights['w13'].device), 'iota8': _om_torch.arange(_om_TOP_K, dtype=_om_torch.int32, device=weights['w13'].device)}
    state['local_start_t'][0] = 0
    _om_torch.cuda.synchronize()
    return state

def _om__decode1_direct(x, ids, tw, state, H, I, L):
    """M = 1: three launches (W13 + SiLU, W2, post-expert norm); each top-k slot is one item, no routing pass."""
    dev = x.device
    inter = _om_torch.empty((_om_TOP_K, I), dtype=_om_torch.bfloat16, device=dev)
    if 1 in _om_GU_M:
        _om__w2[2 * I // _om_BN, _om_TOP_K](x, state['w13g'], state['w13g_sf'], state['w13_gs'], state['iota8'], ids, state['local_start_t'], inter, K=H, N=2 * I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=8, XS=True, SMB=False, HOP=False, CHK=False, PF=True, DIRECT=True, NLOC=L, GU=True, num_warps=_om_NUM_WARPS)
    else:
        S = _om_D1_W13_SPLIT
        part = _om_torch.empty((S, _om_TOP_K, 2 * I), dtype=_om_torch.float32, device=dev)
        _om__w13_silu[I // _om_BN, _om_TOP_K, S](x, state['w13'], state['w13_sf'], state['w13_gs'], state['iota8'], ids, state['local_start_t'], part, K=H, I=I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=8, XS=True, SMB=False, HOP=False, CHK=False, PF=True, DIRECT=True, NLOC=L, SPLIT=S, num_warps=_om_NUM_WARPS)
        _om__d1_reduce[_om_TOP_K, I // 1024](part, state['w13_gs'], state['local_start_t'][1:], inter, I=I, S=S, BLOCK=1024, NLOC=L, num_warps=4)
    ks = _om_D1_W2_KS
    yp = _om_torch.empty((ks, _om_TOP_K, H), dtype=_om_torch.float32, device=dev)
    _om__w2_ks[H // _om_BN, _om_TOP_K, ks](inter, state['w2'], state['w2_sf'], state['w2_gs'], state['iota8'], ids, state['local_start_t'], yp, K=I, N=H, _om_BK=_om_BK, _om_BM=8, KS=ks, PF=True, NLOC=L, num_warps=_om_NUM_WARPS)
    out = _om_torch.empty((1, H), dtype=_om_torch.bfloat16, device=dev)
    _om__wsum_ks[1,](yp, state['norm_w'], tw, out, state['local_start_t'][1:], _om_POST_NORM_EPS, H=H, KS=ks, TOPK=_om_TOP_K, FP32=True, num_warps=8)
    return out

def _om__w13_regcap(M):
    n = (_om_W13_MAXNREG_MIX if M >= _om_MIX_MIN_M else _om_W13_MAXNREG_BIG) if M >= _om_SMB_MIN_M else _om_W13_MAXNREG.get(M)
    return {'maxnreg': n} if n else {}

def _om_kernel_fn(data, state):
    cfg = data['config']
    x = data['hidden_states'].contiguous()
    M, H = x.shape
    I = cfg['d_expert']
    L = cfg['local_count']
    ls = cfg['local_start']
    assert (ls, L) == state['local'], 'prepared for experts [0, 32) (ep8, this GPU)'
    dev = x.device
    P = M * _om_TOP_K
    ids = data['topk_ids'].contiguous()
    tw = data['topk_weights'].contiguous()
    if M == 1:
        return _om__decode1_direct(x, ids, tw, state, H, I, L)
    p_block = _om_triton.next_power_of_2(P)
    bm = 8 if M <= 32 else _om_BM
    w_max = _om_triton.next_power_of_2(_om_triton.cdiv(P, bm) + L)
    local = _om_torch.empty((M, _om_TOP_K), dtype=_om_torch.int32, device=dev)
    sorted_pairs = _om_torch.empty((P,), dtype=_om_torch.int32, device=dev)
    work = _om_torch.empty((w_max, 3), dtype=_om_torch.int32, device=dev)
    nwork = _om_torch.empty((1,), dtype=_om_torch.int32, device=dev)
    if M >= _om_MIX_MIN_M:
        bm13 = _om_BM_W13
        w_max13 = _om_triton.next_power_of_2(_om_triton.cdiv(P, bm13) + L)
        work13 = _om_torch.empty((w_max13, 3), dtype=_om_torch.int32, device=dev)
        nwork13 = _om_torch.empty((1,), dtype=_om_torch.int32, device=dev)
        _om__route2[1,](ids, local, sorted_pairs, work, nwork, work13, nwork13, P, ls, LOCAL=L, _om_BM=bm, BM2=bm13, P_BLOCK=p_block, W_MAX=w_max, W_MAX2=w_max13, num_warps=16 if p_block >= 1024 else 4)
    else:
        _om__route[1,](ids, local, sorted_pairs, work, nwork, P, ls, LOCAL=L, _om_BM=bm, P_BLOCK=p_block, W_MAX=w_max, num_warps=16 if p_block >= 1024 else 4)
    inter = _om_torch.empty((P, I), dtype=_om_torch.bfloat16, device=dev)
    il = M in _om_IL_M
    w13c, w13s = state['il13'] if il else (state['w13'], state['w13_sf'])
    w2c, w2s = state['il2'] if il else (state['w2'], state['w2_sf'])
    if M in _om_GU_M:
        _om__w2[2 * I // _om_BN, w_max](x, state['w13g'], state['w13g_sf'], state['w13_gs'], sorted_pairs, work, nwork, inter, K=H, N=2 * I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=bm, XS=M >= _om_XS_MIN_M, SMB=False, HOP=False, CHK=True, PF=M <= _om_PF_MAX_M, GU=True, num_warps=_om_NUM_WARPS)
    elif M >= _om_MIX_MIN_M:
        smb = _om_VERIFY_SMB and M >= _om_SMB_MIN_M
        _om__w13_mix[I // _om_BN, w_max13](x, state['w13'], state['w13_sf'], state['w13_gs'], sorted_pairs, work13, nwork13, inter, K=H, I=I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=bm13, XS=M >= _om_XS_MIN_M, SMB=smb, HOP=state['hopper'] and smb, CHK=True, PF=M <= _om_PF_MAX_M, num_warps=_om_NUM_WARPS, **_om__w13_regcap(M))
    else:
        _om__w13_silu[I // _om_BN, w_max](x, w13c, w13s, state['w13_gs'], sorted_pairs, work, nwork, inter, K=H, I=I, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=bm, XS=M >= _om_XS_MIN_M, SMB=M >= _om_SMB_MIN_M, HOP=state['hopper'] and M >= _om_SMB_MIN_M, CHK=True, PF=M <= _om_PF_MAX_M, IL=il, num_warps=_om_NUM_WARPS, **_om__w13_regcap(M))
    y = _om_torch.empty((P, H), dtype=_om_torch.bfloat16, device=dev)
    if M >= _om_MIX_MIN_M and (not _om_W2_VERIFY_C175):
        smb2 = _om_VERIFY_SMB and M >= _om_SMB_MIN_M
        _om__w2_mix[H // _om_BN, w_max](inter, state['w2'], state['w2_sf'], state['w2_gs'], sorted_pairs, work, nwork, y, K=I, N=H, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=bm, XS=M >= _om_XS_MIN_M, SMB=smb2, HOP=state['hopper'] and smb2, CHK=True, PF=M <= _om_PF_MAX_M, num_warps=_om_NUM_WARPS)
    else:
        _om__w2[H // _om_BN, w_max](inter, w2c, w2s, state['w2_gs'], sorted_pairs, work, nwork, y, K=I, N=H, _om_BN=_om_BN, _om_BK=_om_BK, _om_BM=bm, XS=M >= _om_XS_MIN_M, SMB=M >= _om_SMB_MIN_M, HOP=state['hopper'] and M >= _om_SMB_MIN_M, CHK=True, PF=M <= _om_PF_MAX_M, IL=il, num_warps=_om_NUM_WARPS)
    out = _om_torch.empty((M, H), dtype=_om_torch.bfloat16, device=dev)
    block_h = _om_triton.next_power_of_2(H)
    _om__post_expert_norm_weighted_sum_kernel[M,](y, state['norm_w'], tw, out, local, TOPK=_om_TOP_K, H=H, eps=_om_POST_NORM_EPS, APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=block_h, num_warps=8 if block_h >= 4096 else 4, num_stages=2)
    return out

_native_bucket_prepare=prepare
def prepare(weights):
    global _om_BK
    state=_native_bucket_prepare(weights)
    state['operatorstack']=_om_prepare(weights)
    old_bk=_om_BK
    _om_BK=512
    try:
        state['operatorstack_bk512']=_om_prepare(weights)
    finally:
        _om_BK=old_bk
    return state
_native_bucket_kernel_fn=kernel_fn
def kernel_fn(data,state):
    global _om_BK
    c=data['config'];m=data['hidden_states'].shape[0]
    valid=(c['local_start']==0 and c['local_count']==32 and c['d_hidden']==4096 and c['d_expert']==4096)
    if m==32 and valid:
        old_bk=_om_BK
        _om_BK=512
        try:
            return _om_kernel_fn(data,state['operatorstack_bk512'])
        finally:
            _om_BK=old_bk
    if m==128 and valid:
        return _om_kernel_fn(data,state['operatorstack'])
    return _native_bucket_kernel_fn(data,state)
