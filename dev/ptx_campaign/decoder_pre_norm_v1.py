# okc: problem=muse_decode_layer
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "wm": _wm_prepare(weights),
        "dn": _dense_prepare(weights, device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state



# ---- weight-major NVFP4 experts (CUDA) ----
_WM_CUDA = r'''
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

#define HD 4096
#define HD_ 4096
#define ID 4096
#define KC 64            // k64 chunks along K = 4096
#define KCG 16           // groups of 4 chunks (scale words)
#define RT0 512          // 16-row tiles of w13 (gate 256 + up 256)
#define RT1 256          // 16-row tiles of w2

constexpr uint32_t MASK = 0x81C081C0u;

// Weight word -> bf16x2 holding the E2M1 value times 2^-126 (denormal-linear encoding).
template <int J>
__device__ __forceinline__ uint32_t dec(uint32_t w) {
  if constexpr (J == 0) {
    return w & MASK;
  } else if constexpr (J < 3) {
    return __funnelshift_r(w, w, 3 * J) & MASK;
  } else {
    return ((w & 0x001B001Bu) * 0x1084u) & MASK;  // sources m0@4 m1@0 m2@1 sign@3 (+16 for the high half)
  }
}

__device__ __forceinline__ uint32_t scale2(uint32_t q, int c) {
  // byte c of q holds b'; scale = bf16 bits (b'<<4) + 0x7600 = S * 2^118
  uint32_t bits = ((q >> (8 * c)) & 0xFFu) << 4;
  return (bits + 0x7600u) * 0x10001u;
}

__device__ __forceinline__ uint32_t hmul(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

static int g_pdl = 0;
static int g_persist = 0;
extern "C" void wm_set_persist(int v) { g_persist = v; }
extern "C" void wm_set_pdl(int v) { g_pdl = v; }

template <typename... KArgs, typename... Args>
static void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, bool pdl,
                     Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

__global__ void route_kernel(const int* __restrict__ ids, int npair, const int* __restrict__ ls_ptr, int ls_val, int lmax,
                             int* __restrict__ cnt, int* __restrict__ lst, int* __restrict__ act,
                             int* __restrict__ nact, int* __restrict__ local_ids, int* __restrict__ woff, int grp, int nslots) {
  __shared__ int s_cnt[32];
  pdl_wait();
  const int ls = ls_ptr ? *ls_ptr : ls_val;
  pdl_trigger();
  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;
  for (int i = threadIdx.x; i < nslots; i += blockDim.x) reinterpret_cast<int4*>(act)[i] = make_int4(0, 0, 0, 0);  // work items {expert, offset, tokens, 0}
  __syncthreads();
  for (int p = threadIdx.x; p < npair; p += blockDim.x) {
    int e = (ids[p] - ls) & 255;  // experts are ordered mod 256 starting at this GPU's first expert
    bool ok = e < 32;
    local_ids[p] = ok ? e : -1;
    if (ok) {
      int pos = atomicAdd(&s_cnt[e], 1);
      lst[e * lmax + pos] = p;
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    int lane = threadIdx.x;
    int c = s_cnt[lane];
    cnt[lane] = c;
    // work items: one per group of `grp` tokens of an expert, so a hot expert spreads over several CTAs
    const int groups = (c + grp - 1) / grp;
    // longest-processing-time-first: experts with the most tokens (largest NCH) get the earliest work items
    int start = 0, total = 0;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      const int cj = __shfl_sync(0xffffffffu, c, j), gj = __shfl_sync(0xffffffffu, groups, j);
      if (cj > c || (cj == c && j < lane)) start += gj;
      total += gj;
    }
    for (int j = 0; j < groups; j++)
      reinterpret_cast<int4*>(act)[start + j] = make_int4(lane, j * grp, min(c - j * grp, grp), 1);
    if (lane == 31) *nact = total;
  }
}

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc));
}
__device__ __forceinline__ void cp16z(void* sdst, const void* gsrc, uint32_t nbytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc), "r"(nbytes));
}
template <int NTB_, int NCH_>
__host__ __device__ constexpr int ring2_stages() {
  return (14336 / (512 * (NTB_ + 2 * NCH_))) > 6 ? 6 : ((14336 / (512 * (NTB_ + 2 * NCH_))) < 3 ? 3 : (14336 / (512 * (NTB_ + 2 * NCH_))));
}
__device__ __forceinline__ void cp8(void* sdst, const void* gsrc) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(sdst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(d), "l"(gsrc));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int NTB, int NCH, int NW>
__device__ __forceinline__ void wm_epilogue(float (&acc)[NTB][NCH][4], float* red, const float gs, const int* lst,
                                            int ntok, int tbase, int mode, int nt, bf16* OUT, const int bx) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++)
        red[(((warp * NTB + n) * NCH + c) * 4 + r) * 32 + lane] = acc[n][c][r];
  __syncthreads();
  for (int item = threadIdx.x; item < NCH * 4 * 32; item += NW * 32) {
    const int li = item & 31, r = (item >> 5) & 3, c = item >> 7;
    const int row = (li >> 2) + 8 * (r >> 1);
    const int col = c * 8 + 2 * (li & 3) + (r & 1);
    if (col >= ntok) continue;
    const int p = __ldcg(lst + col);
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      if (mode == 0 && (n & 1)) continue;
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) s += red[(((w * NTB + n) * NCH + c) * 4 + r) * 32 + li];
      if (mode == 0) {
        float s2 = 0.f;
#pragma unroll
        for (int w = 0; w < NW; w++) s2 += red[(((w * NTB + n + 1) * NCH + c) * 4 + r) * 32 + li];
        const float gate = bfr(bfr(s * 256.0f) * gs), up = bfr(bfr(s2 * 256.0f) * gs);
        const float sil = gate / (1.0f + __expf(-gate));
        OUT[(size_t)p * ID + (bx * (nt / 2) + ((tbase + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);
      } else {
        OUT[(size_t)p * HD + (bx * nt + tbase + n) * 16 + row] = __float2bfloat16(bfr(s * 256.0f) * gs);
      }
    }
  }
  __syncthreads();
}

// RING == false: every lane issues its weight loads up front, group by group (best for small problems).
// RING == true : weights stream through a per-warp cp.async ring in shared memory (lane-private slots).
template <int MODE, int NT, int NTB, int NCH, int NW, int RING, int S, int DBG>
__device__ __forceinline__ void wm_body(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq,
                                        const float* __restrict__ GS, const bf16* __restrict__ X,
                                        bf16* __restrict__ OUT, const int* __restrict__ lst, int e, int ntok,
                                        int tbase, uint4* smem, const int bx) {
  constexpr int RT = MODE == 0 ? RT0 : RT1;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int CPW = 64 / NW;  // chunks of k64 per warp
  constexpr int GPW = CPW / 4;  // scale groups per warp
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  float* red = reinterpret_cast<float*>(smem);

  int tile[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) {
    const int nn = tbase + n;
    if (MODE == 0) tile[n] = (nn & 1) ? RT0 / 2 + bx * (NT / 2) + (nn >> 1) : bx * (NT / 2) + (nn >> 1);
    else tile[n] = bx * NT + nn;
  }

  const bf16* xp[NCH];
#pragma unroll
  for (int c = 0; c < NCH; c++) {
    int col = c * 8 + g;
    xp[c] = nullptr;
    if (col < ntok) {
      int p = __ldcg(lst + col);
      xp[c] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + 16 * t + warp * CPW * 64;
    }
  }

  float acc[NTB][NCH][4];
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;

  uint32_t sink = 0;
  const uint4* wsrc[NTB];
  const uint2* ssrc[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) {
    wsrc[n] = Wq + ((size_t)e * RT + tile[n]) * KC * 32 + warp * CPW * 32 + lane;
    ssrc[n] = Sq + ((size_t)e * RT + tile[n]) * KCG * 32 + warp * GPW * 32 + lane;
  }

  if (RING == 0) {
#pragma unroll
    for (int q = 0; q < GPW; q++) {
      uint2 sc[NTB];
      uint4 wv[NTB][4];
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        sc[n] = __ldg(ssrc[n] + q * 32);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          if constexpr (DBG & 8) wv[n][c] = make_uint4(lane * 7 + c, lane * 13 + q, lane + n, lane * 3);
          else wv[n][c] = __ldg(wsrc[n] + (q * 4 + c) * 32);
        }
      }
#pragma unroll
      for (int c = 0; c < 4; c++) {
        uint4 xa[NCH], xb[NCH];
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          if (xp[h] && !(DBG & 4)) {
            const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (q * 4 + c) * 64);
            xa[h] = __ldcg(xv);
            xb[h] = __ldcg(xv + 1);
          } else {
            xa[h] = make_uint4(0, 0, 0, 0);
            xb[h] = make_uint4(0, 0, 0, 0);
          }
        }
#pragma unroll
        for (int n = 0; n < NTB; n++) {
          const uint32_t s0 = scale2(sc[n].x, c);
          const uint32_t s1 = scale2(sc[n].y, c);
          const uint32_t ww[4] = {wv[n][c].x, wv[n][c].y, wv[n][c].z, wv[n][c].w};
#pragma unroll
          for (int ks = 0; ks < 4; ks++) {
            const uint32_t w = ww[ks];
            uint32_t a0, a1, a2, a3;
            if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }
            else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }
#pragma unroll
            for (int h = 0; h < NCH; h++) {
              const uint4 xs = ks < 2 ? xa[h] : xb[h];
              if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
              else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
            }
          }
        }
      }
    }
  } else if (RING == 2) {
    constexpr int XS = NCH * 2;
    constexpr int STG = NTB + XS;
    uint4* ring = smem + warp * (S * STG * 32);
    auto issue = [&](int c) {
      if (c < CPW && !(DBG & 8)) {
#pragma unroll
        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * STG + n) * 32 + lane], wsrc[n] + c * 32);
      }
      if (c < CPW) {
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const bool ok = xp[h] != nullptr && !(DBG & 4);
          const bf16* src = ok ? xp[h] + c * 64 : X;
          cp16z(&ring[((c % S) * STG + NTB + 2 * h) * 32 + lane], src, ok ? 16u : 0u);
          cp16z(&ring[((c % S) * STG + NTB + 2 * h + 1) * 32 + lane], src + 8, ok ? 16u : 0u);
        }
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);
    uint2 scur[NTB], snxt[NTB];
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      scur[n] = __ldg(ssrc[n]);
      snxt[n] = GPW > 1 ? __ldg(ssrc[n] + 32) : scur[n];
    }
#pragma unroll 1
    for (int i = 0; i < CPW; i++) {
      cp_wait<S - 2>();
      issue(i + S - 1);
      const int cq = i & 3;
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        const uint4 wv = ring[((i % S) * STG + n) * 32 + lane];
        const uint32_t s0 = scale2(scur[n].x, cq);
        const uint32_t s1 = scale2(scur[n].y, cq);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint32_t w = ww[ks];
          uint32_t a0, a1, a2, a3;
          if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }
          else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = ring[((i % S) * STG + NTB + 2 * h + (ks >> 1)) * 32 + lane];
            if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
            else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
          }
        }
      }
      if (cq == 3) {
#pragma unroll
        for (int n = 0; n < NTB; n++) {
          scur[n] = snxt[n];
          if ((i >> 2) + 2 < GPW) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);
        }
      }
    }
    cp_wait<0>();
  } else {
    uint4* ring = smem + warp * (S * NTB * 32);
    auto issue = [&](int c) {
      if (c < CPW && !(DBG & 8)) {
#pragma unroll
        for (int n = 0; n < NTB; n++) cp16(&ring[((c % S) * NTB + n) * 32 + lane], wsrc[n] + c * 32);
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);

    uint2 scur[NTB], snxt[NTB];
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      scur[n] = __ldg(ssrc[n]);
      snxt[n] = GPW > 1 ? __ldg(ssrc[n] + 32) : scur[n];
    }
    uint4 xcur[NCH][2], xnxt[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      if (xp[h] && !(DBG & 4)) {
        const uint4* xv = reinterpret_cast<const uint4*>(xp[h]);
        xcur[h][0] = __ldcg(xv);
        xcur[h][1] = __ldcg(xv + 1);
      } else {
        xcur[h][0] = make_uint4(0, 0, 0, 0);
        xcur[h][1] = make_uint4(0, 0, 0, 0);
      }
      xnxt[h][0] = xcur[h][0];
      xnxt[h][1] = xcur[h][1];
    }

#pragma unroll 1
    for (int i = 0; i < CPW; i++) {
      cp_wait<S - 2>();
      issue(i + S - 1);
#pragma unroll
      for (int h = 0; h < NCH; h++) {
        if (xp[h] && i + 1 < CPW && !(DBG & 4)) {
          const uint4* xv = reinterpret_cast<const uint4*>(xp[h] + (i + 1) * 64);
          xnxt[h][0] = __ldcg(xv);
          xnxt[h][1] = __ldcg(xv + 1);
        }
      }
      const int cq = i & 3;
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        const uint4 wv = ring[((i % S) * NTB + n) * 32 + lane];
        const uint32_t s0 = scale2(scur[n].x, cq);
        const uint32_t s1 = scale2(scur[n].y, cq);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint32_t w = ww[ks];
          uint32_t a0, a1, a2, a3;
          if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }
          else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = xcur[h][ks >> 1];
            if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
            else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
          }
        }
      }
      if (cq == 3) {
#pragma unroll
        for (int n = 0; n < NTB; n++) {
          scur[n] = snxt[n];
          if ((i >> 2) + 2 < GPW) snxt[n] = __ldg(ssrc[n] + ((i >> 2) + 2) * 32);
        }
      }
#pragma unroll
      for (int h = 0; h < NCH; h++) {
        xcur[h][0] = xnxt[h][0];
        xcur[h][1] = xnxt[h][1];
      }
    }
    cp_wait<0>();
  }

  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);
  __syncthreads();  // the ring is dead; the same smem holds the cross-warp reduction
  wm_epilogue<NTB, NCH, NW>(acc, red, GS[e], lst, ntok, tbase, MODE, NT, OUT, bx);
}

#ifndef WM_MINB
#define WM_MINB 4
#endif
template <int MODE, int NT, int NMAX, int NW, int RING, int DBG>
__global__ void __launch_bounds__(NW * 32, (RING == 3 ? 2 : WM_MINB))
wm_gemm(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,
        const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ cnt,
        const int* __restrict__ lst, const int* __restrict__ act, const int* __restrict__ nact_p,
        const int* __restrict__ woff, int lmax, int persist) {
  extern __shared__ uint4 smem[];
  constexpr int TILES = (MODE == 0 ? 512 : 256) / NT;
  pdl_wait();  // everything below (items, token lists, activations) comes from earlier kernels
  pdl_trigger();
  const int nitems = persist ? __ldcg(nact_p) : 1;
  for (int u = blockIdx.x;; u += gridDim.x) {
  int slot, bx;
  if (persist) {
    if (u >= nitems * TILES) break;
    slot = u / TILES; bx = u - slot * TILES;
  } else { slot = blockIdx.y; bx = blockIdx.x; }
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + slot);  // {expert, offset, tokens, valid}
  if (item.z != 0) {
  const int e = item.x;
  const int off = item.y;
  {
    const int ntok = item.z;
    const int* l = lst + e * lmax + off;
    if (ntok <= 8) {
      wm_body<MODE, NT, NT, 1, NW, RING, (RING == 2 ? ring2_stages<NT, 1>() : (NT == 8 ? 3 : NT == 4 ? 4 : 6)), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, 0, smem, bx);
    } else if (NMAX >= 2 && ntok <= 16) {
#pragma unroll 1
      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))
        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 2 ? 2 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 2 ? 2 : 1)>() : RING == 3 ? 4 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);
    } else if (NMAX >= 4) {
#pragma unroll 1
      for (int tb = 0; tb < NT; tb += (RING == 3 ? 4 : 2))
        wm_body<MODE, NT, (RING == 3 ? 4 : 2), (NMAX >= 4 ? 4 : 1), NW, RING, (RING == 2 ? ring2_stages<2, (NMAX >= 4 ? 4 : 1)>() : RING == 3 ? 4 : 6), DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, tb, smem, bx);
    }
  }
  }
  if (!persist) break;
  }
}



// ---- row-split variant: the four warps of a CTA own different weight rows but the same activations, which are
// staged once per CTA through shared memory.  No cross-warp reduction; x traffic from L2 drops 4x. ----
template <int MODE, int NTB, int NCH, int S, int DBG>
__device__ __forceinline__ void wm_body_rs(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq,
                                           const float* __restrict__ GS, const bf16* __restrict__ X,
                                           bf16* __restrict__ OUT, const int* __restrict__ lst, int e, int ntok,
                                           uint4* smem, const int bx) {
  constexpr int RT = MODE == 0 ? RT0 : RT1;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int NTC = 4 * NTB;  // tiles per CTA
  constexpr int XSLOTS = NCH * 2 * 32;
  constexpr int RPT = (XSLOTS + 127) / 128;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  constexpr int SOFF = 3 * 4 * NTB * 16;     // scale ring: [3][4][NTB][32] uint2, in uint4 units
  uint2* sring = reinterpret_cast<uint2*>(smem);
  uint4* wring = smem + SOFF;                // [S][4][NTB][32]
  uint4* xring = wring + S * 4 * NTB * 32;   // [S][XSLOTS]

  const uint4* wsrc[NTB];
  const uint2* ssrc[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) {
    const int nn = warp * NTB + n;
    const int tile = MODE == 0 ? ((nn & 1) ? RT0 / 2 + bx * (NTC / 2) + (nn >> 1) : bx * (NTC / 2) + (nn >> 1)) : bx * NTC + nn;
    wsrc[n] = Wq + ((size_t)e * RT + tile) * KC * 32 + lane;
    ssrc[n] = Sq + ((size_t)e * RT + tile) * KCG * 32 + lane;
  }
  const bf16* xb[RPT];
  bool xok[RPT];
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    const int sl = tid + 128 * r;
    xok[r] = false;
    xb[r] = X;
    if (sl < XSLOTS) {
      const int h = sl >> 6, j = (sl >> 5) & 1, L = sl & 31;
      const int col = h * 8 + (L >> 2);
      if (col < ntok) {
        const int p = __ldcg(lst + col);
        xb[r] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + 16 * (L & 3) + 8 * j;
        xok[r] = true;
      }
    }
  }

  uint32_t sink = 0;
  float acc[NTB][NCH][4];
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;

  auto issue_sc = [&](int g) {
    if (g < KCG && !(DBG & 32)) {
#pragma unroll
      for (int n = 0; n < NTB; n++) cp8(&sring[(((g % 3) * 4 + warp) * NTB + n) * 32 + lane], ssrc[n] + g * 32);
    }
  };
  auto issue = [&](int c) {
    if ((c & 3) == 0) issue_sc((c >> 2) + 1);
    if (c < KC) {
#pragma unroll
      for (int n = 0; n < NTB; n++) { if constexpr (!(DBG & 8)) cp16(&wring[(((c % S) * 4 + warp) * NTB + n) * 32 + lane], wsrc[n] + c * 32); }
#pragma unroll
      for (int r = 0; r < RPT; r++)
        if (tid + 128 * r < XSLOTS && !(DBG & 4)) cp16z(&xring[(c % S) * XSLOTS + tid + 128 * r], xb[r] + c * 64, xok[r] ? 16u : 0u);
    }
    cp_commit();
  };
  issue_sc(0);
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
#pragma unroll 1
  for (int i = 0; i < KC; i++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(i + S - 1);
    const int cq = i & 3;
    uint4 xr[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      xr[h][0] = xring[(i % S) * XSLOTS + (h * 2) * 32 + lane];
      xr[h][1] = xring[(i % S) * XSLOTS + (h * 2 + 1) * 32 + lane];
    }
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      const uint4 wv = wring[(((i % S) * 4 + warp) * NTB + n) * 32 + lane];
      const uint2 sc = (DBG & 32) ? make_uint2(0x38383838u, 0x38383838u) : sring[((((i >> 2) % 3) * 4 + warp) * NTB + n) * 32 + lane];
      const uint32_t s0 = scale2(sc.x, cq);
      const uint32_t s1 = scale2(sc.y, cq);
      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        const uint32_t w = ww[ks];
        uint32_t a0, a1, a2, a3;
        if constexpr (DBG & 1) { a0 = w; a1 = w + 1; a2 = w ^ s0; a3 = w ^ s1; }
        else { a0 = hmul(dec<0>(w), s0); a1 = hmul(dec<1>(w), s1); a2 = hmul(dec<2>(w), s0); a3 = hmul(dec<3>(w), s1); }
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const uint4 xs = xr[h][ks >> 1];
          if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
          else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
        }
      }
    }
  }
  cp_wait<0>();
  if constexpr (DBG & 2) acc[0][0][0] = __uint_as_float(sink);

  const float gs = GS[e];
#pragma unroll
  for (int c = 0; c < NCH; c++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int col = c * 8 + 2 * t + (r & 1);
      if (col >= ntok || (DBG & 16)) continue;
      const int p = (DBG & 64) ? col * 7 : __ldcg(lst + col);
      const int row = g + 8 * (r >> 1);
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        if (MODE == 0) {
          if (n & 1) continue;
          const float gate = bfr(bfr(acc[n][c][r] * 256.0f) * gs), up = bfr(bfr(acc[n + 1][c][r] * 256.0f) * gs);
          const float sil = gate / (1.0f + __expf(-gate));
          OUT[(size_t)p * ID + (bx * (NTC / 2) + ((warp * NTB + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);
        } else {
          OUT[(size_t)p * HD + (bx * NTC + warp * NTB + n) * 16 + row] = __float2bfloat16(bfr(acc[n][c][r] * 256.0f) * gs);
        }
      }
    }
}

template <int MODE, int NTB, int NMAX, int S, int DBG>
__global__ void __launch_bounds__(128, (S <= 3 ? 4 : 3))
wm_gemm_rs(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,
           const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst,
           const int* __restrict__ act, int lmax) {
  extern __shared__ uint4 smem[];
  pdl_wait();
  pdl_trigger();
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);
  if (item.z == 0) return;
  const int e = item.x, ntok = item.z;
  const int* l = lst + e * lmax + item.y;
  if (ntok <= 8) wm_body_rs<MODE, NTB, 1, S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
  else if (NMAX >= 2 && ntok <= 16) wm_body_rs<MODE, NTB, (NMAX >= 2 ? 2 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
  else if (NMAX >= 4 && (NMAX < 8 || ntok <= 32)) wm_body_rs<MODE, NTB, (NMAX >= 4 ? 4 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
  else if (NMAX >= 8) wm_body_rs<MODE, NTB, (NMAX >= 8 ? 8 : 1), S, DBG>(Wq, Sq, GS, X, OUT, l, e, ntok, smem, blockIdx.x);
}

template <int MODE, int NTB, int NMAX, int S, int DBG>
static void launch_rs(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst,
                      const int* act, int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid((MODE == 0 ? 512 : 256) / (4 * NTB), (unsigned)maxslots);
  const int dyn = S * (4 * NTB * 512 + NMAX * 2 * 32 * 16) + 3 * 4 * NTB * 256;
  cudaFuncSetAttribute(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(wm_gemm_rs<MODE, NTB, NMAX, S, DBG>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const uint2*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);
}


// ---- wgmma (Hopper) building blocks ----
__device__ __forceinline__ uint64_t wg_desc(const void* p, uint32_t lbo, uint32_t sbo) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  return (uint64_t)((a >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit_wait() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void proxy_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
template <int N> struct WgN;

template <> struct WgN<32> {
  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
  }
};
template <> struct WgN<64> {
  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
  }
};
template <> struct WgN<128> {
  static __device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
  }
};
// position of natural k inside its 64-chunk after the 'fragment order' permutation the wgmma path reads its activations in
__device__ __forceinline__ int wg_perm_k(int k) {
  const int c = k >> 6, kl = k & 63;
  const int t = kl >> 4, ks = (kl >> 2) & 3, p = (kl >> 1) & 1, h = kl & 1;
  return c * 64 + (ks * 2 + p) * 8 + t * 2 + h;
}
__global__ void wm_permx_kernel(const bf16* __restrict__ X, bf16* __restrict__ Xp, int ld) {
  pdl_wait();
  pdl_trigger();
  const size_t base = (size_t)blockIdx.x * ld;
  for (int k = threadIdx.x; k < ld; k += blockDim.x) Xp[base + wg_perm_k(k)] = X[base + k];
}
extern "C" void wm_permx_c(const void* X, void* Xp, int M, cudaStream_t st) {
  launch_k(wm_permx_kernel, dim3(M), dim3(256), 0, st, g_pdl != 0, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(Xp), 4096);
}

// ---- wgmma row-split variant: a warpgroup owns 64 weight rows (one 16-row tile per warp, decoded into registers) and up to
// 128 tokens; B (activations) lives in shared memory in core-matrix layout ----
template <int MODE, int NCH, int S>
__device__ __forceinline__ void wm_body_wg(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS,
                                           const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst, int e,
                                           int ntok, uint8_t* smem_raw, const int bx) {
  constexpr int RT = MODE == 0 ? RT0 : RT1;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int N = NCH * 8;
  constexpr int NG = MODE == 0 ? 2 : 1;                  // weight tiles per warp: gate + up (MODE 0)
  constexpr int XST = N * 128;                           // bytes of one x stage (N tokens x 64 k bf16)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  uint8_t* xring = smem_raw;                             // [S][N tokens in core layout]
  uint4* wring = reinterpret_cast<uint4*>(smem_raw + S * XST);   // [S][4 warps][NG][32 lanes]: one uint4 (4 ks words) per lane
  const uint4* wsrc[NG];
  const uint2* ssrc[NG];
#pragma unroll
  for (int q = 0; q < NG; q++) {
    // MODE 0: gate tile bx*4+warp, up tile RT0/2 + same; MODE 1: tile bx*4+warp
    const int tile = MODE == 0 ? (q == 0 ? bx * 4 + warp : RT0 / 2 + bx * 4 + warp) : bx * 4 + warp;
    wsrc[q] = Wq + ((size_t)e * RT + tile) * KC * 32 + lane;
    ssrc[q] = Sq + ((size_t)e * RT + tile) * KCG * 32 + lane;
  }
  // x staging: granule id = tid + 128 * u : token = id >> 3, k chunk = id & 7
  const bf16* xb[NCH / 2];
  bool xok[NCH / 2];
#pragma unroll
  for (int u = 0; u < NCH / 2; u++) {
    const int id = tid + 128 * u;
    const int col = id >> 3;
    xok[u] = col < ntok;
    xb[u] = X;
    if (xok[u]) {
      const int p = __ldcg(lst + col);
      xb[u] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + (id & 7) * 8;
    }
  }
  auto issue = [&](int c) {
    if (c < KC) {
#pragma unroll
      for (int q = 0; q < NG; q++) cp16(&wring[(((c % S) * 4 + warp) * NG + q) * 32 + lane], wsrc[q] + c * 32);
#pragma unroll
      for (int u = 0; u < NCH / 2; u++) {
        const int id = tid + 128 * u;
        cp16z(xring + (c % S) * XST + (id >> 6) * 1024 + (id & 7) * 128 + ((id >> 3) & 7) * 16, xb[u] + c * 64, xok[u] ? 16u : 0u);
      }
    }
    cp_commit();
  };
  float acc[NG][N / 2];
#pragma unroll
  for (int q = 0; q < NG; q++)
#pragma unroll
    for (int i = 0; i < N / 2; i++) acc[q][i] = 0.f;
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
  uint2 scur[NG], snxt[NG];
#pragma unroll
  for (int q = 0; q < NG; q++) {
    scur[q] = __ldg(ssrc[q]);
    snxt[q] = __ldg(ssrc[q] + 32);
  }
#pragma unroll 1
  for (int i = 0; i < KC; i++) {
    cp_wait<S - 2>();
    proxy_fence();
    __syncthreads();
    issue(i + S - 1);
    const int cq = i & 3;
    uint32_t a[NG][4][4];
#pragma unroll
    for (int q = 0; q < NG; q++) {
      const uint32_t s0 = scale2(scur[q].x, cq);
      const uint32_t s1 = scale2(scur[q].y, cq);
      const uint4 wv = wring[(((i % S) * 4 + warp) * NG + q) * 32 + lane];
      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        a[q][ks][0] = hmul(dec<0>(ww[ks]), s0);
        a[q][ks][1] = hmul(dec<1>(ww[ks]), s1);
        a[q][ks][2] = hmul(dec<2>(ww[ks]), s0);
        a[q][ks][3] = hmul(dec<3>(ww[ks]), s1);
      }
    }
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
#pragma unroll
      for (int q = 0; q < NG; q++) WgN<N>::mma(acc[q], a[q][ks][0], a[q][ks][1], a[q][ks][2], a[q][ks][3], wg_desc(xring + (i % S) * XST + ks * 256, 128, 1024));
    wg_commit_wait();
    if (cq == 3) {
#pragma unroll
      for (int q = 0; q < NG; q++) {
        scur[q] = snxt[q];
        if ((i >> 2) + 2 < KCG) snxt[q] = __ldg(ssrc[q] + ((i >> 2) + 2) * 32);
      }
    }
  }
  cp_wait<0>();
  const float gs = GS[e];
#pragma unroll
  for (int jb = 0; jb < N / 8; jb++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int col = jb * 8 + 2 * t + (r & 1);
      if (col >= ntok) continue;
      const int p = __ldcg(lst + col);
      const int row = warp * 16 + g + 8 * (r >> 1);
      if (MODE == 0) {
        const float gate = bfr(bfr(acc[0][jb * 4 + r] * 256.0f) * gs), up = bfr(bfr(acc[1][jb * 4 + r] * 256.0f) * gs);
        const float sil = gate / (1.0f + __expf(-gate));
        OUT[(size_t)p * ID + wg_perm_k(bx * 64 + row)] = __float2bfloat16(sil * up);
      } else {
        OUT[(size_t)p * HD + bx * 64 + row] = __float2bfloat16(bfr(acc[0][jb * 4 + r] * 256.0f) * gs);
      }
    }
}

template <int MODE, int NMAX, int S>
__global__ void __launch_bounds__(128, 2)
wm_gemm_wg(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ GS, const bf16* __restrict__ X,
           bf16* __restrict__ OUT, const int* __restrict__ lst, const int* __restrict__ act, int lmax) {
  extern __shared__ __align__(16) uint8_t smem_wg[];
  pdl_wait();
  pdl_trigger();
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);
  if (item.z == 0) return;
  const int e = item.x, ntok = item.z;
  const int* l = lst + e * lmax + item.y;
  if (ntok <= 32) wm_body_wg<MODE, 4, S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);
  else if (NMAX >= 8 && ntok <= 64) wm_body_wg<MODE, (NMAX >= 8 ? 8 : 4), S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);
  else if (NMAX >= 16) wm_body_wg<MODE, (NMAX >= 16 ? 16 : 4), S>(Wq, Sq, GS, X, OUT, l, e, ntok, smem_wg, blockIdx.x);
}

template <int MODE, int NMAX, int S>
static void launch_wg(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst, const int* act,
                      int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid(64, (unsigned)maxslots);
  constexpr int NG = MODE == 0 ? 2 : 1;
  const int dyn = S * (NMAX * 8 * 128 + 4 * NG * 512);
  cudaFuncSetAttribute(wm_gemm_wg<MODE, NMAX, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(wm_gemm_wg<MODE, NMAX, S>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const uint2*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);
}

// Post-expert RMSNorm + weighted sum: out[m] = sum_k w[m,k] * rmsnorm_k(y[m,k] * nw), skipping slots routed elsewhere.
// One CTA per token; warp k owns slot k for the sum of squares, then all threads accumulate their columns.
__global__ void __launch_bounds__(256)
wm_finish_kernel(const bf16* __restrict__ Y, const float* __restrict__ TW, const int* __restrict__ lids,
                 const bf16* __restrict__ NW_, bf16* __restrict__ OUT, float eps) {
  __shared__ float s_scale[8];
  pdl_wait();
  pdl_trigger();
  const int m = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool live = lids[m * 8 + warp] >= 0;
  float ss = 0.f;
  if (live) {
    const uint4* yv = reinterpret_cast<const uint4*>(Y + ((size_t)m * 8 + warp) * HD_ * 1);
    const uint4* wv = reinterpret_cast<const uint4*>(NW_);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const uint4 a = __ldcg(yv + i * 32 + lane), w = __ldg(wv + i * 32 + lane);
      const bf162* ap = reinterpret_cast<const bf162*>(&a);
      const bf162* wp = reinterpret_cast<const bf162*>(&w);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float2 af = __bfloat1622float2(ap[j]), wf = __bfloat1622float2(wp[j]);
        const float v0 = af.x * wf.x, v1 = af.y * wf.y;
        ss += v0 * v0 + v1 * v1;
      }
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
  }
  if (lane == 0) s_scale[warp] = live ? TW[m * 8 + warp] * rsqrtf(ss / HD_ + eps) : 0.f;
  __syncthreads();
  // each thread: 16 columns = two uint4 of bf16
  float acc[16];
#pragma unroll
  for (int j = 0; j < 16; j++) acc[j] = 0.f;
  const uint4* wv = reinterpret_cast<const uint4*>(NW_) + tid * 2;
  const uint4 w0 = __ldg(wv), w1 = __ldg(wv + 1);
  float wf[16];
  {
    const bf162* p0 = reinterpret_cast<const bf162*>(&w0);
    const bf162* p1 = reinterpret_cast<const bf162*>(&w1);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float2 a = __bfloat1622float2(p0[j]), b = __bfloat1622float2(p1[j]);
      wf[2 * j] = a.x; wf[2 * j + 1] = a.y; wf[8 + 2 * j] = b.x; wf[8 + 2 * j + 1] = b.y;
    }
  }
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const float sc = s_scale[k];
    if (sc != 0.f) {
      const uint4* yv = reinterpret_cast<const uint4*>(Y + ((size_t)m * 8 + k) * HD_) + tid * 2;
      const uint4 a0 = __ldcg(yv), a1 = __ldcg(yv + 1);
      const bf162* p0 = reinterpret_cast<const bf162*>(&a0);
      const bf162* p1 = reinterpret_cast<const bf162*>(&a1);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float2 a = __bfloat1622float2(p0[j]), b = __bfloat1622float2(p1[j]);
        acc[2 * j] += sc * (a.x * wf[2 * j]);
        acc[2 * j + 1] += sc * (a.y * wf[2 * j + 1]);
        acc[8 + 2 * j] += sc * (b.x * wf[8 + 2 * j]);
        acc[8 + 2 * j + 1] += sc * (b.y * wf[8 + 2 * j + 1]);
      }
    }
  }
  uint4 o0, o1;
  bf162* q0 = reinterpret_cast<bf162*>(&o0);
  bf162* q1 = reinterpret_cast<bf162*>(&o1);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    q0[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
    q1[j] = __floats2bfloat162_rn(acc[8 + 2 * j], acc[8 + 2 * j + 1]);
  }
  uint4* ov = reinterpret_cast<uint4*>(OUT + (size_t)m * HD_) + tid * 2;
  ov[0] = o0;
  ov[1] = o1;
}
extern "C" void wm_finish_c(const void* Y, const float* TW, const int* lids, const void* NWT, void* OUT, int M, float eps,
                            cudaStream_t st) {
  launch_k(wm_finish_kernel, dim3(M), dim3(256), 0, st, g_pdl != 0, reinterpret_cast<const bf16*>(Y), TW, lids,
           reinterpret_cast<const bf16*>(NWT), reinterpret_cast<bf16*>(OUT), eps);
}

extern "C" void wm_route_c(const int* ids, int npair, const int* ls_ptr, int ls_val, int lmax, int* cnt, int* lst,
                           int* act, int* nact, int* local_ids, int* woff, int grp, int nslots, cudaStream_t st) {
  const int pdl = g_pdl;
  launch_k(route_kernel, dim3(1), dim3(1024), 0, st, pdl != 0, ids, npair, ls_ptr, ls_val, lmax, cnt, lst, act, nact, local_ids, woff, grp, nslots);
}

template <int MODE, int NT, int NMAX, int NW, int RING, int DBG>
static void launch(const void* W, const void* S, const float* GS, const void* X, void* OUT, const int* cnt,
                   const int* lst, const int* act, const int* nact, const int* woff, int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid((MODE == 0 ? 512 : 256) / NT, (unsigned)maxslots);
  if (g_persist) { const unsigned cap = 132u * (unsigned)g_persist; const unsigned tot = grid.x * grid.y; grid = dim3(tot < cap ? tot : cap, 1); }
  int dyn = NW * NT * 6 * 512 > 49152 ? 49152 : NW * NT * 6 * 512;
  if (NT == 8) dyn = NW * 3 * 8 * 512 + 1024;
  if (RING == 3) { const int d3 = NW * 4 * 4 * 512; dyn = d3 > dyn ? d3 : dyn; if (NW * NT * 4 * 512 > dyn) dyn = NW * NT * 4 * 512; }
  if (RING == 2) {
    int per = ring2_stages<NT, 1>() * (NT + 2);
    if (NMAX >= 2) { const int p2 = ring2_stages<2, 2>() * (2 + 4); per = per > p2 ? per : p2; }
    if (NMAX >= 4) { const int p4 = ring2_stages<2, 4>() * (2 + 8); per = per > p4 ? per : p4; }
    dyn = NW * per * 512;
  }
  cudaFuncSetAttribute(wm_gemm<MODE, NT, NMAX, NW, RING, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(wm_gemm<MODE, NT, NMAX, NW, RING, DBG>, grid, dim3(NW * 32), (size_t)dyn, st, pdl != 0,
           reinterpret_cast<const uint4*>(W), reinterpret_cast<const uint2*>(S), GS, reinterpret_cast<const bf16*>(X),
           reinterpret_cast<bf16*>(OUT), cnt, lst, act, nact, woff, lmax, g_persist);
}

// cfg bits: nmax (1,2,4) | nt (2,4) | nw (4,8) | ring (0/1)
extern "C" int wm_gemm_c(int mode, int nmax, int nt, int nw, int ring, int dbg, const void* W, const void* S, const float* GS,
                         const void* X, void* OUT, const int* cnt, const int* lst, const int* act, const int* nact,
                         const int* woff, int maxslots, int lmax, cudaStream_t st) {
  const int pdl = g_pdl;
#define WM_WG(M, N_, S_) if (mode == M && ring == 7 && nmax == N_ && nt == S_ && dbg == 0) { launch_wg<M, N_, S_>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
#undef WM_WG
#define WM_RS(M, T, N, S_) \
  if (mode == M && ring == 4 && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RS(0, 4, 4, 4) WM_RS(1, 4, 4, 4) WM_RS(0, 4, 2, 4) WM_RS(1, 4, 2, 4) WM_RS(0, 4, 1, 4) WM_RS(1, 4, 1, 4)
  WM_RS(0, 2, 4, 4) WM_RS(1, 2, 4, 4) WM_RS(0, 2, 2, 4) WM_RS(1, 2, 2, 4) WM_RS(0, 2, 1, 4) WM_RS(1, 2, 1, 4)
  WM_RS(0, 2, 8, 4) WM_RS(1, 2, 8, 4) WM_RS(0, 2, 8, 3) WM_RS(1, 2, 8, 3) WM_RS(0, 4, 8, 3) WM_RS(1, 4, 8, 3)
#undef WM_RS
#define WM_RS(M, T, N, S_, R_) \
  if (mode == M && ring == R_ && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RS(0, 4, 4, 6, 5) WM_RS(1, 4, 4, 6, 5) WM_RS(0, 2, 4, 6, 5) WM_RS(1, 2, 4, 6, 5) WM_RS(0, 4, 4, 8, 6) WM_RS(1, 4, 4, 8, 6) WM_RS(0, 2, 4, 8, 6) WM_RS(1, 2, 4, 8, 6)
  WM_RS(0, 4, 2, 6, 5) WM_RS(1, 4, 2, 6, 5) WM_RS(0, 4, 1, 6, 5) WM_RS(1, 4, 1, 6, 5)
  WM_RS(0, 4, 4, 3, 8) WM_RS(1, 4, 4, 3, 8) WM_RS(0, 2, 4, 3, 8) WM_RS(1, 2, 4, 3, 8) WM_RS(0, 2, 8, 3, 9) WM_RS(1, 2, 8, 3, 9) WM_RS(0, 2, 4, 3, 9) WM_RS(1, 2, 4, 3, 9)
#undef WM_RS
#define WM_RSD(M, D) if (mode == M && ring == 4 && nt == 4 && nmax == 4 && dbg == D) { launch_rs<M, 4, 4, 4, D>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RSD(0, 1) WM_RSD(1, 1) WM_RSD(0, 2) WM_RSD(1, 2) WM_RSD(0, 8) WM_RSD(1, 8) WM_RSD(0, 4) WM_RSD(1, 4) WM_RSD(0, 15) WM_RSD(1, 15) WM_RSD(0, 3) WM_RSD(1, 3) WM_RSD(0, 16) WM_RSD(1, 16) WM_RSD(0, 32) WM_RSD(1, 32) WM_RSD(0, 48) WM_RSD(1, 48) WM_RSD(0, 63) WM_RSD(1, 63) WM_RSD(0, 31) WM_RSD(1, 31) WM_RSD(0, 64) WM_RSD(1, 64)
#undef WM_RSD
#define WM_D(M, T, N, W_, R)                                                                             \
  if (mode == M && nt == T && nmax == N && nw == W_ && ring == R && dbg == 0) {                           \
    launch<M, T, N, W_, R, 0>(W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax, pdl, st);          \
    return 0;                                                                                             \
  }
#define WM_DBG(M, T, N, W_, R, D)                                                                         \
  if (mode == M && nt == T && nmax == N && nw == W_ && ring == R && dbg == D) {                           \
    launch<M, T, N, W_, R, D>(W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax, pdl, st);          \
    return 0;                                                                                             \
  }
#define WM_DBGS(M, T, N, W_, R) WM_DBG(M, T, N, W_, R, 1) WM_DBG(M, T, N, W_, R, 2) WM_DBG(M, T, N, W_, R, 8) WM_DBG(M, T, N, W_, R, 3) WM_DBG(M, T, N, W_, R, 11) WM_DBG(M, T, N, W_, R, 4) WM_DBG(M, T, N, W_, R, 9)
#define WM_ALL(T, N, W_, R) WM_D(0, T, N, W_, R) WM_D(1, T, N, W_, R)
  WM_ALL(2, 1, 4, 0) WM_ALL(2, 2, 4, 0) WM_ALL(2, 1, 8, 0) WM_ALL(2, 2, 8, 0) WM_ALL(4, 1, 4, 0) WM_ALL(4, 2, 4, 0)
  WM_ALL(2, 1, 4, 1) WM_ALL(2, 2, 4, 1) WM_ALL(2, 4, 4, 1) WM_ALL(4, 1, 4, 1) WM_ALL(4, 2, 4, 1) WM_ALL(4, 4, 4, 1)
  WM_ALL(2, 1, 8, 1) WM_ALL(2, 2, 8, 1) WM_ALL(2, 4, 8, 1)
  WM_ALL(8, 1, 4, 1) WM_ALL(8, 2, 4, 1) WM_ALL(8, 4, 4, 1)
  WM_ALL(4, 4, 4, 3) WM_ALL(4, 2, 4, 3) WM_ALL(4, 1, 4, 3) WM_ALL(8, 4, 4, 3) WM_ALL(8, 2, 4, 3)
  WM_ALL(4, 1, 2, 1) WM_ALL(4, 2, 2, 1) WM_ALL(4, 4, 2, 1) WM_ALL(2, 1, 2, 0) WM_ALL(2, 2, 2, 0) WM_ALL(2, 1, 2, 1)
  WM_ALL(2, 1, 4, 2) WM_ALL(2, 2, 4, 2) WM_ALL(2, 4, 4, 2) WM_ALL(4, 1, 4, 2) WM_ALL(4, 2, 4, 2) WM_ALL(4, 4, 4, 2)
  WM_ALL(2, 1, 8, 2) WM_ALL(2, 2, 8, 2) WM_ALL(2, 4, 8, 2)
#ifdef WM_DBG_BUILD
  WM_DBGS(0, 2, 1, 4, 0) WM_DBGS(1, 2, 1, 4, 0) WM_DBGS(0, 2, 1, 4, 2) WM_DBGS(1, 2, 1, 4, 2) WM_DBGS(0, 2, 2, 4, 2) WM_DBGS(1, 2, 2, 4, 2)
  WM_DBGS(0, 4, 4, 4, 2) WM_DBGS(1, 4, 4, 4, 2) WM_DBGS(0, 2, 4, 4, 2) WM_DBGS(1, 2, 4, 4, 2)
#endif
#undef WM_ALL
#undef WM_D
  return 1;
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" void wm_route_c(const int*, int, const int*, int, int, int*, int*, int*, int*, int*, int*, int, int, cudaStream_t);
extern "C" void wm_set_pdl(int);
void wm_set_pdl_t(int64_t v) { wm_set_pdl((int)v); }
extern "C" void wm_set_persist(int);
void wm_set_persist_t(int64_t v) { wm_set_persist((int)v); }
extern "C" void wm_finish_c(const void*, const float*, const int*, const void*, void*, int, float, cudaStream_t);
void wm_finish(torch::Tensor y, torch::Tensor tw, torch::Tensor lids, torch::Tensor nw, torch::Tensor out, double eps) {
  wm_finish_c(y.data_ptr(), tw.data_ptr<float>(), lids.data_ptr<int>(), nw.data_ptr(), out.data_ptr(), (int)out.size(0), (float)eps,
              at::cuda::getCurrentCUDAStream());
}
extern "C" void wm_permx_c(const void*, void*, int, cudaStream_t);
void wm_permx(torch::Tensor x, torch::Tensor xp) { wm_permx_c(x.data_ptr(), xp.data_ptr(), (int)x.size(0), at::cuda::getCurrentCUDAStream()); }
extern "C" int wm_gemm_c(int, int, int, int, int, int, const void*, const void*, const float*, const void*, void*, const int*,
                         const int*, const int*, const int*, const int*, int, int, cudaStream_t);

void wm_route(torch::Tensor ids, torch::Tensor ls, int64_t ls_val, int64_t lmax, torch::Tensor cnt, torch::Tensor lst,
              torch::Tensor act, torch::Tensor nact, torch::Tensor local_ids, torch::Tensor woff, int64_t grp) {
  wm_route_c(ids.data_ptr<int>(), (int)ids.numel(), ls.numel() ? ls.data_ptr<int>() : nullptr, (int)ls_val, (int)lmax, cnt.data_ptr<int>(),
             lst.data_ptr<int>(), act.data_ptr<int>(), nact.data_ptr<int>(), local_ids.data_ptr<int>(), woff.data_ptr<int>(), (int)grp,
             (int)(act.numel() / 4), at::cuda::getCurrentCUDAStream());
}

void wm_gemm_run(int64_t mode, int64_t nmax, int64_t nt, int64_t nw, int64_t ring, int64_t dbg, torch::Tensor W, torch::Tensor S, torch::Tensor GS, torch::Tensor X,
                 torch::Tensor OUT, torch::Tensor cnt, torch::Tensor lst, torch::Tensor act, torch::Tensor nact,
                 torch::Tensor woff, int64_t maxslots, int64_t lmax) {
  int r = wm_gemm_c((int)mode, (int)nmax, (int)nt, (int)nw, (int)ring, (int)dbg, W.data_ptr(), S.data_ptr(), GS.data_ptr<float>(), X.data_ptr(),
                    OUT.data_ptr(), cnt.data_ptr<int>(), lst.data_ptr<int>(), act.data_ptr<int>(),
                    nact.data_ptr<int>(), woff.data_ptr<int>(), (int)maxslots, (int)lmax, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported config");
}
'''
_WM_CPP = r'''
void wm_set_pdl_t(int64_t v);
void wm_set_persist_t(int64_t v);
void wm_permx(torch::Tensor x, torch::Tensor xp);
void wm_finish(torch::Tensor y, torch::Tensor tw, torch::Tensor lids, torch::Tensor nw, torch::Tensor out, double eps);
void wm_route(torch::Tensor ids, torch::Tensor ls, int64_t ls_val, int64_t lmax, torch::Tensor cnt, torch::Tensor lst,
              torch::Tensor act, torch::Tensor nact, torch::Tensor local_ids, torch::Tensor woff, int64_t grp);
void wm_gemm_run(int64_t mode, int64_t nmax, int64_t nt, int64_t nw, int64_t ring, int64_t dbg, torch::Tensor W, torch::Tensor S, torch::Tensor GS, torch::Tensor X,
                 torch::Tensor OUT, torch::Tensor cnt, torch::Tensor lst, torch::Tensor act, torch::Tensor nact,
                 torch::Tensor woff, int64_t maxslots, int64_t lmax);
'''

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
_WM_EXTRA = []
_WM_PDL = True
_WM_FIN = False
_WM_CFG = [(1,(2,4,0)),(8,(2,2,0)),(16,(2,4,0)),(32,(4,4,1)),(10**9,(4,4,4))]


def _wm_cfg(m):
    for bound, c in _WM_CFG:
        if m <= bound:
            return c
    return _WM_CFG[-1][1]


def _wm_module():
    global _WM_MOD
    if _WM_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _WM_MOD = load_inline("muse_wm_moe", _WM_CPP, cuda_sources=_WM_CUDA, functions=["wm_route", "wm_gemm_run", "wm_set_pdl_t", "wm_set_persist_t", "wm_finish"],
                              extra_cuda_cflags=["-O3", "--use_fast_math"] + _WM_EXTRA, verbose=False)
        _WM_MOD.wm_set_pdl_t(int(_WM_PDL and torch.cuda.get_device_capability()[0] >= 9))
    return _WM_MOD


def _wm_prepare(weights):
    w13, s13 = _wm_repack(weights["w13"], weights["w13_sf"])
    w2, s2 = _wm_repack(weights["w2"], weights["w2_sf"])
    return {"mod": _wm_module(), "w13": w13, "s13": s13, "gs13": weights["w13_gs"].contiguous(),
            "w2": w2, "s2": s2, "gs2": weights["w2_gs"].contiguous(),
            "nols": torch.empty(0, dtype=torch.int32, device=weights["w13"].device)}


def _experts_wm(x, topk_weights, local_mod_ids, state):
    wm = state["wm"]
    mod = wm["mod"]
    m = x.shape[0]
    x = x.contiguous()
    nch = 1 if m <= 8 else 2 if m <= 16 else 4
    lmax = (m + 7) // 8 * 8
    nt, nw, ring = _wm_cfg(m)
    dev = x.device
    ids = local_mod_ids.to(torch.int32).contiguous()
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(512, dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    mod.wm_route(ids, wm["nols"], 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
    maxslots = min(m * TOP_K, 32 + (m * TOP_K) // (nch * 8))
    inter = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=dev)
    mod.wm_gemm_run(0, nch, nt, nw, ring, _WM_DBG, wm["w13"], wm["s13"], wm["gs13"], x, inter, cnt, lst, act, nact, woff, maxslots, lmax)
    mod.wm_gemm_run(1, nch, nt, nw, ring, _WM_DBG, wm["w2"], wm["s2"], wm["gs2"], inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if _WM_FIN:
        out = torch.empty((m, D_EXPERT), dtype=torch.bfloat16, device=dev)
        mod.wm_finish(y, topk_weights, lids, state["post_expert_w"], out, POST_NORM_EPS)
        return out
    return _finish_experts(
        y.view(m, TOP_K, D_EXPERT), topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=lids.view(m, TOP_K))


WM_MAX_TOKENS = 256


# ---- CUDA attention (fd2 / fused sliding) ----
_FD_CUDA = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
static int g_fd_pdl = 0;
extern "C" void fd_set_pdl(int v) { g_fd_pdl = v; }
template <typename... KArgs, typename... Args>
static void launch_pdl(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_fd_pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

#define HD 64
#define STAGE 16          // tokens per warp stage
#define KP 144            // bytes per converted f16 row (64 halves + 16 pad)
#define RP 80             // bytes per raw fp8 row slot (64 + 16 pad)

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t hmul2_h(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// four e4m3 bytes -> two f16x2 (exact)
__device__ __forceinline__ void fp8x4_to_h2(uint32_t w, uint32_t& lo, uint32_t& hi) {
#if defined(FD_HW_CVT) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n}"
      : "=r"(lo), "=r"(hi) : "r"(w));
  return;
#endif
  uint32_t tl = prmt(w, 0u, 0x1404u), th = prmt(w, 0u, 0x3424u);
  tl = (tl & 0x80008000u) | ((tl >> 1) & 0x3F803F80u);
  th = (th & 0x80008000u) | ((th >> 1) & 0x3F803F80u);
  lo = hmul2_h(tl, 0x5C005C00u);  // x 2^8 undoes the exponent-bias shift
  hi = hmul2_h(th, 0x5C005C00u);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// grid (splits, HKV, B), 128 threads.  Each warp owns every 4th 16-token stage of the CTA's range.
template <int MT, int S, bool QK8>
__global__ void __launch_bounds__(128)
fd_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
            const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
            const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
            int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = MT * 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;  // scores in log2 domain
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;

  const int nstages = end > begin ? (end - begin + STAGE - 1) / STAGE : 0;

  // ---- per-warp shared layout ----
  constexpr int RAW = 2 * 16 * RP;                       // K and V raw slot
  constexpr int PER_WARP = S * RAW + 2 * 16 * KP;        // ring + converted K and V
  uint8_t* wbase = smem + warp * PER_WARP;
  uint8_t* kf = wbase + S * RAW;
  uint8_t* vf = kf + 16 * KP;

  // ---- Q fragments (f16, or raw e4m3 for the fp8 mma path) ----
  uint32_t qa[MT][4][4];
  if constexpr (QK8) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int row = mt * 16 + g + 8 * half;
        const int token = row >> 3, qh = kh * 8 + (row & 7);
        const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          uint32_t x0 = 0, x1 = 0;
          if (token < L) {
            x0 = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
            x1 = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
          }
          qa[mt][ks][half ? 1 : 0] = x0;  // a0 / a1
          qa[mt][ks][half ? 3 : 2] = x1;  // a2 / a3
        }
      }
    }
  } else {
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int row = mt * 16 + g + 8 * half;
      const int token = row >> 3, qh = kh * 8 + (row & 7);
      const bool ok = token < L;
      const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        // elements d = 16 ks + 2t,2t+1 (a0/a1) and 16 ks + 8 + 2t,2t+1 (a2/a3)
        uint32_t lo = 0, hi = 0, x = 0, y = 0;
        if (ok) {
          uint32_t v0 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 2 * t);
          uint32_t v1 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 8 + 2 * t);
          fp8x4_to_h2(v0 | (v1 << 16), lo, hi);
          x = lo;
          y = hi;
        }
        qa[mt][ks][half ? 1 : 0] = x;       // a0 (row g) / a1 (row g+8): first column pair
        qa[mt][ks][half ? 3 : 2] = y;       // a2 / a3: second column pair
      }
    }
  }
  }

  float o[MT][8][4];
  float m_run[MT][2], l_run[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    m_run[mt][0] = m_run[mt][1] = -INFINITY;
    l_run[mt][0] = l_run[mt][1] = 0.f;
#pragma unroll
    for (int d = 0; d < 8; d++)
#pragma unroll
      for (int r = 0; r < 4; r++) o[mt][d][r] = 0.f;
  }

  const int pshift = __ffs(page) - 1, pmask = page - 1;
  auto row_of = [&](int tok) -> long {
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    return phys < 0 ? -1L : ((long)phys << pshift) + (tok & pmask);
  };
  // stage s of this warp covers tokens begin + (warp + 4 s) * STAGE ...
  auto issue = [&](int s, long r0, long r1) {
    if (s < (nstages - warp + 3) / 4) {
      uint8_t* slot = wbase + (s % S) * RAW;
      const int c = lane & 3;
      const int tok0 = lane >> 2;
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const long r = i ? r1 : r0;
        const int tok = tok0 + 8 * i;
        // invalid rows read row 0 (masked later)
        const long rr = r < 0 ? 0 : r;
        cp16(slot + tok * RP + c * 16, KC + (rr * HKV + kh) * HD + c * 16);
        cp16(slot + 16 * RP + tok * RP + c * 16, VC + (rr * HKV + kh) * HD + c * 16);
      }
    }
    cp_commit();
  };

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto stage_tok = [&](int s) { return begin + (warp + 4 * s) * STAGE; };
  // gather-index prefetch for stage s: tokens stage_tok(s) + lane/4 and +8
  long ra0 = -1, ra1 = -1, rb0 = -1, rb1 = -1;
  auto load_rows = [&](int s, long& x0, long& x1) {
    const int tb = stage_tok(s) + (lane >> 2);
    x0 = x1 = -1;
    if (s < my_stages) {
      if (tb < end) x0 = row_of(tb);
      if (tb + 8 < end) x1 = row_of(tb + 8);
    }
  };
  {
    long pr0[S], pr1[S];
#pragma unroll
    for (int s = 0; s < S; s++) load_rows(s, pr0[s], pr1[s]);
#pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, pr0[s], pr1[s]);
    rb0 = pr0[S - 1];
    rb1 = pr1[S - 1];
  }

  const int srow = lane & 7;  // ldmatrix row within an 8x8 tile
  const int smat = lane >> 3; // matrix index within x4

#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1, rb0, rb1);
    load_rows(s + S, rb0, rb1);
    const uint8_t* slot = wbase + (s % S) * RAW;
    // ---- convert K and V rows of this stage to f16 (lane handles one 16B raw chunk x 2) ----
    {
      const int tk = lane >> 1, half16 = (lane & 1) * 32;  // 16 tokens x 2 halves of 32 bytes
#pragma unroll
      for (int which = (QK8 ? 1 : 0); which < 2; which++) {
        const uint8_t* src = slot + which * 16 * RP + tk * RP + half16;
        uint8_t* dst = (which ? vf : kf) + tk * KP + half16 * 2;
        const uint4 u0 = *reinterpret_cast<const uint4*>(src);
        const uint4 u1 = *reinterpret_cast<const uint4*>(src + 16);
        uint4 o0, o1, o2, o3;
        fp8x4_to_h2(u0.x, o0.x, o0.y);
        fp8x4_to_h2(u0.y, o0.z, o0.w);
        fp8x4_to_h2(u0.z, o1.x, o1.y);
        fp8x4_to_h2(u0.w, o1.z, o1.w);
        fp8x4_to_h2(u1.x, o2.x, o2.y);
        fp8x4_to_h2(u1.y, o2.z, o2.w);
        fp8x4_to_h2(u1.z, o3.x, o3.y);
        fp8x4_to_h2(u1.w, o3.z, o3.w);
        *reinterpret_cast<uint4*>(dst) = o0;
        *reinterpret_cast<uint4*>(dst + 16) = o1;
        *reinterpret_cast<uint4*>(dst + 32) = o2;
        *reinterpret_cast<uint4*>(dst + 48) = o3;
      }
    }
    __syncwarp();

    const int tb = stage_tok(s);
    const bool full_vis = (tb + STAGE <= end) && (L == 1 || tb + STAGE - 1 <= prefix) &&
                          (!window || L == 1 || tb > prefix + L - 1 - window);
    // ---- S = Q K^T for each m-tile: 2 n-tiles of 8 tokens ----
    uint32_t kb[2][4][2];
    if constexpr (QK8) {
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          const uint8_t* kr = slot + (nt * 8 + g) * RP + 32 * ks + 4 * t;
          kb[nt][ks][0] = *reinterpret_cast<const uint32_t*>(kr);
          kb[nt][ks][1] = *reinterpret_cast<const uint32_t*>(kr + 16);
        }
    } else {
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4(r0, r1, r2, r3, kf + (nt * 8 + srow) * KP + pr * 64 + smat * 16);
        kb[nt][2 * pr][0] = r0; kb[nt][2 * pr][1] = r1;
        kb[nt][2 * pr + 1][0] = r2; kb[nt][2 * pr + 1][1] = r3;
      }
    }
    uint32_t vb[4][4];
#pragma unroll
    for (int dp = 0; dp < 4; dp++)
      ldsm4t(vb[dp][0], vb[dp][1], vb[dp][2], vb[dp][3],
             vf + ((smat & 1) * 8 + srow) * KP + dp * 32 + (smat >> 1) * 16);

#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float sacc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      if constexpr (QK8) {
#pragma unroll
        for (int ks = 0; ks < 2; ks++)
#pragma unroll
          for (int nt = 0; nt < 2; nt++)
            mma_e4m3(sacc[nt], qa[mt][ks][0], qa[mt][ks][1], qa[mt][ks][2], qa[mt][ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      } else {
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
          mma_f16(sacc[nt], qa[mt][ks][0], qa[mt][ks][1], qa[mt][ks][2], qa[mt][ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      }

      // masking (only on boundary stages), online softmax (two rows per thread: g and g+8)
      float mx[2] = {-INFINITY, -INFINITY};
      float sv[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sv[nt][r] = sacc[nt][r];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = mt * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sv[nt][r] = -INFINITY;
          }
      }
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sv[nt][r]);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;  // running max lives in the scaled log2 domain
      }
      float corr[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        newm[h] = fmaxf(m_run[mt][h], mx[h]);
        const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[mt][h] - safe);
        newm[h] = safe == 0.f && newm[h] == -INFINITY ? -INFINITY : newm[h];
      }
      uint32_t pa[4];
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int h = r >> 1;
          const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
          p[r] = exp2f(fmaf(sv[nt][r], sc2, -safe));
          rs[h] += p[r];
        }
        pa[nt * 2 + 0] = pack_h2(p[0], p[1]);  // row g
        pa[nt * 2 + 1] = pack_h2(p[2], p[3]);  // row g+8
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[mt][h] = l_run[mt][h] * corr[h] + rs[h];
        m_run[mt][h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[mt][d][0] *= corr[0]; o[mt][d][1] *= corr[0];
          o[mt][d][2] *= corr[1]; o[mt][d][3] *= corr[1];
        }
      }
      // PV: A = P (a0: row g tokens 0-7, a1: row g+8 tokens 0-7, a2: row g tokens 8-15, a3: row g+8 tokens 8-15)
#pragma unroll
      for (int dp = 0; dp < 4; dp++) {
        mma_f16(o[mt][2 * dp], pa[0], pa[1], pa[2], pa[3], vb[dp][0], vb[dp][1]);
        mma_f16(o[mt][2 * dp + 1], pa[0], pa[1], pa[2], pa[3], vb[dp][2], vb[dp][3]);
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();

  // ---- merge the four warps' states ----
  float* sm_m = reinterpret_cast<float*>(smem);                 // [4][BM]
  float* sm_l = sm_m + 4 * BM;                                  // [4][BM]
  float* sm_o = sm_l + 4 * BM;                                  // [4][BM][HD]
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = mt * 16 + g + 8 * h;
      if (t == 0) {
        sm_m[warp * BM + row] = m_run[mt][h];
        sm_l[warp * BM + row] = l_run[mt][h];
      }
#pragma unroll
      for (int d = 0; d < 8; d++) {
        sm_o[(warp * BM + row) * HD + d * 8 + 2 * t] = o[mt][d][2 * h];
        sm_o[(warp * BM + row) * HD + d * 8 + 2 * t + 1] = o[mt][d][2 * h + 1];
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
    const int row = idx / HD, d = idx % HD;
    float mm = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mm = fmaxf(mm, sm_m[w * BM + row]);
    float ls = 0.f, acc = 0.f;
    if (mm != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float f = sm_m[w * BM + row] == -INFINITY ? 0.f : exp2f(sm_m[w * BM + row] - mm);
        ls += sm_l[w * BM + row] * f;
        acc += sm_o[(w * BM + row) * HD + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mm + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}


// Verify blocks: 64 query rows (8 heads x 8 tokens) per KV head -> one warp per 16-row m-tile; the four warps share each
// 16-token K/V stage (converted once per CTA), so no cross-warp merge is needed.
template <int S>
__global__ void __launch_bounds__(128)
fd_partials_v(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
              const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
              const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
              int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 64;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
  const int nstages = end > begin ? (end - begin + STAGE - 1) / STAGE : 0;

  constexpr int RAW = 2 * 16 * RP;
  uint8_t* ring = smem;
  uint8_t* kf = smem + S * RAW;            // [2][16 * KP]
  uint8_t* vf = kf + 2 * 16 * KP;          // [2][16 * KP]

  // Q fragments for this warp's m-tile (rows warp*16 + g and +8)
  uint32_t qa[4][4];
#pragma unroll
  for (int half = 0; half < 2; half++) {
    const int row = warp * 16 + g + 8 * half;
    const int token = row >> 3, qh = kh * 8 + (row & 7);
    const bool ok = token < L;
    const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      uint32_t lo = 0, hi = 0;
      if (ok) {
        uint32_t v0 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 2 * t);
        uint32_t v1 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 8 + 2 * t);
        fp8x4_to_h2(v0 | (v1 << 16), lo, hi);
      }
      qa[ks][half ? 1 : 0] = lo;
      qa[ks][half ? 3 : 2] = hi;
    }
  }
  float o[8][4];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int d = 0; d < 8; d++)
#pragma unroll
    for (int r = 0; r < 4; r++) o[d][r] = 0.f;

  const int which = tid >> 6, tk = (tid >> 2) & 15, c4 = tid & 3;
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  auto row_of = [&](int tok) -> long {
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    return phys < 0 ? -1L : ((long)phys << pshift) + (tok & pmask);
  };
  auto load_row = [&](int s) -> long {
    const int tok = begin + s * STAGE + tk;
    return (s < nstages && tok < end) ? row_of(tok) : -1L;
  };
  auto issue = [&](int s, long r) {
    if (s < nstages) {
      const long rr = r < 0 ? 0 : r;
      const uint8_t* base = which ? VC : KC;
      cp16(ring + (s % S) * RAW + which * 16 * RP + tk * RP + c4 * 16, base + (rr * HKV + kh) * HD + c4 * 16);
    }
    cp_commit();
  };
  {
    long pr[S];
#pragma unroll
    for (int s = 0; s < S; s++) pr[s] = load_row(s);
#pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, pr[s]);
    long nxt = pr[S - 1];
    const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
    for (int s = 0; s < nstages; s++) {
      cp_wait<S - 2>();
      issue(s + S - 1, nxt);
      nxt = load_row(s + S);
      // convert this thread's own chunk (K or V row segment) to f16
      {
        const uint8_t* src = ring + (s % S) * RAW + which * 16 * RP + tk * RP + c4 * 16;
        const uint4 u = *reinterpret_cast<const uint4*>(src);
        uint4 o0, o1;
        fp8x4_to_h2(u.x, o0.x, o0.y);
        fp8x4_to_h2(u.y, o0.z, o0.w);
        fp8x4_to_h2(u.z, o1.x, o1.y);
        fp8x4_to_h2(u.w, o1.z, o1.w);
        uint8_t* dst = (which ? vf : kf) + (s & 1) * 16 * KP + tk * KP + c4 * 32;
        *reinterpret_cast<uint4*>(dst) = o0;
        *reinterpret_cast<uint4*>(dst + 16) = o1;
      }
      __syncthreads();
      const uint8_t* kfs = kf + (s & 1) * 16 * KP;
      const uint8_t* vfs = vf + (s & 1) * 16 * KP;
      const int tb = begin + s * STAGE;
      const bool full_vis = (tb + STAGE <= end) && (L == 1 || tb + STAGE - 1 <= prefix) &&
                            (!window || L == 1 || tb > prefix + L - 1 - window);
      uint32_t kb[2][4][2];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int pr2 = 0; pr2 < 2; pr2++) {
          uint32_t r0, r1, r2, r3;
          ldsm4(r0, r1, r2, r3, kfs + (nt * 8 + srow) * KP + pr2 * 64 + smat * 16);
          kb[nt][2 * pr2][0] = r0; kb[nt][2 * pr2][1] = r1;
          kb[nt][2 * pr2 + 1][0] = r2; kb[nt][2 * pr2 + 1][1] = r3;
        }
      uint32_t vb[4][4];
#pragma unroll
      for (int dp = 0; dp < 4; dp++)
        ldsm4t(vb[dp][0], vb[dp][1], vb[dp][2], vb[dp][3], vfs + ((smat & 1) * 8 + srow) * KP + dp * 32 + (smat >> 1) * 16);
      float sacc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
          mma_f16(sacc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      float mx[2] = {-INFINITY, -INFINITY};
      float sv[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sv[nt][r] = sacc[nt][r];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = warp * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sv[nt][r] = -INFINITY;
          }
      }
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sv[nt][r]);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;
      }
      float corr[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        newm[h] = fmaxf(m_run[h], mx[h]);
        const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[h] - safe);
      }
      uint32_t pa[4];
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int h = r >> 1;
          const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
          p[r] = exp2f(fmaf(sv[nt][r], sc2, -safe));
          rs[h] += p[r];
        }
        pa[nt * 2 + 0] = pack_h2(p[0], p[1]);
        pa[nt * 2 + 1] = pack_h2(p[2], p[3]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[h] = l_run[h] * corr[h] + rs[h];
        m_run[h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[d][0] *= corr[0]; o[d][1] *= corr[0];
          o[d][2] *= corr[1]; o[d][3] *= corr[1];
        }
      }
#pragma unroll
      for (int dp = 0; dp < 4; dp++) {
        mma_f16(o[2 * dp], pa[0], pa[1], pa[2], pa[3], vb[dp][0], vb[dp][1]);
        mma_f16(o[2 * dp + 1], pa[0], pa[1], pa[2], pa[3], vb[dp][2], vb[dp][3]);
      }
    }
  }
  cp_wait<0>();
  // rows owned by this warp: no cross-warp merge
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = warp * 16 + g + 8 * h;
    const float ls = l_run[h];
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
    for (int d = 0; d < 8; d++) {
      PART[(base_row + row) * HD + d * 8 + 2 * t] = o[d][2 * h] * inv * value_scale;
      PART[(base_row + row) * HD + d * 8 + 2 * t + 1] = o[d][2 * h + 1] * inv * value_scale;
    }
    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}

template <int S>
static void launch_v(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                     const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                     int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  const size_t dyn = (size_t)S * 2 * 16 * RP + 4 * 16 * KP;
  cudaFuncSetAttribute(fd_partials_v<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  fd_partials_v<S><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                          reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                          table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---------------------------------------------------------------------------------------------------------------
// fd2: decode (one query token, 8 heads per KV head) with 32-token stages.  K and V stay raw FP8 in shared memory:
// K fragments come straight from ldmatrix (e4m3 mma), V fragments from ldmatrix.trans + byte permutes + cvt.
// ---------------------------------------------------------------------------------------------------------------
#define ST2 32
#define RMAX 4096

__device__ __forceinline__ void mma_qk8(float* c, uint32_t qa0, uint32_t qa2, uint32_t b0, uint32_t b1) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(qa0), "r"(0u), "r"(qa2), "r"(0u), "r"(b0), "r"(b1));
#else
  // exact emulation with f16 mma: each fp8 register holds 4 consecutive k, used as two k16 slots
  uint32_t ql, qh, kl, kh2;
  fp8x4_to_h2(qa0, ql, qh);
  fp8x4_to_h2(b0, kl, kh2);
  mma_f16(c, ql, 0u, qh, 0u, kl, kh2);
  fp8x4_to_h2(qa2, ql, qh);
  fp8x4_to_h2(b1, kl, kh2);
  mma_f16(c, ql, 0u, qh, 0u, kl, kh2);
#endif
}
// two e4m3 bytes (low 16 bits of x) -> f16x2
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  uint32_t r;
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %1;\ncvt.rn.f16x2.e4m3x2 %0, a;\n}" : "=r"(r) : "r"(x));
  return r;
#else
  uint32_t lo, hi;
  fp8x4_to_h2(x, lo, hi);
  return lo;
#endif
}

template <int S>
__global__ void __launch_bounds__(128)
fd2_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int HQ, int HKV,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 16;
  constexpr int SLOT = 2 * ST2 * RP;  // K and V raw rows of one stage
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + 1);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
  const int n = end > begin ? end - begin : 0;
  const int nstages = (n + ST2 - 1) / ST2;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + 31) / 32) * 32;
  uint8_t* wbase = smem + rmax_rt * 4 + warp * (S * SLOT);
  // physical cache row of every token of this CTA's range
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  pdl_wait();  // Q and the new cache rows come from the previous kernel
  // Q fragments: row g = query head kh*8+g, e4m3 bytes d = 32 ks + 4 t.. and +16
  uint32_t q0[2], q2[2];
  {
    const uint8_t* qp = Q + ((long)b * HQ + kh * 8 + g) * HD;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      q0[ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
      q2[ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
    }
  }
  __syncthreads();  // rows[] ready

  float o[8][2];  // [d-tile (E0,O0,E1,O1,...)][row g, columns 2t, 2t+1]
  float m_run = -INFINITY, l_run = 0.f;
#pragma unroll
  for (int d = 0; d < 8; d++) o[d][0] = o[d][1] = 0.f;

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + 4 * s) * ST2;  // offset within the CTA range
      const int c = lane & 3;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int tk = (lane >> 2) + 8 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = (rr * HKV + kh) * HD + c * 16;
        cp16(slot + tk * RP + c * 16, KC + goff);
        cp16(slot + ST2 * RP + tk * RP + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    const uint8_t* slot = wbase + (s % S) * SLOT;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP;
    const int tb = begin + (warp + 4 * s) * ST2;
    // ---- S = Q K^T over 4 n-tiles of 8 tokens ----
    float sacc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#pragma unroll
      for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      uint32_t k0, k1, k2, k3;
      ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP + smat * 16);
      mma_qk8(sacc[nt], q0[0], q2[0], k0, k1);
      mma_qk8(sacc[nt], q0[1], q2[1], k2, k3);
    }
    // ---- softmax over the 32 columns of row g ----
    const bool full_vis = tb + ST2 <= end;
    float sv[4][2];
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int r = 0; r < 2; r++) {
        float v = sacc[nt][r];
        if (!full_vis && tb + nt * 8 + 2 * t + r >= end) v = -INFINITY;
        sv[nt][r] = v;
        mx = fmaxf(mx, v);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx *= sc2;
    const float newm = fmaxf(m_run, mx);
    const float safe = newm == -INFINITY ? 0.f : newm;
    const float corr = exp2f(m_run - safe);
    float rs = 0.f;
    uint32_t pa[4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
      rs += p0 + p1;
      pa[nt] = pack_h2(p0, p1);
    }
    rs += __shfl_xor_sync(0xffffffffu, rs, 1);
    rs += __shfl_xor_sync(0xffffffffu, rs, 2);
    l_run = l_run * corr + rs;
    m_run = newm;
    if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
      for (int d = 0; d < 8; d++) { o[d][0] *= corr; o[d][1] *= corr; }
    }
    // ---- O += P V: two 16-token k-tiles ----
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        // pr selects d bytes [32 pr, 32 pr + 32): matrices (tok 0-7 | 8-15) x (m = 2 pr, 2 pr + 1)
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP + pr * 32 + (smat >> 1) * 16);
        // r0: (tok 0-7, m=2pr)  r1: (tok 8-15, m=2pr)  r2: (tok 0-7, m=2pr+1)  r3: (tok 8-15, m=2pr+1)
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
        float acc[4];
        acc[0] = o[4 * pr + 0][0]; acc[1] = o[4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
        o[4 * pr + 0][0] = acc[0]; o[4 * pr + 0][1] = acc[1];
        acc[0] = o[4 * pr + 1][0]; acc[1] = o[4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
        o[4 * pr + 1][0] = acc[0]; o[4 * pr + 1][1] = acc[1];
        acc[0] = o[4 * pr + 2][0]; acc[1] = o[4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
        o[4 * pr + 2][0] = acc[0]; o[4 * pr + 2][1] = acc[1];
        acc[0] = o[4 * pr + 3][0]; acc[1] = o[4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
        o[4 * pr + 3][0] = acc[0]; o[4 * pr + 3][1] = acc[1];
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();
  pdl_trigger();

  // ---- merge the four warps (rows g only: 8 query heads) ----
  float* sm_m = reinterpret_cast<float*>(smem);   // [4][8]
  float* sm_l = sm_m + 32;                        // [4][8]
  float* sm_o = sm_l + 32;                        // [4][8][64]
  if (t == 0) {
    sm_m[warp * 8 + g] = m_run;
    sm_l[warp * 8 + g] = l_run;
  }
#pragma unroll
  for (int mm = 0; mm < 4; mm++) {
    // tile 2mm = E (d = 16 mm + 2 n), tile 2mm+1 = O (d = 16 mm + 2 n + 1), n = 2t + c
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[2 * mm][cc];
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[2 * mm + 1][cc];
    }
  }
  __syncthreads();
  for (int idx = tid; idx < 8 * HD; idx += 128) {
    const int row = idx / HD, d = idx % HD;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mmx = fmaxf(mmx, sm_m[w * 8 + row]);
    float ls = 0.f, acc = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float f = sm_m[w * 8 + row] == -INFINITY ? 0.f : exp2f(sm_m[w * 8 + row] - mmx);
        ls += sm_l[w * 8 + row] * f;
        acc += sm_o[(w * 8 + row) * 64 + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mmx + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}


// ---- fd3: one CTA serves BOTH kv heads (HKV == 2): rows are fetched as whole 128 B lines ------------------------
#define RP3 144   // bytes per K/V smem row: 128 + 16 pad

template <int S>
__global__ void __launch_bounds__(128)
fd3_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int HQ, int HKV,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 16;
  constexpr int SLOT = 2 * ST2 * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + 1);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int nstages = (n + ST2 - 1) / ST2;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + 31) / 32) * 32;
  uint8_t* wbase = smem + rmax_rt * 4 + warp * (S * SLOT);
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  pdl_wait();
  uint32_t q0[2][2], q2[2][2];
#pragma unroll
  for (int kh = 0; kh < 2; kh++) {
    const uint8_t* qp = Q + ((long)b * HQ + kh * 8 + g) * HD;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      q0[kh][ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
      q2[kh][ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
    }
  }
  __syncthreads();

  float o[2][8][2];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int kh = 0; kh < 2; kh++)
#pragma unroll
    for (int d = 0; d < 8; d++) o[kh][d][0] = o[kh][d][1] = 0.f;

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + 4 * s) * ST2;
      const int c = lane & 7;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int tk = (lane >> 3) + 4 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = rr * 128 + c * 16;
        cp16(slot + tk * RP3 + c * 16, KC + goff);
        cp16(slot + ST2 * RP3 + tk * RP3 + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    const uint8_t* slot = wbase + (s % S) * SLOT;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP3;
    const int tb = begin + (warp + 4 * s) * ST2;
    const bool full_vis = tb + ST2 <= end;
#pragma unroll
    for (int kh = 0; kh < 2; kh++) {
      float sacc[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
        uint32_t k0, k1, k2, k3;
        ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
        mma_qk8(sacc[nt], q0[kh][0], q2[kh][0], k0, k1);
        mma_qk8(sacc[nt], q0[kh][1], q2[kh][1], k2, k3);
      }
      float sv[4][2];
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          float v = sacc[nt][r];
          if (!full_vis && tb + nt * 8 + 2 * t + r >= end) v = -INFINITY;
          sv[nt][r] = v;
          mx = fmaxf(mx, v);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx *= sc2;
      const float newm = fmaxf(m_run[kh], mx);
      const float safe = newm == -INFINITY ? 0.f : newm;
      const float corr = exp2f(m_run[kh] - safe);
      float rs = 0.f;
      uint32_t pa[4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
        rs += p0 + p1;
        pa[nt] = pack_h2(p0, p1);
      }
      rs += __shfl_xor_sync(0xffffffffu, rs, 1);
      rs += __shfl_xor_sync(0xffffffffu, rs, 2);
      l_run[kh] = l_run[kh] * corr + rs;
      m_run[kh] = newm;
      if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) { o[kh][d][0] *= corr; o[kh][d][1] *= corr; }
      }
#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
#pragma unroll
        for (int pr = 0; pr < 2; pr++) {
          uint32_t r0, r1, r2, r3;
          ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
          const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
          const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
          const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
          const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
          float acc[4];
          acc[0] = o[kh][4 * pr + 0][0]; acc[1] = o[kh][4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
          o[kh][4 * pr + 0][0] = acc[0]; o[kh][4 * pr + 0][1] = acc[1];
          acc[0] = o[kh][4 * pr + 1][0]; acc[1] = o[kh][4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
          o[kh][4 * pr + 1][0] = acc[0]; o[kh][4 * pr + 1][1] = acc[1];
          acc[0] = o[kh][4 * pr + 2][0]; acc[1] = o[kh][4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
          o[kh][4 * pr + 2][0] = acc[0]; o[kh][4 * pr + 2][1] = acc[1];
          acc[0] = o[kh][4 * pr + 3][0]; acc[1] = o[kh][4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
          o[kh][4 * pr + 3][0] = acc[0]; o[kh][4 * pr + 3][1] = acc[1];
        }
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();
  pdl_trigger();

  // merge the four warps for each kv head
  float* sm_m = reinterpret_cast<float*>(smem);   // [2][4][8]
  float* sm_l = sm_m + 64;                        // [2][4][8]
  float* sm_o = sm_l + 64;                        // [2][4][8][64]
#pragma unroll
  for (int kh = 0; kh < 2; kh++) {
    if (t == 0) {
      sm_m[(kh * 4 + warp) * 8 + g] = m_run[kh];
      sm_l[(kh * 4 + warp) * 8 + g] = l_run[kh];
    }
#pragma unroll
    for (int mm = 0; mm < 4; mm++)
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        sm_o[((kh * 4 + warp) * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[kh][2 * mm][cc];
        sm_o[((kh * 4 + warp) * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[kh][2 * mm + 1][cc];
      }
  }
  __syncthreads();
  for (int idx = tid; idx < 2 * 8 * HD; idx += 128) {
    const int kh = idx / (8 * HD), rem = idx % (8 * HD);
    const int row = rem / HD, d = rem % HD;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mmx = fmaxf(mmx, sm_m[(kh * 4 + w) * 8 + row]);
    float ls = 0.f, acc = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float mw = sm_m[(kh * 4 + w) * 8 + row];
        const float f = mw == -INFINITY ? 0.f : exp2f(mw - mmx);
        ls += sm_l[(kh * 4 + w) * 8 + row] * f;
        acc += sm_o[((kh * 4 + w) * 8 + row) * 64 + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mmx + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}

template <int S>
static void launch3(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                    const float* VS, float* PART, float* LSE, int B, int HQ, int HKV, int table_n, int page, int window,
                    int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP3;
  size_t dyn = (size_t)(((chunk + 31) / 32) * 32) * 4 + (size_t)4 * S * SLOT;
  const size_t merge = (size_t)(128 + 2 * 4 * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd3_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  launch_pdl(fd3_partials<S>, grid, dim3(128), dyn, st, reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                         reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, HQ, HKV,
                                         table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---- fd3v: verify blocks (8 tokens x 8 heads = 64 query rows per KV head), both KV heads per CTA, 128 B row fetches ----
// warp = (kv head, 32-row half); the four warps share each 64-token K/V tile; no cross-warp merge is needed.
#define T3V 64
static int g_fd2_dbg = 0;
extern "C" void fd2_set_dbg(int v) { g_fd2_dbg = v; }
__device__ __forceinline__ void mma_q16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  uint32_t gl, gh, hl, hh, bl, bh;
  fp8x4_to_h2(a0, gl, gh); fp8x4_to_h2(a1, hl, hh); fp8x4_to_h2(b0, bl, bh);
  mma_f16(c, gl, hl, gh, hh, bl, bh);
  fp8x4_to_h2(a2, gl, gh); fp8x4_to_h2(a3, hl, hh); fp8x4_to_h2(b1, bl, bh);
  mma_f16(c, gl, hl, gh, hh, bl, bh);
#endif
}

template <int S>
__global__ void __launch_bounds__(128)
fd3v_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
              const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
              const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
              int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 64;
  constexpr int SLOTB = 2 * T3V * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kh = warp >> 1, mhalf = warp & 1;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int ntiles = (n + T3V - 1) / T3V;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + T3V - 1) / T3V) * T3V;
  uint8_t* ring = smem + rmax_rt * 4;
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  // Q fragments (e4m3 bytes): m-tile mg = 2 mhalf + mt, rows g and g + 8
  // f16 A fragments of the four k16 steps (QK runs on f16 mma: the e4m3 mma.sync is emulated on sm_90)
  uint32_t qa[2][4][4];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    uint32_t xr[2][2][2];  // [half][ks][lo/hi segment]
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int row = (mhalf * 2 + mt) * 16 + g + 8 * half;
      const int token = row >> 3, qh = kh * 8 + (row & 7);
      const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        xr[half][ks][0] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
        xr[half][ks][1] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
      }
    }
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int seg2 = 0; seg2 < 2; seg2++) {
        uint32_t gl, gh, hl, hh;
        fp8x4_to_h2(xr[0][ks][seg2], gl, gh);
        fp8x4_to_h2(xr[1][ks][seg2], hl, hh);
        qa[mt][2 * ks + seg2][0] = gl; qa[mt][2 * ks + seg2][1] = hl;
        qa[mt][2 * ks + seg2][2] = gh; qa[mt][2 * ks + seg2][3] = hh;
      }
  }
  __syncthreads();  // rows[] ready

  float o[2][8][4];
  float m_run[2][2], l_run[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    m_run[mt][0] = m_run[mt][1] = -INFINITY;
    l_run[mt][0] = l_run[mt][1] = 0.f;
#pragma unroll
    for (int d = 0; d < 8; d++) o[mt][d][0] = o[mt][d][1] = o[mt][d][2] = o[mt][d][3] = 0.f;
  }

  auto issue = [&](int s) {
    if (s < ntiles) {
      uint8_t* slot = ring + (s % S) * SLOTB;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int ci = tid + 128 * i;
        const int which = ci >> 9, rem = ci & 511;
        const int row = rem >> 3, c = rem & 7;
        const int li = s * T3V + row;
        const long rr = li < n ? rows[li] : 0;
        if (!(dbg & 8)) cp16(slot + which * (T3V * RP3) + row * RP3 + c * 16, (which ? VC : KC) + rr * 128 + c * 16);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < ntiles; s++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(s + S - 1);
    const uint8_t* slot = ring + (s % S) * SLOTB;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + T3V * RP3;
    const int tb = begin + s * T3V;
    const bool full_vis = (tb + T3V <= end) && (tb + T3V - 1 <= prefix) && (!window || tb > prefix + L - 1 - window);
    uint32_t pa[2][8][2];
    float sall[2][8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      uint32_t kk4[4], bl[4], bh[4];
      if (!(dbg & 1)) {
      ldsm4(kk4[0], kk4[1], kk4[2], kk4[3], kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
#pragma unroll
      for (int j = 0; j < 4; j++) fp8x4_to_h2(kk4[j], bl[j], bh[j]);
      }
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int r = 0; r < 4; r++) sall[mt][nt][r] = (dbg & 1) ? (float)(nt + r) : 0.f;
        if (!(dbg & 1))
#pragma unroll
        for (int j = 0; j < 4; j++) mma_f16(sall[mt][nt], qa[mt][j][0], qa[mt][j][1], qa[mt][j][2], qa[mt][j][3], bl[j], bh[j]);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
      float (&sacc)[8][4] = sall[mt];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = (mhalf * 2 + mt) * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sacc[nt][r] = -INFINITY;
          }
      }
      float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sacc[nt][r]);
      float corr[2], safe[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;
        newm[h] = fmaxf(m_run[mt][h], mx[h]);
        safe[h] = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[mt][h] - safe[h]);
      }
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          p[r] = (dbg & 2) ? fmaf(sacc[nt][r], sc2, -safe[r >> 1]) : exp2f(fmaf(sacc[nt][r], sc2, -safe[r >> 1]));
          rs[r >> 1] += p[r];
        }
        pa[mt][nt][0] = pack_h2(p[0], p[1]);  // row g
        pa[mt][nt][1] = pack_h2(p[2], p[3]);  // row g + 8
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[mt][h] = l_run[mt][h] * corr[h] + rs[h];
        m_run[mt][h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[mt][d][0] *= corr[0]; o[mt][d][1] *= corr[0];
          o[mt][d][2] *= corr[1]; o[mt][d][3] *= corr[1];
        }
      }
    }
    // ---- O += P V: four 16-token k-steps; the V fragments are shared by both m-tiles ----
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      if (dbg & 4) break;
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
          const uint32_t a0 = pa[mt][2 * kk][0], a1 = pa[mt][2 * kk][1], a2 = pa[mt][2 * kk + 1][0], a3 = pa[mt][2 * kk + 1][1];
          mma_f16(o[mt][4 * pr + 0], a0, a1, a2, a3, bE0, bE1);
          mma_f16(o[mt][4 * pr + 1], a0, a1, a2, a3, bO0, bO1);
          mma_f16(o[mt][4 * pr + 2], a0, a1, a2, a3, cE0, cE1);
          mma_f16(o[mt][4 * pr + 3], a0, a1, a2, a3, cO0, cO1);
        }
      }
    }
  }
  cp_wait<0>();
  // rows owned by this warp: no cross-warp merge.  Column mapping of the E/O d-tile pairs: d = 16 mm + 2 (2 t + c) + {0: E, 1: O}.
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = (mhalf * 2 + mt) * 16 + g + 8 * h;
      const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
      const float ls = l_run[mt][h];
      const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
      for (int mm = 0; mm < 4; mm++)
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
          PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc)] = o[mt][2 * mm][2 * h + cc] * inv * value_scale;
          PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc) + 1] = o[mt][2 * mm + 1][2 * h + cc] * inv * value_scale;
        }
      if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[mt][h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
    }
}


// ---- fd3w: like fd3v but 8 warps per CTA, one 16-row m-tile per warp (fewer registers -> more warps in flight) ----
template <int S>
__global__ void __launch_bounds__(256, 2)
fd3w_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
              const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
              const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
              int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 64;
  constexpr int SLOTB = 2 * T3V * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kh = warp >> 2, mg = warp & 3;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int ntiles = (n + T3V - 1) / T3V;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + T3V - 1) / T3V) * T3V;
  uint8_t* ring = smem + rmax_rt * 4;
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 256) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  uint32_t qa[4][4];
  {
    uint32_t xr[2][2][2];
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int row = mg * 16 + g + 8 * half;
      const int token = row >> 3, qh = kh * 8 + (row & 7);
      const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        xr[half][ks][0] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
        xr[half][ks][1] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
      }
    }
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int seg2 = 0; seg2 < 2; seg2++) {
        uint32_t gl, gh, hl, hh;
        fp8x4_to_h2(xr[0][ks][seg2], gl, gh);
        fp8x4_to_h2(xr[1][ks][seg2], hl, hh);
        qa[2 * ks + seg2][0] = gl; qa[2 * ks + seg2][1] = hl;
        qa[2 * ks + seg2][2] = gh; qa[2 * ks + seg2][3] = hh;
      }
  }
  __syncthreads();

  float o[8][4];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int d = 0; d < 8; d++) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;

  auto issue = [&](int s) {
    if (s < ntiles) {
      uint8_t* slot = ring + (s % S) * SLOTB;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int ci = tid + 256 * i;
        const int which = ci >> 9, rem = ci & 511;
        const int row = rem >> 3, c = rem & 7;
        const int li = s * T3V + row;
        const long rr = li < n ? rows[li] : 0;
        if (!(dbg & 8)) cp16(slot + which * (T3V * RP3) + row * RP3 + c * 16, (which ? VC : KC) + rr * 128 + c * 16);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < ntiles; s++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(s + S - 1);
    const uint8_t* slot = ring + (s % S) * SLOTB;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + T3V * RP3;
    const int tb = begin + s * T3V;
    const bool full_vis = (tb + T3V <= end) && (tb + T3V - 1 <= prefix) && (!window || tb > prefix + L - 1 - window);
    float sacc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
#pragma unroll
      for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      if (!(dbg & 1)) {
        uint32_t kk4[4], bl[4], bh[4];
        ldsm4(kk4[0], kk4[1], kk4[2], kk4[3], kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
#pragma unroll
        for (int j = 0; j < 4; j++) fp8x4_to_h2(kk4[j], bl[j], bh[j]);
#pragma unroll
        for (int j = 0; j < 4; j++) mma_f16(sacc[nt], qa[j][0], qa[j][1], qa[j][2], qa[j][3], bl[j], bh[j]);
      } else {
        sacc[nt][0] = (float)nt;
      }
    }
    if (!full_vis) {
#pragma unroll
      for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int row = mg * 16 + g + 8 * (r >> 1);
          const int token = row >> 3;
          const int col = tb + nt * 8 + 2 * t + (r & 1);
          const int pos = prefix + token;
          const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
          if (!vis) sacc[nt][r] = -INFINITY;
        }
    }
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
      for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sacc[nt][r]);
    float corr[2], safe[2], newm[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      mx[h] *= sc2;
      newm[h] = fmaxf(m_run[h], mx[h]);
      safe[h] = newm[h] == -INFINITY ? 0.f : newm[h];
      corr[h] = exp2f(m_run[h] - safe[h]);
    }
    uint32_t pa[8][2];
    float rs[2] = {0.f, 0.f};
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      float p[4];
#pragma unroll
      for (int r = 0; r < 4; r++) {
        p[r] = (dbg & 2) ? fmaf(sacc[nt][r], sc2, -safe[r >> 1]) : exp2f(fmaf(sacc[nt][r], sc2, -safe[r >> 1]));
        rs[r >> 1] += p[r];
      }
      pa[nt][0] = pack_h2(p[0], p[1]);
      pa[nt][1] = pack_h2(p[2], p[3]);
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
      rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
      l_run[h] = l_run[h] * corr[h] + rs[h];
      m_run[h] = newm[h];
    }
    if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
      for (int d = 0; d < 8; d++) {
        o[d][0] *= corr[0]; o[d][1] *= corr[0];
        o[d][2] *= corr[1]; o[d][3] *= corr[1];
      }
    }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      if (dbg & 4) break;
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
        const uint32_t a0 = pa[2 * kk][0], a1 = pa[2 * kk][1], a2 = pa[2 * kk + 1][0], a3 = pa[2 * kk + 1][1];
        mma_f16(o[4 * pr + 0], a0, a1, a2, a3, bE0, bE1);
        mma_f16(o[4 * pr + 1], a0, a1, a2, a3, bO0, bO1);
        mma_f16(o[4 * pr + 2], a0, a1, a2, a3, cE0, cE1);
        mma_f16(o[4 * pr + 3], a0, a1, a2, a3, cO0, cO1);
      }
    }
  }
  cp_wait<0>();
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = mg * 16 + g + 8 * h;
    const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
    const float ls = l_run[h];
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
    for (int mm = 0; mm < 4; mm++)
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc)] = o[2 * mm][2 * h + cc] * inv * value_scale;
        PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc) + 1] = o[2 * mm + 1][2 * h + cc] * inv * value_scale;
      }
    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}

template <int S>
static void launch3w(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                     const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                     int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  const size_t dyn = (size_t)(((chunk + T3V - 1) / T3V) * T3V) * 4 + (size_t)S * 2 * T3V * RP3;
  cudaFuncSetAttribute(fd3w_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  fd3w_partials<S><<<grid, 256, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                          reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                          table_n, page, window, splits, chunk, sm_scale, scaled, g_fd2_dbg);
}

template <int S>
static void launch3v(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                     const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                     int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  const size_t dyn = (size_t)(((chunk + T3V - 1) / T3V) * T3V) * 4 + (size_t)S * 2 * T3V * RP3;
  cudaFuncSetAttribute(fd3v_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  fd3v_partials<S><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                          reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                          table_n, page, window, splits, chunk, sm_scale, scaled, g_fd2_dbg);
}

template <int S>
static void launch2(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                    const float* VS, float* PART, float* LSE, int B, int HQ, int HKV, int table_n, int page, int window,
                    int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP;
  size_t dyn = (size_t)(((chunk + 31) / 32) * 32) * 4 + (size_t)4 * S * SLOT;
  const size_t merge = (size_t)(64 + 4 * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd2_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  launch_pdl(fd2_partials<S>, grid, dim3(128), dyn, st, reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                         reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, HQ, HKV,
                                         table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---------------------------------------------------------------------------------------------------------------
// slide1: the whole sliding-window decode attention in ONE kernel (one query token, 8 heads per KV head):
// q/k RMS norm + RoPE, fp8 conversion, cache write of the new k/v, streamed attention over the window,
// cross-warp merge, per-head RMS norm and sigmoid gate.  grid (HKV, B), 256 threads.
// ---------------------------------------------------------------------------------------------------------------
#include <cuda_fp8.h>
#define SW 8                 // warps per CTA
#define SRMAX 2112           // max streamed tokens (window 2048 + slack)

__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

template <int S>
__global__ void __launch_bounds__(SW * 32)
slide1_kernel(const bf16* __restrict__ QKV, const long* __restrict__ POS, const long* __restrict__ LOC,
              uint8_t* __restrict__ KC, uint8_t* __restrict__ VC, const int* __restrict__ TABLE,
              const int* __restrict__ SEQLEN, const float* __restrict__ KS, const float* __restrict__ VS,
              bf16* __restrict__ OUT, int HQ, int HKV, int table_n, int page, int window, float theta, float sm_scale,
              int scaled, int qkv_stride) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int SLOT = 2 * ST2 * RP;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kh = blockIdx.x, b = blockIdx.y;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int n = prefix - first;            // streamed tokens (the new token is handled separately)
  const int nst = (n + ST2 - 1) / ST2;
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const bf16* qkv = QKV + (long)b * qkv_stride;
  const float pos = (float)POS[b];

  // shared layout: rows[SRMAX] | qs[8][64] | newslot (K row 0 / V row 0, zero filled) | per-warp rings | merge area
  int* rows = reinterpret_cast<int*>(smem);
  uint8_t* qs = smem + SRMAX * 4;
  uint8_t* newslot = qs + 512;                       // SLOT bytes
  uint8_t* rings = newslot + SLOT;                   // SW * S * SLOT bytes
  uint8_t* wbase = rings + warp * (S * SLOT);

  // ---- zero the new-token slot, fill the row table ----
  for (int i = tid; i < SLOT / 16; i += SW * 32) reinterpret_cast<uint4*>(newslot)[i] = make_uint4(0, 0, 0, 0);
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += SW * 32) {
    const int tok = first + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  // ---- q: 8 query heads of this KV head (threads 0..127), k_new (warp 4), v_new (warp 5) ----
  if (tid < 128) {
    const int j = tid >> 4, part = tid & 15;
    const bf16* src = qkv + (kh * 8 + j) * HD + 4 * part;
    float v[4];
    {
      const uint2 raw = *reinterpret_cast<const uint2*>(src);
      const bf162* p = reinterpret_cast<const bf162*>(&raw);
      const float2 a = __bfloat1622float2(p[0]), c = __bfloat1622float2(p[1]);
      v[0] = a.x; v[1] = a.y; v[2] = c.x; v[3] = c.y;
    }
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
    for (int d = 1; d <= 8; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float inv = rsqrtf(ss / 64.f + 1e-5f);
    float o4[4];
#pragma unroll
    for (int k = 0; k < 4; k++) o4[k] = v[k] * inv;
    if (theta != 0.f) {
#pragma unroll
      for (int pp = 0; pp < 2; pp++) {
        const int i = 2 * part + pp;
        const float freq = expf((-2.f * (float)i / 64.f) * logf(theta));
        float sn, cs;
        sincosf(pos * freq, &sn, &cs);
        const float x0 = o4[2 * pp], x1 = o4[2 * pp + 1];
        o4[2 * pp] = x0 * cs - x1 * sn;
        o4[2 * pp + 1] = x0 * sn + x1 * cs;
      }
    }
    uint32_t w4 = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) w4 |= (uint32_t)f2e4m3(o4[k]) << (8 * k);
    *reinterpret_cast<uint32_t*>(qs + j * 64 + 4 * part) = w4;
  } else if (warp == 4 || warp == 5) {
    const int is_v = warp == 5;
    const bf16* src = qkv + (HQ + (is_v ? HKV : 0) + kh) * HD + 2 * lane;
    const bf162 raw = *reinterpret_cast<const bf162*>(src);
    float2 v = __bfloat1622float2(raw);
    if (!is_v) {
      float ss = v.x * v.x + v.y * v.y;
#pragma unroll
      for (int d = 1; d <= 16; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
      const float inv = rsqrtf(ss / 64.f + 1e-5f);
      v.x *= inv; v.y *= inv;
      if (theta != 0.f) {
        const float freq = expf((-2.f * (float)lane / 64.f) * logf(theta));
        float sn, cs;
        sincosf(pos * freq, &sn, &cs);
        const float x0 = v.x, x1 = v.y;
        v.x = x0 * cs - x1 * sn;
        v.y = x0 * sn + x1 * cs;
      }
      v.x /= key_scale; v.y /= key_scale;
    } else {
      v.x /= value_scale; v.y /= value_scale;
    }
    const uint8_t b0 = f2e4m3(v.x), b1 = f2e4m3(v.y);
    const long loc = LOC[b];
    uint8_t* dst = (is_v ? VC : KC) + (loc * HKV + kh) * HD + 2 * lane;
    *reinterpret_cast<uint16_t*>(dst) = (uint16_t)b0 | ((uint16_t)b1 << 8);
    uint8_t* ns = newslot + (is_v ? ST2 * RP : 0) + 2 * lane;     // row 0 of the new-token slot
    *reinterpret_cast<uint16_t*>(ns) = (uint16_t)b0 | ((uint16_t)b1 << 8);
  }
  __syncthreads();

  uint32_t q0[2], q2[2];
#pragma unroll
  for (int ks = 0; ks < 2; ks++) {
    q0[ks] = *reinterpret_cast<const uint32_t*>(qs + g * 64 + 32 * ks + 4 * t);
    q2[ks] = *reinterpret_cast<const uint32_t*>(qs + g * 64 + 32 * ks + 16 + 4 * t);
  }
  float o[8][2];
  float m_run = -INFINITY, l_run = 0.f;
#pragma unroll
  for (int d = 0; d < 8; d++) o[d][0] = o[d][1] = 0.f;

  const int my_stages = (nst - warp + SW - 1) / SW > 0 ? (nst - warp + SW - 1) / SW : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + SW * s) * ST2;
      const int c = lane & 3;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int tk = (lane >> 2) + 8 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = (rr * HKV + kh) * HD + c * 16;
        cp16(slot + tk * RP + c * 16, KC + goff);
        cp16(slot + ST2 * RP + tk * RP + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);
  const int srow = lane & 7, smat = lane >> 3;

  auto process = [&](const uint8_t* slot, int tb, int end_tok) {
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP;
    float sacc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#pragma unroll
      for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      uint32_t k0, k1, k2, k3;
      ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP + smat * 16);
      mma_qk8(sacc[nt], q0[0], q2[0], k0, k1);
      mma_qk8(sacc[nt], q0[1], q2[1], k2, k3);
    }
    const bool full_vis = tb + ST2 <= end_tok;
    float sv[4][2];
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int r = 0; r < 2; r++) {
        float v = sacc[nt][r];
        if (!full_vis && tb + nt * 8 + 2 * t + r >= end_tok) v = -INFINITY;
        sv[nt][r] = v;
        mx = fmaxf(mx, v);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx *= sc2;
    const float newm = fmaxf(m_run, mx);
    const float safe = newm == -INFINITY ? 0.f : newm;
    const float corr = exp2f(m_run - safe);
    float rs = 0.f;
    uint32_t pa[4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
      rs += p0 + p1;
      pa[nt] = pack_h2(p0, p1);
    }
    rs += __shfl_xor_sync(0xffffffffu, rs, 1);
    rs += __shfl_xor_sync(0xffffffffu, rs, 2);
    l_run = l_run * corr + rs;
    m_run = newm;
    if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
      for (int d = 0; d < 8; d++) { o[d][0] *= corr; o[d][1] *= corr; }
    }
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP + pr * 32 + (smat >> 1) * 16);
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
        float acc[4];
        acc[0] = o[4 * pr + 0][0]; acc[1] = o[4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
        o[4 * pr + 0][0] = acc[0]; o[4 * pr + 0][1] = acc[1];
        acc[0] = o[4 * pr + 1][0]; acc[1] = o[4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
        o[4 * pr + 1][0] = acc[0]; o[4 * pr + 1][1] = acc[1];
        acc[0] = o[4 * pr + 2][0]; acc[1] = o[4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
        o[4 * pr + 2][0] = acc[0]; o[4 * pr + 2][1] = acc[1];
        acc[0] = o[4 * pr + 3][0]; acc[1] = o[4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
        o[4 * pr + 3][0] = acc[0]; o[4 * pr + 3][1] = acc[1];
      }
    }
  };

#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    process(wbase + (s % S) * SLOT, first + (warp + SW * s) * ST2, prefix);
    __syncwarp();
  }
  cp_wait<0>();
  if (warp == 0) process(newslot, prefix, prefix + 1);   // the new token (position prefix), fp8-rounded as stored
  __syncthreads();

  // ---- merge the warps ----
  float* sm_m = reinterpret_cast<float*>(smem);   // reuses the row table: [SW][8], [SW][8], [SW][8][64]
  float* sm_l = sm_m + SW * 8;
  float* sm_o = sm_l + SW * 8;
  if (t == 0) { sm_m[warp * 8 + g] = m_run; sm_l[warp * 8 + g] = l_run; }
#pragma unroll
  for (int mm = 0; mm < 4; mm++)
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[2 * mm][cc];
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[2 * mm + 1][cc];
    }
  __syncthreads();
  // warp h finalizes query head h of this KV head; lane owns dims 2 lane, 2 lane + 1
  {
    const int h = warp;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < SW; w++) mmx = fmaxf(mmx, sm_m[w * 8 + h]);
    float ls = 0.f, a0 = 0.f, a1 = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < SW; w++) {
        const float f = sm_m[w * 8 + h] == -INFINITY ? 0.f : exp2f(sm_m[w * 8 + h] - mmx);
        ls += sm_l[w * 8 + h] * f;
        a0 += sm_o[(w * 8 + h) * 64 + 2 * lane] * f;
        a1 += sm_o[(w * 8 + h) * 64 + 2 * lane + 1] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    // bf16 round, RMS norm over the 64 dims, sigmoid gate
    const float u0 = __bfloat162float(__float2bfloat16(a0 * inv * value_scale));
    const float u1 = __bfloat162float(__float2bfloat16(a1 * inv * value_scale));
    float ss = u0 * u0 + u1 * u1;
#pragma unroll
    for (int d = 1; d <= 16; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float rinv = rsqrtf(ss / 64.f + 1e-5f);
    const int qh = kh * 8 + h;
    const bf162 gate = *reinterpret_cast<const bf162*>(qkv + (HQ + 2 * HKV) * HD + qh * HD + 2 * lane);
    const float2 gf = __bfloat1622float2(gate);
    const float s0 = 1.f / (1.f + expf(-gf.x)), s1 = 1.f / (1.f + expf(-gf.y));
    *reinterpret_cast<bf162*>(OUT + ((long)b * HQ + qh) * HD + 2 * lane) = __floats2bfloat162_rn(u0 * rinv * s0, u1 * rinv * s1);
  }
}

template <int S>
static void launch_slide(const void* QKV, const void* POS, const void* LOC, void* KC, void* VC, const int* TABLE,
                         const int* SEQLEN, const float* KS, const float* VS, void* OUT, int B, int HQ, int HKV,
                         int table_n, int page, int window, float theta, float sm_scale, int scaled, int qkv_stride,
                         cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP;
  size_t dyn = (size_t)SRMAX * 4 + 512 + SLOT + (size_t)SW * S * SLOT;
  const size_t merge = (size_t)(2 * SW * 8 + SW * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(slide1_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(HKV, B);
  slide1_kernel<S><<<grid, SW * 32, dyn, st>>>(reinterpret_cast<const bf16*>(QKV), reinterpret_cast<const long*>(POS),
                                              reinterpret_cast<const long*>(LOC), reinterpret_cast<uint8_t*>(KC),
                                              reinterpret_cast<uint8_t*>(VC), TABLE, SEQLEN, KS, VS,
                                              reinterpret_cast<bf16*>(OUT), HQ, HKV, table_n, page, window, theta,
                                              sm_scale, scaled, qkv_stride);
}

extern "C" int slide_run(int stages, const void* QKV, const void* POS, const void* LOC, void* KC, void* VC,
                         const int* TABLE, const int* SEQLEN, const float* KS, const float* VS, void* OUT, int B, int HQ,
                         int HKV, int table_n, int page, int window, float theta, float sm_scale, int scaled,
                         int qkv_stride, cudaStream_t st) {
#define SL(S_) if (stages == S_) { launch_slide<S_>(QKV, POS, LOC, KC, VC, TABLE, SEQLEN, KS, VS, OUT, B, HQ, HKV, table_n, page, window, theta, sm_scale, scaled, qkv_stride, st); return 0; }
  SL(2) SL(3) SL(4)
#undef SL
  return 1;
}

__global__ void pf_kernel(const char* p, size_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const size_t chunk = 32768;
  for (size_t off = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * chunk; off < bytes; off += (size_t)gridDim.x * blockDim.x * chunk) {
    const size_t n = bytes - off < chunk ? bytes - off : chunk;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p + off), "r"((uint32_t)n) : "memory");
  }
#endif
}
extern "C" void pf_run(const void* p, size_t bytes, cudaStream_t st) { pf_kernel<<<4, 128, 0, st>>>(reinterpret_cast<const char*>(p), bytes); }

template <int MT, int S, bool QK8>
static void launch(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                   const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                   int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int BM = MT * 16;
  constexpr int PER_WARP = S * 2 * 16 * RP + 2 * 16 * KP;
  size_t dyn = 4 * PER_WARP;
  const size_t merge = (size_t)(8 * BM + 4 * BM * HD) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd_partials<MT, S, QK8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  fd_partials<MT, S, QK8><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                             reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ,
                                             HKV, table_n, page, window, splits, chunk, sm_scale, scaled);
}

extern "C" int fd_run(int mt, int stages, int qk8, const void* Q, const void* KC, const void* VC, const int* TABLE,
                      const int* SEQLEN, const float* KS, const float* VS, float* PART, float* LSE, int B, int L, int HQ,
                      int HKV, int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled,
                      cudaStream_t st) {
#define FD(MT_, S_) if (mt == MT_ && stages == S_ && qk8 == 0) { launch<MT_, S_, false>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
#define FD8(MT_, S_) if (mt == MT_ && stages == S_ && qk8 == 1) { launch<MT_, S_, true>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD(1, 3) FD(1, 4) FD(1, 6) FD(1, 8) FD(4, 3) FD(4, 4) FD(4, 6)
  FD8(1, 3) FD8(1, 4) FD8(1, 6) FD8(4, 3) FD8(4, 4) FD8(4, 6)
#undef FD8
#define FD2(S_) if (mt == 1 && stages == 200 + S_) { launch2<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD2(2) FD2(3) FD2(4)
#undef FD2
#define FD3W(S_) if (mt == 4 && HKV == 2 && L == 8 && stages == 500 + S_) { launch3w<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3W(2) FD3W(3) FD3W(4)
#undef FD3W
#define FD3V(S_) if (mt == 4 && HKV == 2 && L == 8 && stages == 400 + S_) { launch3v<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3V(2) FD3V(3) FD3V(4)
#undef FD3V
#define FD3(S_) if (mt == 1 && HKV == 2 && stages == 300 + S_) { launch3<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3(2) FD3(3) FD3(4)
#undef FD3
#define FDV(S_) if (mt == 4 && stages == 100 + S_) { launch_v<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FDV(3) FDV(4) FDV(6) FDV(8)
#undef FDV
#undef FD
  return 1;
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int slide_run(int, const void*, const void*, const void*, void*, void*, const int*, const int*, const float*,
                         const float*, void*, int, int, int, int, int, int, float, float, int, int, cudaStream_t);
void slide_run_t(int64_t stages, torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
                 torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t HQ,
                 int64_t HKV, int64_t page, int64_t window, double theta, double sm_scale, int64_t scaled) {
  int r = slide_run((int)stages, qkv.data_ptr(), pos.data_ptr(), loc.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(),
                    seqlens.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(), out.data_ptr(), (int)seqlens.numel(),
                    (int)HQ, (int)HKV, (int)table.size(1), (int)page, (int)window, (float)theta, (float)sm_scale, (int)scaled,
                    (int)qkv.stride(0), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported slide config");
}
extern "C" void fd2_set_dbg(int);
void fd2_set_dbg_t(int64_t v) { fd2_set_dbg((int)v); }
extern "C" void fd_set_pdl(int);
void fd_set_pdl_t(int64_t v) { fd_set_pdl((int)v); }
extern "C" void pf_run(const void*, size_t, cudaStream_t);
void pf_t(torch::Tensor w) { pf_run(w.data_ptr(), (size_t)w.numel() * w.element_size(), at::cuda::getCurrentCUDAStream()); }
extern "C" int fd_run(int, int, int, const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*,
                      float*, int, int, int, int, int, int, int, int, int, float, int, cudaStream_t);

void fd_run_t(int64_t mt, int64_t stages, int64_t qk8, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table,
              torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L,
              int64_t HQ, int64_t HKV, int64_t page, int64_t window, int64_t splits, int64_t chunk, double sm_scale,
              int64_t scaled) {
  int r = fd_run((int)mt, (int)stages, (int)qk8, q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(),
                 seqlens.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(), part.data_ptr<float>(),
                 lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ, (int)HKV, (int)table.size(1), (int)page,
                 (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported fd config");
}
'''
_FD_CPP = r'''
void pf_t(torch::Tensor w);
void fd2_set_dbg_t(int64_t v);
void fd_set_pdl_t(int64_t v);
void slide_run_t(int64_t stages, torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
                 torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t HQ,
                 int64_t HKV, int64_t page, int64_t window, double theta, double sm_scale, int64_t scaled);
void fd_run_t(int64_t mt, int64_t stages, int64_t qk8, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table,
              torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L,
              int64_t HQ, int64_t HKV, int64_t page, int64_t window, int64_t splits, int64_t chunk, double sm_scale,
              int64_t scaled);
'''

_FD_MOD = None
_FD_FLAGS = ["-DFD_HW_CVT", "-DFD_NATIVE"]
_FD_LAB = False
_FD_PDL = False
_PDL_MAXB = 1
# policy: which shapes use the CUDA split-KV kernel, with (stages, splits_fn)
_FD_POLICY = {('full', False): [(2, 24, 2, 64, 3), (25, 64, 2, 32, 3), (1, 1, 2, 128, 2), (2, 24, 2, 64, 2), (25, 64, 3, 32, 2)], ('sliding', False): [(1, 1, 3, 16, 2), (2, 24, 3, 4, 2), (25, 64, 3, 2, 2)]}


def _fd_module():
    global _FD_MOD
    if _FD_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _FD_MOD = load_inline("muse_fd_attn", _FD_CPP, cuda_sources=_FD_CUDA, functions=["fd_run_t", "slide_run_t", "pf_t", "fd2_set_dbg_t", "fd_set_pdl_t"],
                              extra_cuda_cflags=["-O3"] + _FD_FLAGS, verbose=False)
        _FD_MOD.fd_set_pdl_t(int(_FD_PDL and torch.cuda.get_device_capability()[0] >= 9))
    return _FD_MOD


def _fd_forward(data, q, qkv, stages, splits, block=16, qk8=0, variant=1):
    cfg = data['config']
    use_pdl = _FD_PDL and cfg['batch'] <= _PDL_MAXB
    _fd_module().fd_set_pdl_t(int(use_pdl and torch.cuda.get_device_capability()[0] >= 9))
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    scaled = cfg['kv_scale'] != 1.0
    window = cfg['window']
    span = window + length - 1 if window else data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), block) * block
    if variant in (2, 3, 4, 5) and chunk > 4096:
        raise ValueError('chunk too large for fd2')
    bm = max(16, triton.next_power_of_2(length * (hq // hkv)))
    partial = torch.empty((batch, hkv, splits, bm, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, bm), device=qkv.device, dtype=torch.float32)
    _fd_module().fd_run_t(bm // 16, stages + (500 if variant == 5 else 400 if variant == 4 else 100 if bm == 64 else 200 if variant == 2 else 300 if variant == 3 else 0), qk8, q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'],
                          data['k_scale'], data['v_scale'], partial, lse, length, hq, hkv, cfg['page_size'], window,
                          splits, chunk, cfg['sm_scale'] * cfg['qk_scale_factor'], int(scaled))
    result = torch.empty((batch, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, bm, num_warps=4, launch_pdl=use_pdl)
    return result


def _slide_forward(data, stages, qkv=None):
    cfg = data['config']
    batch = cfg['batch']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    qkv = (data['qkv'] if qkv is None else qkv).view(batch, -1)
    out = torch.empty((batch, 1, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _fd_module().slide_run_t(stages, qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'],
                             data['page_table'], data['seq_lens'], data['k_scale'], data['v_scale'], out, hq, hkv,
                             cfg['page_size'], cfg['window'], float(cfg['rope_theta']),
                             cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0))
    return out


def _fd_prepare(data):
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
    return qkv, q


def _fd_fits(cfg, table_n, splits, variant):
    if variant in (3, 4, 5) and cfg['num_kv_heads'] != 2:
        return False
    if variant in (4, 5) and cfg['q_len'] != 8:
        return False
    if variant not in (2, 3, 4, 5):
        return True
    span = cfg['window'] + cfg['q_len'] - 1 if cfg['window'] else table_n * cfg['page_size']
    return -(-(-(-span // splits)) // 16) * 16 <= 4096


def _fd_choice(cfg, context_bound):
    """(stages, splits) when the CUDA kernel should run this shape, else None."""
    key = ('sliding' if cfg['window'] else 'full', cfg['q_len'] > 1)
    for entry in _FD_POLICY.get(key, ()):
        lo_b, hi_b, stages, splits = entry[:4]
        variant = entry[4] if len(entry) > 4 else 1
        if lo_b <= cfg['batch'] <= hi_b and _fd_fits(cfg, context_bound // cfg['page_size'], splits, variant):
            return stages, splits, variant
    return None



_SLIDE_STAGES = 0

@triton.jit
def _b5_matmul(X, W, LO, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
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
def _b5_finish(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    sums = tl.full((B,), 0, tl.float32)
    for s in range(SPLIT):
        sums += tl.load(P + s * SIZE + idx, idx < SIZE, 0)
    tl.store(Y + idx, sums, idx < SIZE)


@triton.jit
def _b5_gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
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




def _b5(x, layer, n, state):
    """BF16 v5 small-token dense paths: FP32 GEMV at one token, transposed split-K for the pooler up to 16 tokens, else cuBLAS."""
    m, k = x.shape
    w = state["dense"][layer]
    if m == 1:
        result = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _b5_gemv[(triton.cdiv(n, 4),)](x, w, result, n, k, 4, 1024, num_warps=4)
        return result
    if layer == "pooler" and m <= 16:
        wt = state["dn"]["b5_poolert"]
        bm, bn, bk = 16, 64, 128
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        split = min(16, triton.next_power_of_2(triton.cdiv(256, tiles)))
        result = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        partial = result if split == 1 else torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        _b5_matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
            x, wt, wt, partial, m, n, k, bm, bn, bk, split, False, wt.shape[1], num_warps=4, num_stages=3)
        if split != 1:
            _b5_finish[(triton.cdiv(m * n, 512),)](partial, result, m * n, split, 512)
        return result
    return _gemm(x, w[:n] if layer == "unpooler" else w)

# ---- weight-streaming dense bf16 GEMM (CUDA) ----
_DENSE_CUDA = r'''
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void l2_prefetch_bulk(const void* p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
#endif
}
static int g_pdl = 0;
static int g_pf = 0;
static int g_dbg = 0;
extern "C" void dense_set_dbg(int v) { g_dbg = v; }
extern "C" void dense_set_pf(int v) { g_pf = v; }
extern "C" void dense_set_pdl(int v) { g_pdl = v; }
template <typename... KArgs, typename... Args>
static void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc));
}
__device__ __forceinline__ void cp16z(void* sdst, const void* gsrc, uint32_t nbytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc), "r"(nbytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// W tiles: uint4 [rt][kc][ks][lane].  grid = (N/16/NTB, KSPLIT).  Each warp covers CPW consecutive k64 chunks.
// part: fp32 [KSPLIT][M][N] (or final bf16 [M][N] when KSPLIT == 1 and FINAL).
template <int NTB, int NCH, int NW, int S, int OM>
__global__ void __launch_bounds__(NW * 32)
dense_kernel(const uint4* __restrict__ Wq, const bf16* __restrict__ X, float* __restrict__ part, int M, int N, int K,
             int ksplit, int* __restrict__ cnt, void* __restrict__ outp, int pair, int nout, int out_fp32, int pf, int dbg) {
  extern __shared__ uint4 smem[];
  pdl_wait();
  pdl_trigger();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kcs = K / 64;                   // chunks along K
  const int cpw = kcs / (ksplit * NW);      // chunks per warp
  const int c0 = (blockIdx.y * NW + warp) * cpw;
  const int tile0 = blockIdx.x * NTB;

  // token chunks are processed in passes of NCH*8 tokens
  for (int tok0 = 0; tok0 < M; tok0 += NCH * 8) {
    const bf16* xp[NCH];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      const int tok = tok0 + h * 8 + g;
      xp[h] = tok < M ? X + (size_t)tok * K + 16 * t + c0 * 64 : nullptr;
    }
    float acc[NTB][NCH][4];
#pragma unroll
    for (int n = 0; n < NTB; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc[n][h][r] = 0.f;

    const uint4* wsrc[NTB];
#pragma unroll
    for (int n = 0; n < NTB; n++) wsrc[n] = Wq + ((size_t)(tile0 + n) * kcs + c0) * 128 + lane;
    constexpr int STG = NTB * 4 + NCH * 2;  // uint4 slots per lane per stage: weights, then activations
    uint4* ring = smem + warp * (S * STG * 32);
    if (pf && lane == 0 && tok0 == 0) {
#pragma unroll
      for (int n = 0; n < NTB; n++) l2_prefetch_bulk(wsrc[n] - lane, (uint32_t)cpw * 2048u);
    }

    auto issue = [&](int c) {
      if (c < cpw) {
#pragma unroll
        for (int n = 0; n < NTB; n++)
#pragma unroll
          for (int ks = 0; ks < 4; ks++) { if (!(dbg & 1)) cp16(&ring[((c % S) * STG + n * 4 + ks) * 32 + lane], wsrc[n] + c * 128 + ks * 32); }
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const bool ok = xp[h] != nullptr;
          const bf16* src = ok ? xp[h] + c * 64 : X;
          cp16z(&ring[((c % S) * STG + NTB * 4 + 2 * h) * 32 + lane], src, ok ? 16u : 0u);
          cp16z(&ring[((c % S) * STG + NTB * 4 + 2 * h + 1) * 32 + lane], src + 8, ok ? 16u : 0u);
        }
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);

#pragma unroll 1
    for (int i = 0; i < cpw; i++) {
      cp_wait<S - 2>();
      issue(i + S - 1);
#pragma unroll
      for (int n = 0; n < NTB; n++) {
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint4 a = ring[((i % S) * STG + n * 4 + ks) * 32 + lane];
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = ring[((i % S) * STG + NTB * 4 + 2 * h + (ks >> 1)) * 32 + lane];
            if (dbg & 2) acc[n][h][0] += __uint_as_float(a.x ^ xs.x); else mma16816(acc[n][h], a.x, a.y, a.z, a.w, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
          }
        }
      }
    }
    cp_wait<0>();
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int n = 0; n < NTB; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) red[(((warp * NTB + n) * NCH + h) * 4 + r) * 32 + lane] = acc[n][h][r];
    __syncthreads();
    for (int item = threadIdx.x; item < NTB * NCH * 4 * 32; item += NW * 32) {
      const int li = item & 31, r = (item >> 5) & 3, h = (item >> 7) % NCH, n = (item >> 7) / NCH;
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) s += red[(((w * NTB + n) * NCH + h) * 4 + r) * 32 + li];
      const int row = (tile0 + n) * 16 + (li >> 2) + 8 * (r >> 1);
      const int tok = tok0 + h * 8 + 2 * (li & 3) + (r & 1);
      if (tok < M) {
        if constexpr (OM == 1) reinterpret_cast<bf16*>(part)[(size_t)tok * N + row] = __float2bfloat16(s);
        else part[((size_t)blockIdx.y * M + tok) * N + row] = s;
      }
    }
    if constexpr (OM == 2) {
      // last-arriving CTA of this output tile reduces all K-split partials (and router hi/lo halves)
      __shared__ int s_last;
      __threadfence();
      __syncthreads();
      const int ntile_out = pair ? (N / 2) / (16 * NTB) : N / (16 * NTB);
      const int slot = (blockIdx.x % ntile_out) * ((M + NCH * 8 - 1) / (NCH * 8)) + tok0 / (NCH * 8);
      if (threadIdx.x == 0) {
        const int old = atomicAdd(&cnt[slot], 1);
        s_last = (old == ksplit * (pair ? 2 : 1) - 1);
      }
      __syncthreads();
      if (s_last) {
        __threadfence();
        const int obase = (blockIdx.x % ntile_out) * NTB * 16;
        const int tn = min(NCH * 8, M - tok0);
        for (int idx = threadIdx.x; idx < NTB * 16 * tn; idx += NW * 32) {
          const int r = idx % (NTB * 16), tk = idx / (NTB * 16);
          float acc = 0.f;
          for (int k = 0; k < ksplit; k++) {
            const float* p = part + ((size_t)k * M + tok0 + tk) * N + obase + r;
            acc += __ldcg(p);
            if (pair) acc += __ldcg(p + nout);
          }
          if (out_fp32) reinterpret_cast<float*>(outp)[(size_t)(tok0 + tk) * nout + obase + r] = acc;
          else reinterpret_cast<bf16*>(outp)[(size_t)(tok0 + tk) * nout + obase + r] = __float2bfloat16(acc);
        }
        if (threadIdx.x == 0) cnt[slot] = 0;
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

// out[m][n] = sum_ks part[ks][m][n] (+ part[..][m][n + N] when pair: router hi/lo), as bf16 or fp32.
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, int M, int NOUT, int NTOT,
                              int ksplit, int pair, int out_fp32) {
  pdl_wait();
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * NOUT) return;
  const int m = idx / NOUT, n = idx % NOUT;
  float s = 0.f;
  for (int k = 0; k < ksplit; k++) {
    s += part[((size_t)k * M + m) * NTOT + n];
    if (pair) s += part[((size_t)k * M + m) * NTOT + n + NOUT];
  }
  if (out_fp32) reinterpret_cast<float*>(out)[idx] = s;
  else reinterpret_cast<bf16*>(out)[idx] = __float2bfloat16(s);
}

template <int NTB, int NCH, int NW, int S, int OM>
static int launch(const void* W, const void* X, float* part, int M, int N, int K, int ksplit, int* cnt, void* outp,
                  int pair, int nout, int out_fp32, cudaStream_t st) {
  dim3 grid(N / 16 / NTB, ksplit);
  const int dyn = NW * S * (NTB * 4 + NCH * 2) * 512;
  cudaFuncSetAttribute(dense_kernel<NTB, NCH, NW, S, OM>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(dense_kernel<NTB, NCH, NW, S, OM>, grid, dim3(NW * 32), (size_t)dyn, st, g_pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const bf16*>(X), part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, g_pf, g_dbg);
  return 0;
}

extern "C" int dense_run(int ntb, int nch, int nw, int s, int om, const void* W, const void* X, float* part, int M, int N,
                         int K, int ksplit, int* cnt, void* outp, int pair, int nout, int out_fp32, cudaStream_t st) {
#define D(NTB_, NCH_, NW_, S_) if (ntb == NTB_ && nch == NCH_ && nw == NW_ && s == S_) { if (om == 1) return launch<NTB_, NCH_, NW_, S_, 1>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); if (om == 2) return launch<NTB_, NCH_, NW_, S_, 2>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); return launch<NTB_, NCH_, NW_, S_, 0>(W, X, part, M, N, K, ksplit, cnt, outp, pair, nout, out_fp32, st); }
  D(1,1,2,4) D(1,1,4,4) D(1,1,8,4) D(1,2,2,4) D(1,2,4,4) D(1,2,8,4) D(1,4,2,4) D(1,4,4,4) D(1,4,8,4) D(2,1,2,4) D(2,1,4,4) D(2,1,8,4) D(2,2,2,4) D(2,2,4,4) D(2,2,8,4) D(2,4,2,4) D(2,4,4,4) D(2,4,8,4) D(1,8,4,4) D(1,8,8,4) D(1,1,16,4) D(1,2,16,4) D(1,4,16,4) D(2,1,16,4)
#undef D
  return 1;
}

__global__ void probe_kernel(const uint4* __restrict__ p, size_t n, float* __restrict__ out) {
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
#pragma unroll 8
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint4 v = __ldcs(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) out[0] = 1.f;
}
extern "C" void probe_run(const void* p, size_t bytes, float* out, int blocks, int threads, cudaStream_t st) {
  probe_kernel<<<blocks, threads, 0, st>>>(reinterpret_cast<const uint4*>(p), bytes / 16, out);
}


// ---- lossless exponent-compressed weights -------------------------------------------------------------------
// Per (rt, kc, lane): 3 uint4 slots = sign|mant bytes of ks0..3 (2 slots) + 4 regs of 4-bit exponent codes (code = emax - exp).
// Elements whose exponent is below emax-15 are "escapes": they decode to code 15 and are corrected from a sparse list.
__device__ __forceinline__ uint32_t dec_reg(uint32_t m, uint32_t e, uint32_t base2, int r) {
  const uint32_t t = __byte_perm(m, 0, (r & 1) ? 0x3322 : 0x1100);
  uint32_t c;
  if (r == 0) c = (e << 7) & 0x07800780u;
  else if (r == 1) c = (e << 3) & 0x07800780u;
  else if (r == 2) c = (e >> 1) & 0x07800780u;
  else c = (e >> 5) & 0x07800780u;
  const uint32_t ex = base2 - c;
  uint32_t o;
  asm("lop3.b32 %0, %1, %2, %3, 0xea;\n" : "=r"(o) : "r"(t), "r"(0x807F807Fu), "r"(ex));  // (t & mask) | ex
  return o;
}

template <int NTB, int NCH, int NW, int S, int OM>
__global__ void __launch_bounds__(NW * 32)
dense_ckernel(const uint4* __restrict__ Wq, const bf16* __restrict__ X, float* __restrict__ part, int M, int N, int K,
              int ksplit, int* __restrict__ cnt, void* __restrict__ outp, int pair, int nout, int out_fp32, int pf,
              int emax, const int* __restrict__ eoff, const int4* __restrict__ esc) {
  extern __shared__ uint4 smem[];
  pdl_wait();
  pdl_trigger();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kcs = K / 64;
  const int cpw = kcs / (ksplit * NW);
  const int c0 = (blockIdx.y * NW + warp) * cpw;
  const int tile0 = blockIdx.x * NTB;
  const int kbeg = blockIdx.y * NW * cpw * 64, kend = kbeg + NW * cpw * 64;
  const uint32_t base2 = (uint32_t)(emax << 7) * 0x00010001u;

  for (int tok0 = 0; tok0 < M; tok0 += NCH * 8) {
    const bf16* xp[NCH];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      const int tok = tok0 + h * 8 + g;
      xp[h] = tok < M ? X + (size_t)tok * K + 16 * t + c0 * 64 : nullptr;
    }
    float acc[NTB][NCH][4];
#pragma unroll
    for (int n = 0; n < NTB; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc[n][h][r] = 0.f;

    const uint4* wsrc[NTB];
#pragma unroll
    for (int n = 0; n < NTB; n++) wsrc[n] = Wq + ((size_t)(tile0 + n) * kcs + c0) * 96 + lane;
    constexpr int STG = NTB * 3 + NCH * 2;
    uint4* ring = smem + warp * (S * STG * 32);
    if (pf && lane == 0 && tok0 == 0) {
#pragma unroll
      for (int n = 0; n < NTB; n++) l2_prefetch_bulk(wsrc[n] - lane, (uint32_t)cpw * 1536u);
    }

    auto issue = [&](int c) {
      if (c < cpw) {
#pragma unroll
        for (int n = 0; n < NTB; n++)
#pragma unroll
          for (int sl = 0; sl < 3; sl++) cp16(&ring[((c % S) * STG + n * 3 + sl) * 32 + lane], wsrc[n] + c * 96 + sl * 32);
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const bool ok = xp[h] != nullptr;
          const bf16* src = ok ? xp[h] + c * 64 : X;
          cp16z(&ring[((c % S) * STG + NTB * 3 + 2 * h) * 32 + lane], src, ok ? 16u : 0u);
          cp16z(&ring[((c % S) * STG + NTB * 3 + 2 * h + 1) * 32 + lane], src + 8, ok ? 16u : 0u);
        }
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);

#pragma unroll 1
    for (int i = 0; i < cpw; i++) {
      cp_wait<S - 2>();
      issue(i + S - 1);
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        const uint4 s0 = ring[((i % S) * STG + n * 3 + 0) * 32 + lane];
        const uint4 s1 = ring[((i % S) * STG + n * 3 + 1) * 32 + lane];
        const uint4 s2 = ring[((i % S) * STG + n * 3 + 2) * 32 + lane];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          const uint32_t m0 = ks == 0 ? s0.x : ks == 1 ? s0.z : ks == 2 ? s1.x : s1.z;
          const uint32_t m1 = ks == 0 ? s0.y : ks == 1 ? s0.w : ks == 2 ? s1.y : s1.w;
          const uint32_t e = ks == 0 ? s2.x : ks == 1 ? s2.y : ks == 2 ? s2.z : s2.w;
          const uint32_t a0 = dec_reg(m0, e, base2, 0), a1 = dec_reg(m0, e, base2, 1);
          const uint32_t a2 = dec_reg(m1, e, base2, 2), a3 = dec_reg(m1, e, base2, 3);
#pragma unroll
          for (int h = 0; h < NCH; h++) {
            const uint4 xs = ring[((i % S) * STG + NTB * 3 + 2 * h + (ks >> 1)) * 32 + lane];
            mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);
          }
        }
      }
    }
    cp_wait<0>();
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int n = 0; n < NTB; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) red[(((warp * NTB + n) * NCH + h) * 4 + r) * 32 + lane] = acc[n][h][r];
    __syncthreads();
    for (int item = threadIdx.x; item < NTB * NCH * 4 * 32; item += NW * 32) {
      const int li = item & 31, r = (item >> 5) & 3, h = (item >> 7) % NCH, n = (item >> 7) / NCH;
      float sacc = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) sacc += red[(((w * NTB + n) * NCH + h) * 4 + r) * 32 + li];
      const int row = (tile0 + n) * 16 + (li >> 2) + 8 * (r >> 1);
      const int tok = tok0 + h * 8 + 2 * (li & 3) + (r & 1);
      if (tok < M) {
        for (int q = eoff[tile0 + n]; q < eoff[tile0 + n + 1]; q++) {
          const int4 en = esc[q];
          if (en.x == row && en.y >= kbeg && en.y < kend) sacc += __int_as_float(en.z) * __bfloat162float(X[(size_t)tok * K + en.y]);
        }
        if constexpr (OM == 1) reinterpret_cast<bf16*>(part)[(size_t)tok * N + row] = __float2bfloat16(sacc);
        else part[((size_t)blockIdx.y * M + tok) * N + row] = sacc;
      }
    }
    if constexpr (OM == 2) {
      __shared__ int s_last;
      __threadfence();
      __syncthreads();
      const int ntile_out = N / (16 * NTB);
      const int slot = (blockIdx.x % ntile_out) * ((M + NCH * 8 - 1) / (NCH * 8)) + tok0 / (NCH * 8);
      if (threadIdx.x == 0) {
        const int old = atomicAdd(&cnt[slot], 1);
        s_last = (old == ksplit - 1);
      }
      __syncthreads();
      if (s_last) {
        __threadfence();
        const int obase = (blockIdx.x % ntile_out) * NTB * 16;
        const int tn = min(NCH * 8, M - tok0);
        for (int idx = threadIdx.x; idx < NTB * 16 * tn; idx += NW * 32) {
          const int r = idx % (NTB * 16), tk = idx / (NTB * 16);
          float a = 0.f;
          for (int k = 0; k < ksplit; k++) a += __ldcg(part + ((size_t)k * M + tok0 + tk) * N + obase + r);
          reinterpret_cast<bf16*>(outp)[(size_t)(tok0 + tk) * nout + obase + r] = __float2bfloat16(a);
        }
        if (threadIdx.x == 0) cnt[slot] = 0;
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

template <int NTB, int NCH, int NW, int S, int OM>
static int launch_c(const void* W, const void* X, float* part, int M, int N, int K, int ksplit, int* cnt, void* outp,
                    int nout, int emax, const int* eoff, const int4* esc, cudaStream_t st) {
  dim3 grid(N / 16 / NTB, ksplit);
  const int dyn = NW * S * (NTB * 3 + NCH * 2) * 512;
  const int dyn2 = dyn > NW * NTB * NCH * 4 * 128 ? dyn : NW * NTB * NCH * 4 * 128;
  cudaFuncSetAttribute(dense_ckernel<NTB, NCH, NW, S, OM>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn2);
  launch_k(dense_ckernel<NTB, NCH, NW, S, OM>, grid, dim3(NW * 32), (size_t)dyn2, st, g_pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const bf16*>(X), part, M, N, K, ksplit, cnt, outp, 0, nout, 0, g_pf, emax, eoff, esc);
  return 0;
}

extern "C" int dense_crun(int ntb, int nch, int nw, int s, int om, const void* W, const void* X, float* part, int M, int N,
                          int K, int ksplit, int* cnt, void* outp, int nout, int emax, const int* eoff, const void* esc, cudaStream_t st) {
#define DC(NTB_, NCH_, NW_, S_) if (ntb == NTB_ && nch == NCH_ && nw == NW_ && s == S_) { if (om == 1) return launch_c<NTB_, NCH_, NW_, S_, 1>(W, X, part, M, N, K, ksplit, cnt, outp, nout, emax, eoff, (const int4*)esc, st); if (om == 2) return launch_c<NTB_, NCH_, NW_, S_, 2>(W, X, part, M, N, K, ksplit, cnt, outp, nout, emax, eoff, (const int4*)esc, st); return launch_c<NTB_, NCH_, NW_, S_, 0>(W, X, part, M, N, K, ksplit, cnt, outp, nout, emax, eoff, (const int4*)esc, st); }
  DC(1,1,2,4) DC(1,1,4,4) DC(1,1,8,4) DC(1,2,2,4) DC(1,2,4,4) DC(1,2,8,4) DC(1,4,2,4) DC(1,4,4,4) DC(1,4,8,4) DC(2,1,2,4) DC(2,1,4,4) DC(2,1,8,4) DC(2,2,2,4) DC(2,2,4,4) DC(2,2,8,4) DC(2,4,2,4) DC(2,4,4,4) DC(2,4,8,4) DC(1,8,4,4) DC(1,8,8,4) DC(1,1,16,4) DC(1,2,16,4) DC(1,4,16,4) DC(2,1,16,4)
#undef DC
  return 1;
}


// ---- dense2: x slice staged once in shared memory, K split across a thread-block cluster (DSMEM reduction) ----
// grid (groups, KS), cluster (1, KS, 1).  Warp w owns NTB 16-row tiles of its CTA's row group and streams its weights for the
// CTA's K slice through a lane-private cp.async ring; the KS CTAs of a cluster each hold a K slice of the same rows and add their
// fp32 partial tiles through distributed shared memory.  Output bf16 [M, N] written directly.
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <int NW, int NTB, int NCH, int S, int KS>
__global__ void __launch_bounds__(NW * 32)
dense2_kernel(const uint4* __restrict__ Wq, const bf16* __restrict__ X, bf16* __restrict__ OUT, int M, int N, int K) {
  extern __shared__ __align__(16) uint8_t smem2[];
  cg::cluster_group cluster = cg::this_cluster();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rank = blockIdx.y;
  const int kcs = K / 64;
  const int kc_per = kcs / KS;
  const int Kc = K / KS;                       // elements in this CTA's K slice
  const int xp_pitch = Kc * 2 + 16;            // bytes per token row in smem
  constexpr int R = NW * NTB * 16;             // rows of the CTA's group
  uint8_t* xs = smem2;
  const size_t x_bytes = (size_t)NCH * 8 * xp_pitch;
  uint4* ring = reinterpret_cast<uint4*>(smem2 + x_bytes) + (size_t)warp * (S * NTB * 4 * 32);
  const int tile0 = blockIdx.x * NW * NTB + warp * NTB;

  const uint4* wsrc[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) wsrc[n] = Wq + ((size_t)(tile0 + n) * kcs + rank * kc_per) * 128 + lane;
  auto issue = [&](int c) {
    if (c < kc_per) {
#pragma unroll
      for (int n = 0; n < NTB; n++)
#pragma unroll
        for (int ks = 0; ks < 4; ks++) cp16(&ring[((c % S) * NTB * 4 + n * 4 + ks) * 32 + lane], wsrc[n] + c * 128 + ks * 32);
    }
    cp_commit();
  };
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);

  // weights are static: their prefetch is issued before waiting for the producer of x
  pdl_wait();
  pdl_trigger();
  // stage this CTA's x slice: [NCH*8 tokens][Kc] bf16 (rows >= M are zero)
  {
    const int vec_per_row = Kc / 8;  // 16-byte vectors per token row
    for (int idx = threadIdx.x; idx < NCH * 8 * vec_per_row; idx += NW * 32) {
      const int tok = idx / vec_per_row, v = idx % vec_per_row;
      const bool ok = tok < M;
      const bf16* src = ok ? X + (size_t)tok * K + (size_t)rank * Kc + v * 8 : X;
      cp16z(xs + (size_t)tok * xp_pitch + v * 16, src, ok ? 16u : 0u);
    }
    cp_commit();
  }
  cp_wait<0>();
  __syncthreads();

  float acc[NTB][NCH][4];
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][h][r] = 0.f;

  // all S-1 prefetch groups plus the x group are complete after the wait above; refill the pipeline normally from here
#pragma unroll 1
  for (int i = 0; i < kc_per; i++) {
    uint4 xr[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      const uint8_t* xrow = xs + (size_t)(h * 8 + g) * xp_pitch + (size_t)(i * 64 + 16 * t) * 2;
      xr[h][0] = *reinterpret_cast<const uint4*>(xrow);
      xr[h][1] = *reinterpret_cast<const uint4*>(xrow + 16);
    }
    issue(i + S - 1);
#pragma unroll
    for (int n = 0; n < NTB; n++) {
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        const uint4 a = ring[((i % S) * NTB * 4 + n * 4 + ks) * 32 + lane];
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const uint4 xv = xr[h][ks >> 1];
          mma16816(acc[n][h], a.x, a.y, a.z, a.w, (ks & 1) ? xv.z : xv.x, (ks & 1) ? xv.w : xv.y);
        }
      }
    }
    cp_wait<S - 2>();
  }
  cp_wait<0>();
  __syncthreads();  // everyone is done with x; reuse its space for the fp32 partial tile

  float* red = reinterpret_cast<float*>(smem2);   // [NCH*8][R + 4]
  const int RP = R + 4;
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int row = (warp * NTB + n) * 16 + g + 8 * (r >> 1);
        const int tok = h * 8 + 2 * t + (r & 1);
        red[tok * RP + row] = acc[n][h][r];
      }
  cluster.sync();
  // this CTA reduces R / KS rows (all tokens) over the KS partial tiles
  constexpr int RS = R / KS;
  const int row0 = rank * RS;
  const int n0 = blockIdx.x * R;
  for (int idx = threadIdx.x; idx < M * (RS / 2); idx += NW * 32) {
    const int tok = idx / (RS / 2), rr = (idx % (RS / 2)) * 2;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int src = 0; src < KS; src++) {
      const float* peer = cluster.map_shared_rank(red, src);
      s0 += peer[tok * RP + row0 + rr];
      s1 += peer[tok * RP + row0 + rr + 1];
    }
    *reinterpret_cast<__nv_bfloat162*>(OUT + (size_t)tok * N + n0 + row0 + rr) = __floats2bfloat162_rn(s0, s1);
  }
  cluster.sync();
}

template <int NW, int NTB, int NCH, int S, int KS>
static int launch2c(const void* W, const void* X, void* OUT, int M, int N, int K, cudaStream_t st) {
  const int Kc = K / KS;
  constexpr int R = NW * NTB * 16;
  if ((K / 64) % KS || N % R || R % (KS * 2) || (Kc % 8)) return 1;
  const size_t x_bytes = (size_t)NCH * 8 * (Kc * 2 + 16);
  const size_t ring_bytes = (size_t)NW * S * NTB * 4 * 32 * 16;
  size_t dyn = x_bytes + ring_bytes;
  const size_t red_bytes = (size_t)NCH * 8 * (R + 4) * 4;
  if (red_bytes > dyn) dyn = red_bytes;
  if (dyn > 227 * 1024) return 1;
  cudaFuncSetAttribute(dense2_kernel<NW, NTB, NCH, S, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / R, KS);
  cfg.blockDim = dim3(NW * 32);
  cfg.dynamicSmemBytes = dyn;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = KS;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl ? 2 : 1;
  cudaLaunchKernelEx(&cfg, dense2_kernel<NW, NTB, NCH, S, KS>, reinterpret_cast<const uint4*>(W), reinterpret_cast<const bf16*>(X),
                     reinterpret_cast<bf16*>(OUT), M, N, K);
  return 0;
}


// ---- dense2c: dense2 with lossless exponent-compressed weights (see dense_crepack) ----
template <int NW, int NTB, int NCH, int S, int KS>
__global__ void __launch_bounds__(NW * 32)
dense2c_kernel(const uint4* __restrict__ Wq, const bf16* __restrict__ X, bf16* __restrict__ OUT, int M, int N, int K, int emax,
               const int* __restrict__ eoff, const int4* __restrict__ esc) {
  extern __shared__ __align__(16) uint8_t smem2[];
  cg::cluster_group cluster = cg::this_cluster();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rank = blockIdx.y;
  const int kcs = K / 64;
  const int kc_per = kcs / KS;
  const int Kc = K / KS;
  const int xp_pitch = Kc * 2 + 16;
  constexpr int R = NW * NTB * 16;
  uint8_t* xs = smem2;
  const size_t x_bytes = (size_t)NCH * 8 * xp_pitch;
  uint4* ring = reinterpret_cast<uint4*>(smem2 + x_bytes) + (size_t)warp * (S * NTB * 3 * 32);
  const int tile0 = blockIdx.x * NW * NTB + warp * NTB;
  const uint32_t base2 = (uint32_t)(emax << 7) * 0x00010001u;

  const uint4* wsrc[NTB];
#pragma unroll
  for (int n = 0; n < NTB; n++) wsrc[n] = Wq + ((size_t)(tile0 + n) * kcs + rank * kc_per) * 96 + lane;
  auto issue = [&](int c) {
    if (c < kc_per) {
#pragma unroll
      for (int n = 0; n < NTB; n++)
#pragma unroll
        for (int sl = 0; sl < 3; sl++) cp16(&ring[((c % S) * NTB * 3 + n * 3 + sl) * 32 + lane], wsrc[n] + c * 96 + sl * 32);
    }
    cp_commit();
  };
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
  pdl_wait();
  pdl_trigger();
  {
    const int vec_per_row = Kc / 8;
    for (int idx = threadIdx.x; idx < NCH * 8 * vec_per_row; idx += NW * 32) {
      const int tok = idx / vec_per_row, v = idx % vec_per_row;
      const bool ok = tok < M;
      const bf16* src = ok ? X + (size_t)tok * K + (size_t)rank * Kc + v * 8 : X;
      cp16z(xs + (size_t)tok * xp_pitch + v * 16, src, ok ? 16u : 0u);
    }
    cp_commit();
  }
  cp_wait<0>();
  __syncthreads();

  float acc[NTB][NCH][4];
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][h][r] = 0.f;

#pragma unroll 1
  for (int i = 0; i < kc_per; i++) {
    uint4 xr[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      const uint8_t* xrow = xs + (size_t)(h * 8 + g) * xp_pitch + (size_t)(i * 64 + 16 * t) * 2;
      xr[h][0] = *reinterpret_cast<const uint4*>(xrow);
      xr[h][1] = *reinterpret_cast<const uint4*>(xrow + 16);
    }
    issue(i + S - 1);
#pragma unroll
    for (int n = 0; n < NTB; n++) {
      const uint4 s0 = ring[((i % S) * NTB * 3 + n * 3 + 0) * 32 + lane];
      const uint4 s1 = ring[((i % S) * NTB * 3 + n * 3 + 1) * 32 + lane];
      const uint4 s2 = ring[((i % S) * NTB * 3 + n * 3 + 2) * 32 + lane];
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        const uint32_t m0 = ks == 0 ? s0.x : ks == 1 ? s0.z : ks == 2 ? s1.x : s1.z;
        const uint32_t m1 = ks == 0 ? s0.y : ks == 1 ? s0.w : ks == 2 ? s1.y : s1.w;
        const uint32_t e = ks == 0 ? s2.x : ks == 1 ? s2.y : ks == 2 ? s2.z : s2.w;
        const uint32_t a0 = dec_reg(m0, e, base2, 0), a1 = dec_reg(m0, e, base2, 1);
        const uint32_t a2 = dec_reg(m1, e, base2, 2), a3 = dec_reg(m1, e, base2, 3);
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          const uint4 xv = xr[h][ks >> 1];
          mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xv.z : xv.x, (ks & 1) ? xv.w : xv.y);
        }
      }
    }
    cp_wait<S - 2>();
  }
  cp_wait<0>();
  __syncthreads();

  const size_t ring_bytes = (size_t)NW * S * NTB * 3 * 32 * 16;
  float* red = reinterpret_cast<float*>(smem2 + x_bytes + ring_bytes);
  const int RP = R + 4;
#pragma unroll
  for (int n = 0; n < NTB; n++)
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int row = (warp * NTB + n) * 16 + g + 8 * (r >> 1);
        const int tok = h * 8 + 2 * t + (r & 1);
        red[tok * RP + row] = acc[n][h][r];
      }
  __syncthreads();
  {
    // sparse corrections for the elements whose exponent fell outside the 16-code window (their k lies in exactly one slice)
    const int n0e = blockIdx.x * R;
    const int q0 = eoff[blockIdx.x * NW * NTB], q1 = eoff[(blockIdx.x + 1) * NW * NTB];
    for (int q = q0 + threadIdx.x; q < q1; q += NW * 32) {
      const int4 en = esc[q];
      const int kl = en.y - rank * Kc;
      if (kl >= 0 && kl < Kc) {
        const float d = __int_as_float(en.z);
        for (int tok = 0; tok < M; tok++)
          atomicAdd(&red[tok * RP + (en.x - n0e)], d * __bfloat162float(*reinterpret_cast<const bf16*>(xs + (size_t)tok * xp_pitch + (size_t)kl * 2)));
      }
    }
  }
  cluster.sync();
  constexpr int RS = R / KS;
  const int row0 = rank * RS;
  const int n0 = blockIdx.x * R;
  for (int idx = threadIdx.x; idx < M * (RS / 2); idx += NW * 32) {
    const int tok = idx / (RS / 2), rr = (idx % (RS / 2)) * 2;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int src = 0; src < KS; src++) {
      const float* peer = cluster.map_shared_rank(red, src);
      s0 += peer[tok * RP + row0 + rr];
      s1 += peer[tok * RP + row0 + rr + 1];
    }
    *reinterpret_cast<__nv_bfloat162*>(OUT + (size_t)tok * N + n0 + row0 + rr) = __floats2bfloat162_rn(s0, s1);
  }
  cluster.sync();
}

extern "C" int dense2_run(int nw, int ntb, int nch, int s, int ks, const void* W, const void* X, void* OUT, int M, int N, int K,
                          cudaStream_t st) {
#define D2(NW_, NTB_, NCH_, S_, KS_) if (nw == NW_ && ntb == NTB_ && nch == NCH_ && s == S_ && ks == KS_) return launch2c<NW_, NTB_, NCH_, S_, KS_>(W, X, OUT, M, N, K, st);
#define D2K(NW_, NTB_, NCH_, S_) D2(NW_, NTB_, NCH_, S_, 2) D2(NW_, NTB_, NCH_, S_, 4) D2(NW_, NTB_, NCH_, S_, 8)
#define D2S(NW_, NTB_, NCH_) D2K(NW_, NTB_, NCH_, 4) D2K(NW_, NTB_, NCH_, 6)
#define D2N(NW_, NTB_) D2S(NW_, NTB_, 1) D2S(NW_, NTB_, 2) D2S(NW_, NTB_, 4)
  D2N(8, 1) D2N(4, 1) D2N(4, 2)
#undef D2N
#undef D2S
#undef D2K
#undef D2
  return 2;
}


template <int NW, int NTB, int NCH, int S, int KS>
static int launch2cc(const void* W, const void* X, void* OUT, int M, int N, int K, int emax, const int* eoff, const void* esc, cudaStream_t st) {
  const int Kc = K / KS;
  constexpr int R = NW * NTB * 16;
  if ((K / 64) % KS || N % R || R % (KS * 2) || (Kc % 8)) return 1;
  const size_t x_bytes = (size_t)NCH * 8 * (Kc * 2 + 16);
  const size_t ring_bytes = (size_t)NW * S * NTB * 3 * 32 * 16;
  const size_t red_bytes = (size_t)NCH * 8 * (R + 4) * 4;
  const size_t dyn = x_bytes + ring_bytes + red_bytes;
  if (dyn > 227 * 1024) return 1;
  cudaFuncSetAttribute(dense2c_kernel<NW, NTB, NCH, S, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / R, KS);
  cfg.blockDim = dim3(NW * 32);
  cfg.dynamicSmemBytes = dyn;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = KS;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl ? 2 : 1;
  cudaLaunchKernelEx(&cfg, dense2c_kernel<NW, NTB, NCH, S, KS>, reinterpret_cast<const uint4*>(W), reinterpret_cast<const bf16*>(X),
                     reinterpret_cast<bf16*>(OUT), M, N, K, emax, eoff, reinterpret_cast<const int4*>(esc));
  return 0;
}

extern "C" int dense2c_run(int nw, int ntb, int nch, int s, int ks, const void* W, const void* X, void* OUT, int M, int N, int K,
                           int emax, const int* eoff, const void* esc, cudaStream_t st) {
#define D2(NW_, NTB_, NCH_, S_, KS_) if (nw == NW_ && ntb == NTB_ && nch == NCH_ && s == S_ && ks == KS_) return launch2cc<NW_, NTB_, NCH_, S_, KS_>(W, X, OUT, M, N, K, emax, eoff, esc, st);
#define D2K(NW_, NTB_, NCH_, S_) D2(NW_, NTB_, NCH_, S_, 2) D2(NW_, NTB_, NCH_, S_, 4) D2(NW_, NTB_, NCH_, S_, 8)
#define D2S(NW_, NTB_, NCH_) D2K(NW_, NTB_, NCH_, 4) D2K(NW_, NTB_, NCH_, 6)
#define D2N(NW_, NTB_) D2S(NW_, NTB_, 1) D2S(NW_, NTB_, 2) D2S(NW_, NTB_, 4)
  D2N(8, 1) D2N(4, 1) D2N(4, 2)
#undef D2N
#undef D2S
#undef D2K
#undef D2
  return 2;
}

extern "C" void dense_reduce(const float* part, void* out, int M, int nout, int ntot, int ksplit, int pair, int fp32,
                             cudaStream_t st) {
  const int n = M * nout;
  launch_k(reduce_kernel, dim3((n + 255) / 256), dim3(256), 0, st, g_pdl != 0, part, out, M, nout, ntot, ksplit, pair, fp32);
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int dense_run(int, int, int, int, int, const void*, const void*, float*, int, int, int, int, int*, void*, int, int, int, cudaStream_t);
extern "C" void dense_reduce(const float*, void*, int, int, int, int, int, int, cudaStream_t);
extern "C" void dense_set_pdl(int);
extern "C" void dense_set_pf(int);
extern "C" void dense_set_dbg(int);
extern "C" int dense2_run(int, int, int, int, int, const void*, const void*, void*, int, int, int, cudaStream_t);
extern "C" int dense2c_run(int, int, int, int, int, const void*, const void*, void*, int, int, int, int, const int*, const void*, cudaStream_t);
void dense_set_dbg_t(int64_t v) { dense_set_dbg((int)v); }
extern "C" int dense_crun(int, int, int, int, int, const void*, const void*, float*, int, int, int, int, int*, void*, int, int, const int*, const void*, cudaStream_t);
extern "C" void probe_run(const void*, size_t, float*, int, int, cudaStream_t);
void probe_t(torch::Tensor p, torch::Tensor out, int64_t blocks, int64_t threads) {
  probe_run(p.data_ptr(), (size_t)p.numel() * p.element_size(), out.data_ptr<float>(), (int)blocks, (int)threads, at::cuda::getCurrentCUDAStream());
}
void dense_set_pdl_t(int64_t v) { dense_set_pdl((int)v); }
void dense_set_pf_t(int64_t v) { dense_set_pf((int)v); }

void dense_cgemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                 torch::Tensor out, int64_t ksplit, int64_t nout, torch::Tensor cnt, int64_t fix, int64_t emax, torch::Tensor eoff,
                 torch::Tensor esc) {
  const int M = X.size(0), K = X.size(1);
  const int N = (int)nout;
  const bool direct = ksplit == 1;
  const int om = direct ? 1 : (fix ? 2 : 0);
  int r = dense_crun((int)ntb, (int)nch, (int)nw, (int)s, om, W.data_ptr(), X.data_ptr(),
                     direct ? reinterpret_cast<float*>(out.data_ptr()) : part.data_ptr<float>(), M, N, K, (int)ksplit,
                     cnt.numel() ? cnt.data_ptr<int>() : nullptr, out.data_ptr(), (int)nout, (int)emax, eoff.data_ptr<int>(),
                     esc.data_ptr(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported dense config");
  if (om == 0) dense_reduce(part.data_ptr<float>(), out.data_ptr(), M, (int)nout, N, (int)ksplit, 0, 0, at::cuda::getCurrentCUDAStream());
}

void dense2_gemm(int64_t nw, int64_t ntb, int64_t nch, int64_t s, int64_t ks, torch::Tensor W, torch::Tensor X, torch::Tensor out) {
  int r = dense2_run((int)nw, (int)ntb, (int)nch, (int)s, (int)ks, W.data_ptr(), X.data_ptr(), out.data_ptr(), (int)X.size(0), (int)out.size(1), (int)X.size(1),
                     at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported dense2 config");
}

void dense2c_gemm(int64_t nw, int64_t ntb, int64_t nch, int64_t s, int64_t ks, torch::Tensor W, torch::Tensor X, torch::Tensor out,
                  int64_t emax, torch::Tensor eoff, torch::Tensor esc) {
  int r = dense2c_run((int)nw, (int)ntb, (int)nch, (int)s, (int)ks, W.data_ptr(), X.data_ptr(), out.data_ptr(), (int)X.size(0), (int)out.size(1),
                      (int)X.size(1), (int)emax, eoff.data_ptr<int>(), esc.data_ptr(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported dense2c config");
}

void dense_gemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                torch::Tensor out, int64_t ksplit, int64_t pair, int64_t nout, torch::Tensor cnt, int64_t fix) {
  const int M = X.size(0), K = X.size(1);
  const int N = (int)(pair ? 2 * nout : nout);
  const bool fp32 = out.scalar_type() == torch::kFloat32;
  const bool direct = ksplit == 1 && !pair && !fp32;
  const int om = direct ? 1 : (fix && ksplit > 1 ? 2 : 0);
  int r = dense_run((int)ntb, (int)nch, (int)nw, (int)s, om, W.data_ptr(), X.data_ptr(),
                    direct ? reinterpret_cast<float*>(out.data_ptr()) : part.data_ptr<float>(), M, N, K, (int)ksplit,
                    cnt.numel() ? cnt.data_ptr<int>() : nullptr, out.data_ptr(), (int)pair, (int)nout, fp32 ? 1 : 0,
                    at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported dense config");
  if (om == 0)
    dense_reduce(part.data_ptr<float>(), out.data_ptr(), M, (int)nout, N, (int)ksplit, (int)pair, fp32,
                 at::cuda::getCurrentCUDAStream());
}
'''
_DENSE_CPP = r'''
void probe_t(torch::Tensor p, torch::Tensor out, int64_t blocks, int64_t threads);
void dense_set_pdl_t(int64_t v);
void dense_set_pf_t(int64_t v);
void dense_set_dbg_t(int64_t v);
void dense2_gemm(int64_t nw, int64_t ntb, int64_t nch, int64_t s, int64_t ks, torch::Tensor W, torch::Tensor X, torch::Tensor out);
void dense2c_gemm(int64_t nw, int64_t ntb, int64_t nch, int64_t s, int64_t ks, torch::Tensor W, torch::Tensor X, torch::Tensor out,
                  int64_t emax, torch::Tensor eoff, torch::Tensor esc);
void dense_gemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                torch::Tensor out, int64_t ksplit, int64_t pair, int64_t nout, torch::Tensor cnt, int64_t fix);
void dense_cgemm(int64_t ntb, int64_t nch, int64_t nw, int64_t s, torch::Tensor W, torch::Tensor X, torch::Tensor part,
                 torch::Tensor out, int64_t ksplit, int64_t nout, torch::Tensor cnt, int64_t fix, int64_t emax, torch::Tensor eoff,
                 torch::Tensor esc);
'''
def _dense_repack(w):
    """w: bf16 [N, K] -> bf16 tiles [N/16, K/64, 4(ks), 32(lane), 8] (one A fragment per lane per k16 step)."""
    N, K = w.shape
    RT, KCn = N // 16, K // 64
    # [rt, rh, g, kc, t, ks, p, h]  with k = kc*64 + t*16 + ks*4 + p*2 + h, row = rt*16 + rh*8 + g
    v = w.view(RT, 2, 8, KCn, 4, 4, 2, 2)
    # fragment regs: (rh, p) -> reg index = rh + 2*p, each reg = (h=0,h=1)
    out = v.permute(0, 3, 5, 2, 4, 6, 1, 7)  # [rt, kc, ks, g, t, p, rh, h]
    return out.contiguous().view(RT, KCn, 4, 32, 8)


def dense_crepack(w):
    """Lossless exponent compression of bf16 [N, K]: (tiles uint8 [N/16, K/64, 3, 32, 16], emax, esc offsets, escapes int32 [n, 4]) or None."""
    import torch
    N, K = w.shape
    RT, KCn = N // 16, K // 64
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    emax = int(e.max())
    if emax < 20 or emax > 254:
        return None
    code = emax - e
    escm = code > 15
    nesc = int(escm.sum())
    if nesc > N * K // 2000:
        return None
    code = code.clamp(max=15)
    sm = ((sign << 7) | m).to(torch.uint8)
    smt = dense_repack(sm)            # [RT, KCn, 4(ks), 32, 8]
    cdt = dense_repack(code.to(torch.int32))
    smb = smt.permute(0, 1, 3, 2, 4).contiguous().view(RT, KCn, 32, 2, 16).permute(0, 1, 3, 2, 4)  # [RT, KCn, 2, 32, 16]
    sh = torch.tensor([4 * (j >> 1) + 16 * (j & 1) for j in range(8)], device=w.device, dtype=torch.int32)
    ev = (cdt << sh).sum(-1).to(torch.int32)          # [RT, KCn, 4, 32]
    eb = ev.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(RT, KCn, 1, 32, 16)
    tiles = torch.cat([smb, eb], 2).contiguous()
    rows, cols = escm.nonzero(as_tuple=True)
    wt = w[rows, cols].float()
    wd_bits = ((sign[rows, cols] << 15) | ((emax - 15) << 7) | m[rows, cols]).to(torch.int16)
    wd = wd_bits.view(torch.bfloat16).float()
    delta = (wt.double() - wd.double()).float()
    esc = torch.stack([rows.to(torch.int32), cols.to(torch.int32), delta.view(torch.int32), torch.zeros_like(rows, dtype=torch.int32)], 1).contiguous()
    if nesc == 0:
        esc = torch.zeros((1, 4), dtype=torch.int32, device=w.device)
    cntp = torch.bincount(rows // 16, minlength=RT)
    eoff = torch.zeros(RT + 1, dtype=torch.int32, device=w.device)
    eoff[1:] = torch.cumsum(cntp, 0).to(torch.int32)
    return tiles, emax, eoff, esc

_DMOD = None
_DENSE_PDL = True
_DENSE_PF = False
# (layer) -> [(max tokens, (ntb, nch, nw, s, ksplit, fix))]
_DENSE_CFG = {}


def _dense_module():
    global _DMOD
    if _DMOD is None:
        from torch.utils.cpp_extension import load_inline
        _DMOD = load_inline("muse_dense_bf16", _DENSE_CPP, cuda_sources=_DENSE_CUDA,
                            functions=["dense_gemm", "dense_set_pdl_t", "dense_set_pf_t"],
                            extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
        _DMOD.dense_set_pdl_t(int(_DENSE_PDL and torch.cuda.get_device_capability()[0] >= 9))
        _DMOD.dense_set_pf_t(int(_DENSE_PF and torch.cuda.get_device_capability()[0] >= 9))
    return _DMOD


def _dense_prepare(weights, device):
    state = {"mod": _dense_module(), "tiles": {}, "cnt": {}}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state["tiles"][name] = _dense_repack(weights[name].contiguous())
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["tiles"]["router"] = _dense_repack(torch.cat([hi, lo], 0).contiguous())
    for name in state["tiles"]:
        state["cnt"][name] = torch.zeros(4096, dtype=torch.int32, device=device)
    state["b5_poolert"] = weights["pooler"].T.contiguous()
    return state


def _dense_cfg(layer, m):
    for bound, cfg in _DENSE_CFG.get(layer, ()):
        if m <= bound:
            return cfg
    return None


def _dgemm(x, layer, n, state):
    """x @ W[:n].T through the CUDA weight-streaming kernel when configured, else cuBLAS."""
    m = x.shape[0]
    cfg = _dense_cfg(layer, m)
    if cfg is None:
        return _b5(x.contiguous(), layer, n, state)
    ntb, nch, nw, s, ksplit, fix = cfg
    d = state["dn"]
    router = layer == "router"
    ntot = 512 if router else n
    wt = d["tiles"][layer][: ntot // 16]
    part = torch.empty((ksplit, m, ntot), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    d["mod"].dense_gemm(ntb, nch, nw, s, wt, x.contiguous(), part, out, ksplit, 1 if router else 0, n, d["cnt"][layer], fix)
    return out


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    if x.shape[0] <= WM_MAX_TOKENS:
        return _experts_wm(x, topk_weights, topk_ids, state)
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    if 32 <= num_tokens <= 512:
        block_size_m = 16
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
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
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


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



def fused_split_router(x, hi, lo):
    """Return FP32 x @ (hi + lo), with transposed contiguous BF16 weights."""
    m, k = x.shape
    n = hi.shape[1]
    bm, bn, bk = (16 if m <= 32 else 32), 64, 128
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    split = min(16, triton.next_power_of_2(triton.cdiv(256, tiles)))
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    _matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        x, hi, lo, partial, m, n, k, bm, bn, bk, split, True, n,
        num_warps=4, num_stages=3,
    )
    _finish[(triton.cdiv(m * n, 512),)](partial, output, m * n, split, 512)
    return output


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
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
        partner = tl.load(X + token * STRIDE + heads[:, None] * D + (dims[None, :] ^ 1),
                          valid[:, None], 0).to(tl.float32) * inv_rms[:, None]
        partner = partner.to(tl.bfloat16).to(tl.float32)
        position = tl.load(POS + token).to(tl.float32)
        frequency = tl.exp((-2.0 * (dims // 2).to(tl.float32) / D) * tl.log(THETA))
        angle = position * frequency
        cs, sn = libdevice.cos(angle), libdevice.sin(angle)
        normalized = normalized * cs[None, :] + tl.where(
            (dims[None, :] & 1) == 0, -partner, partner) * sn[None, :]
    normalized = normalized.to(tl.bfloat16).to(tl.float32)
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


@triton.jit
def _sliding_partials(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE,
                      HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                      LENGTH: tl.constexpr, PAGE: tl.constexpr, TABLE_N: tl.constexpr,
                      WINDOW: tl.constexpr, SPLITS: tl.constexpr,
                      CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      SM_SCALE: tl.constexpr, SCALED: tl.constexpr):
    b, kh, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    GROUP: tl.constexpr = HQ // HKV
    rows = tl.arange(0, BM)
    dims = tl.arange(0, D)
    token = rows // GROUP
    qh = kh * GROUP + rows % GROUP
    prefix = tl.load(SEQLEN + b)
    position = prefix + token
    if WINDOW:
        first = tl.maximum(prefix - WINDOW + 1, 0)
    else:
        first = 0
    key_scale, value_scale = 1., 1.
    if SCALED:
        key_scale, value_scale = tl.load(KS), tl.load(VS)
    q = tl.load(Q + ((b * LENGTH + token[:, None]) * HQ + qh[:, None]) * D + dims[None, :],
                (rows < LENGTH * GROUP)[:, None], 0.0).to(tl.bfloat16)
    max_score = tl.full((BM,), -float('inf'), tl.float32)
    normalizer = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, D), 0., tl.float32)
    begin = first + split * CHUNK
    end = tl.minimum(begin + CHUNK, prefix + LENGTH)
    for start in range(begin, end, BN):
        logical = start + tl.arange(0, BN)
        pages = logical // PAGE
        valid = (logical < prefix + LENGTH) & (pages < TABLE_N)
        physical = tl.load(TABLE + b * TABLE_N + pages, valid, -1)
        valid = valid & (physical >= 0)
        cache_row = tl.maximum(physical, 0) * PAGE + logical % PAGE
        k = tl.load(KC + (cache_row[None, :] * HKV + kh) * D + dims[:, None],
                    valid[None, :], 0.0).to(tl.bfloat16)
        scores = tl.dot(q, k) * (SM_SCALE * key_scale)
        visible = (valid[None, :] & (logical[None, :] <= position[:, None]) &
                   ((WINDOW == 0) | (logical[None, :] > position[:, None] - WINDOW)) &
                   (rows < LENGTH * GROUP)[:, None])
        scores = tl.where(visible, scores, -float('inf'))
        next_max = tl.maximum(max_score, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float('inf'), 0., next_max)
        correction = tl.exp(max_score - safe_max)
        probabilities = tl.exp(scores - safe_max[:, None])
        normalizer = normalizer * correction + tl.sum(probabilities, 1)
        acc *= correction[:, None]
        v = tl.load(VC + (cache_row[:, None] * HKV + kh) * D + dims[None, :],
                    valid[:, None], 0.0).to(tl.bfloat16)
        acc = tl.dot(probabilities.to(tl.bfloat16), v, acc)
        max_score = next_max
    inverse = 1. / tl.where(normalizer > 0, normalizer, 1.)
    output = acc * inverse[:, None] * value_scale
    logsumexp = tl.where(normalizer > 0, max_score + tl.log(normalizer), -float('inf'))
    base = ((b * HKV + kh) * SPLITS + split) * BM
    tl.store(PART + (base + rows[:, None]) * D + dims[None, :], output)
    tl.store(LSE + base + rows, logsumexp)


@triton.jit
def _merge_norm_gate(PART, LSE, X, OUT,
                     HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                     LENGTH: tl.constexpr, STRIDE: tl.constexpr,
                     SPLITS: tl.constexpr, BM: tl.constexpr):
    t, qh = tl.program_id(0), tl.program_id(1)
    b, token = t // LENGTH, t % LENGTH
    GROUP: tl.constexpr = HQ // HKV
    kh, row = qh // GROUP, token * GROUP + qh % GROUP
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, D)
    bases = ((b * HKV + kh) * SPLITS + splits) * BM + row
    logs = tl.load(LSE + bases)
    high = tl.max(logs, 0)
    factors = tl.exp(logs - high)
    factors /= tl.sum(factors, 0)
    partial = tl.load(PART + bases[:, None] * D + dims[None, :])
    value = tl.sum(partial * factors[:, None], 0).to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / D + 1.e-5)
    gate = tl.load(X + t * STRIDE + (HQ + 2 * HKV + qh) * D + dims).to(tl.float32)
    output = value * inv_rms * tl.sigmoid(gate)
    tl.store(OUT + (t * HQ + qh) * D + dims, output)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _dgemm(data["attn_input"], "qkv_gate", 2304, state)
    q = torch.empty((tokens, hq, d), device=qkv.device, dtype=FP8)
    gate = qkv[:, (hq + 2*hkv)*d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    use_transposed = (length == 1 and 8 <= batch <= 64 and hq == 16 and hkv == 2 and d == 64
                      and data['page_table'].shape[1] * cfg['page_size'] <= (1 << 20) + 64
                      and cfg['page_size'] & (cfg['page_size'] - 1) == 0)
    fused_slide = (not use_transposed and _SLIDE_STAGES and cfg['window'] and length == 1
                   and cfg['page_size'] & (cfg['page_size'] - 1) == 0)
    if not fused_slide:
        _prepare_qkv[(tokens, triton.cdiv(hq + hkv, 4))](
            qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
            q, k_cache, v_cache, hq, hkv, d, qkv.stride(0), cfg["rope_theta"],
            scaled, 4, num_warps=4, enable_fp_fusion=False)
    fd = None if fused_slide else _fd_choice(cfg, data['page_table'].shape[1] * cfg['page_size'])
    if use_transposed:
        o = _sp_forward(data, q, qkv).view(tokens, hq*d)
    elif fused_slide:
        o = _slide_forward(data, _SLIDE_STAGES, qkv).view(tokens, hq*d)
    elif fd is not None and cfg['page_size'] & (cfg['page_size'] - 1) == 0:
        o = _fd_forward(data, q, qkv, fd[0], fd[1], variant=fd[2]).view(tokens, hq*d)
    elif (cfg['window'] and batch <= 4) or (not cfg['window'] and length == 1):
        # Partition the actual window, never the entire logical context.
        # The extra LENGTH-1 rows cover the union of the causal verify windows.
        if cfg['window']:
            splits = 16 if batch <= 4 else 4
            block_n = 64
            span = cfg['window'] + length - 1
        else:
            splits = min(128, max(4, triton.next_power_of_2(triton.cdiv(512, batch * hkv))))
            block_n = 128
            span = data['page_table'].shape[1] * cfg['page_size']
        chunk = triton.cdiv(triton.cdiv(span, splits), block_n) * block_n
        bm = max(16, triton.next_power_of_2(length * (hq // hkv)))
        partial = torch.empty((batch, hkv, splits, bm, d), device=qkv.device, dtype=torch.float32)
        lse = torch.empty((batch, hkv, splits, bm), device=qkv.device, dtype=torch.float32)
        result = torch.empty((batch, length, hq*d), device=qkv.device, dtype=torch.bfloat16)
        _sliding_partials[(batch, hkv, splits)](
            q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'],
            data['k_scale'], data['v_scale'], partial, lse, hq, hkv, d,
            length, cfg['page_size'], data['page_table'].shape[1], cfg['window'],
            splits, chunk, bm, block_n, cfg['sm_scale'] * cfg['qk_scale_factor'],
            scaled, num_warps=4, num_stages=2)
        _merge_norm_gate[(tokens, hq)](
            partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0),
            splits, bm, num_warps=4)
        o = result.view(tokens, hq*d)
    else:
        window = cfg["window"]
        descale = {}
        if scaled:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                           v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
        out = flash_attn_with_kvcache(
            q=q.view(tokens, hq, d).to(FP8),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            scheduler_metadata=data["scheduler_metadata"],
            **descale,
        )
        if isinstance(out, tuple):
            out = out[0]
        o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _dgemm(o, "o", HIDDEN, state)  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _dgemm(ffn_input, "pooler", D_EXPERT, state)
    with torch.cuda.stream(side_stream):
        if split_router and _dense_cfg("router", tokens) is not None:
            logits = _dgemm(ffn_input, "router", 256, state)
        elif split_router and tokens <= 256:
            logits = fused_split_router(ffn_input, state["router_hi_t"], state["router_lo_t"])
        elif split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _dgemm(expert_output, "unpooler", UNPOOL_SLICE, state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _dgemm(expert_output, "unpooler", HIDDEN, state)

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


_SP_CPP = 'void sparse_decode_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,double,int64_t);'
_SP_CUDA = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n#include <stdint.h>\n__device__ __forceinline__ int off(int key,int byte){return key*128+(((byte>>4)^(key&7))<<4)+(byte&15);}\n__device__ __forceinline__ uint32_t cv2(uint32_t x){uint32_t y;asm("{.reg .b16 a,b; mov.b32 {a,b},%1; cvt.rn.f16x2.e4m3x2 %0,a;}":"=r"(y):"r"(x));return y;}\n__device__ __forceinline__ void ld(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}\n__device__ __forceinline__ void ldt(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}\n__device__ __forceinline__ void qmma(float* c,uint32_t* a,uint32_t b0,uint32_t b1){asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));}\n__device__ __forceinline__ void pmma(float* c,uint32_t* a,uint32_t* b){asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));}\n\ntemplate<int ST>\n__global__ void __launch_bounds__(128,2) trans_decode(const uint8_t* Q,const uint8_t* KC,const uint8_t* VC,\n const int* TABLE,const int* SEQ,const float* KS,const float* VS,float* PART,float* LSE,\n int table_n,int page,int window,int splits,int chunk,float sm_scale,int scaled){\n extern __shared__ __align__(16) uint8_t mem[];\n int tid=threadIdx.x,lane=tid&31,warp=tid>>5,h=warp&1,part=warp>>1,g=lane>>2,t=lane&3,b=blockIdx.y,split=blockIdx.x;\n int prefix=SEQ[b],first=window?max(0,prefix-window+1):0,begin=first+split*chunk;\n int end=min(begin+chunk,prefix+1),n=max(0,end-begin),tiles=(n+31)/32;\n int* rows=reinterpret_cast<int*>(mem+ST*8192);\n int shift=__ffs(page)-1,mask=page-1;\n for(int i=tid;i<n;i+=128){int tok=begin+i,pg=tok>>shift,phys=pg<table_n?TABLE[b*table_n+pg]:-1;rows[i]=phys<0?0:(phys<<shift)+(tok&mask);}\n uint32_t qb[2][2];\n#pragma unroll\n for(int k=0;k<2;k++){const uint8_t* q=Q+(long)b*1024+(h*8+g)*64+k*32+t*4;qb[k][0]=*reinterpret_cast<const uint32_t*>(q);qb[k][1]=*reinterpret_cast<const uint32_t*>(q+16);}\n float scale=sm_scale*(scaled?KS[0]:1.f)*1.4426950408889634f,vscale=scaled?VS[0]:1.f;\n __syncthreads();\n auto issue=[&](int tile){if(tile<tiles){uint8_t* dst=mem+(tile%ST)*8192;\n#pragma unroll\n for(int i=tid;i<256;i+=128){int key=i>>3,byte=(i&7)*16,li=tile*32+key;long rr=li<n?rows[li]:0;uint32_t ka=__cvta_generic_to_shared(dst+off(key,byte)),va=__cvta_generic_to_shared(dst+4096+off(key,byte));asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ka),"l"(KC+rr*128+byte));asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(va),"l"(VC+rr*128+byte));}}\n asm volatile("cp.async.commit_group;");};\n#pragma unroll\n for(int i=0;i<ST-1;i++)issue(i);\n float o[4][4]={},mm[2]={-INFINITY,-INFINITY},ll[2]={0.f,0.f};\n#pragma unroll 1\n for(int tile=0;tile<tiles;tile++){\n  asm volatile("cp.async.wait_group %0;"::"n"(ST-2));__syncthreads();issue(tile+ST-1);\n  const uint8_t* raw=mem+(tile%ST)*8192;float s[4]={};\n#pragma unroll\n  for(int k=0;k<2;k++){uint32_t a[4];ld(a,raw+off(part*16+(lane&7)+8*((lane>>3)&1),h*64+k*32+16*(lane>>4)));qmma(s,a,qb[k][0],qb[k][1]);}\n#pragma unroll\n  for(int r=0;r<4;r++)s[r]=(tile*32+part*16+g+(r>=2?8:0)<n)?s[r]*scale:-INFINITY;\n  float mx[2]={fmaxf(s[0],s[2]),fmaxf(s[1],s[3])};\n#pragma unroll\n  for(int c=0;c<2;c++){\n#pragma unroll\n   for(int x=4;x<=16;x*=2)mx[c]=fmaxf(mx[c],__shfl_xor_sync(0xffffffff,mx[c],x));\n   float nm=fmaxf(mm[c],mx[c]),safe=nm==-INFINITY?0.f:nm,corr=exp2f(mm[c]-safe);ll[c]*=corr;mm[c]=nm;\n#pragma unroll\n   for(int d=0;d<4;d++){o[d][c]*=corr;o[d][c+2]*=corr;}\n  }\n  float p[4];\n#pragma unroll\n  for(int r=0;r<4;r++){float safe=mm[r&1]==-INFINITY?0.f:mm[r&1];p[r]=exp2f(s[r]-safe);}\n#pragma unroll\n  for(int c=0;c<2;c++){float sum=p[c]+p[c+2];\n#pragma unroll\n   for(int x=4;x<=16;x*=2)sum+=__shfl_xor_sync(0xffffffff,sum,x);ll[c]+=sum;}\n  __half2 hp0=__floats2half2_rn(p[0],p[1]),hp8=__floats2half2_rn(p[2],p[3]);\n  uint32_t p0=*reinterpret_cast<uint32_t*>(&hp0),p8=*reinterpret_cast<uint32_t*>(&hp8),z[2];\n  int sa=8*t+(g>>1),sb=sa+4,sh=(g&1)*16;\n  z[0]=((__shfl_sync(0xffffffff,p0,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p0,sb)>>sh)<<16);\n  z[1]=((__shfl_sync(0xffffffff,p8,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p8,sb)>>sh)<<16);\n#pragma unroll\n  for(int pr=0;pr<2;pr++){uint32_t r[4];ldt(r,raw+4096+off(part*16+(lane&7)+8*((lane>>3)&1),h*64+pr*32+16*(lane>>4)));\n#pragma unroll\n   for(int odd=0;odd<2;odd++){uint32_t a[4];a[0]=cv2(__byte_perm(r[0],0,odd?0x4431:0x4420));a[1]=cv2(__byte_perm(r[2],0,odd?0x4431:0x4420));a[2]=cv2(__byte_perm(r[1],0,odd?0x4431:0x4420));a[3]=cv2(__byte_perm(r[3],0,odd?0x4431:0x4420));pmma(o[2*pr+odd],a,z);}}\n  __syncthreads();\n }\n asm volatile("cp.async.wait_group 0;");__syncthreads();\n float* smm=reinterpret_cast<float*>(mem);float* sml=smm+32;float* smo=sml+32;\n#pragma unroll\n for(int c=0;c<2;c++){int head=h*8+2*t+c,row=part*16+head;if(g==0){smm[row]=mm[c];sml[row]=ll[c];}\n#pragma unroll\n for(int pr=0;pr<2;pr++)for(int odd=0;odd<2;odd++){int d=pr*32+2*g+odd;smo[row*64+d]=o[2*pr+odd][c];smo[row*64+d+16]=o[2*pr+odd][c+2];}}\n __syncthreads();\n if(tid<16){float high=fmaxf(smm[tid],smm[16+tid]),ls=0.f;\n#pragma unroll\n for(int p=0;p<2;p++){float m=smm[p*16+tid],f=m==-INFINITY?0.f:exp2f(m-high);ls+=sml[p*16+tid]*f;smm[p*16+tid]=f;}\n sml[tid]=ls>0.f?1.f/ls:0.f;long row=((long)(b*2+(tid>>3))*splits+split)*16+(tid&7);LSE[row]=ls>0.f?(high+log2f(ls))*0.6931471805599453f:-INFINITY;}\n __syncthreads();\n for(int i=tid;i<1024;i+=128){int head=i>>6,d=i&63;float sum=fmaf(smo[i],smm[head],smo[1024+i]*smm[16+head]);long dst=(((long)(b*2+(head>>3))*splits+split)*16+(head&7))*64+d;PART[dst]=sum*sml[head]*vscale;}\n}\nvoid sparse_decode_run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor table,torch::Tensor seq,\n torch::Tensor ks,torch::Tensor vs,torch::Tensor part,torch::Tensor lse,int64_t page,int64_t window,\n int64_t splits,int64_t chunk,double scale,int64_t scaled){\n int dyn=3*8192+((chunk+127)/128)*128*4;cudaFuncSetAttribute(trans_decode<3>,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);\n trans_decode<3><<<dim3(splits,seq.numel()),128,dyn,at::cuda::getCurrentCUDAStream()>>>((const uint8_t*)q.data_ptr(),(const uint8_t*)k.data_ptr(),(const uint8_t*)v.data_ptr(),table.data_ptr<int>(),seq.data_ptr<int>(),ks.data_ptr<float>(),vs.data_ptr<float>(),part.data_ptr<float>(),lse.data_ptr<float>(),table.size(1),page,window,splits,chunk,scale,scaled);\n auto st=cudaGetLastError();TORCH_CHECK(st==cudaSuccess,cudaGetErrorString(st));}\n'

_SP_MOD = None

def _sp_module():
    global _SP_MOD
    if _SP_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _SP_MOD = load_inline('muse_dense_transposed_decode_v1', _SP_CPP, cuda_sources=_SP_CUDA,
            functions=['sparse_decode_run'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'], verbose=False)
    return _SP_MOD

def _sp_forward(data, q, qkv):
    cfg = data['config']
    batch = cfg['batch']
    window = cfg['window']
    # Target enough CTAs to fill H200 while keeping each chunk economical.
    splits = max(4 if window else 8, triton.next_power_of_2(triton.cdiv(256 if window else 512, batch)))
    span = window if window else data['page_table'].shape[1] * cfg['page_size']
    splits = max(splits, triton.cdiv(span, 8192))
    chunk = triton.cdiv(triton.cdiv(span, splits), 32) * 32
    partial = torch.empty((batch, 2, splits, 16, 64), dtype=torch.float32, device=qkv.device)
    lse = torch.empty((batch, 2, splits, 16), dtype=torch.float32, device=qkv.device)
    _sp_module().sparse_decode_run(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],
        data['k_scale'],data['v_scale'],partial,lse,cfg['page_size'],window,splits,chunk,
        cfg['sm_scale']*cfg['qk_scale_factor'],int(cfg['kv_scale'] != 1.0))
    result = torch.empty((batch,1,1024),dtype=torch.bfloat16,device=qkv.device)
    _merge_norm_gate[(batch,16)](partial,lse,qkv,result,16,2,64,1,qkv.stride(0),splits,16,num_warps=4)
    return result


@triton.jit
def _pre_rms_kernel(X,W,Y,N:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,B)
    x=tl.load(X+row*N+c,c<N,0).to(tl.float32)
    w=tl.load(W+c,c<N,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/N+1.e-5)
    tl.store(Y+row*N+c,(x*inv)*w,c<N)
def _pre_rms(x,w,warps=4):
    y=torch.empty_like(x)
    _pre_rms_kernel[(x.shape[0],)](x,w,y,x.shape[1],triton.next_power_of_2(x.shape[1]),num_warps=warps,enable_fp_fusion=False)
    return y

class _FastPreExpertNorm:
    def __init__(self,weight):
        self.weight=weight
    def __call__(self,x):
        return _pre_rms(x,self.weight,4 if x.shape[0]<=16 else 8)
_before_fast_pre_norm_prepare=prepare
def prepare(weights):
    state=_before_fast_pre_norm_prepare(weights)
    state['pre_expert_norm']=_FastPreExpertNorm((weights['pre_expert_norm']+1).contiguous())
    return state
