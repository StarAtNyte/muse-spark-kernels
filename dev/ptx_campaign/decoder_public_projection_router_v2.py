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
    expert_input = state["pre_expert_norm"](expert_input)
    with torch.cuda.stream(side_stream):
        if split_router and _dense_cfg("router", tokens) is not None:
            logits = _dgemm(ffn_input, "router", 256, state)
        elif split_router and tokens <= 256:
            logits = fused_split_router(ffn_input, state["router_hi_t"], state["router_lo_t"])
        elif split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = state["fp32_router_mod"].fp32_router_run(state["router"], router_input.contiguous(), 4 if tokens==1 else 1)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(expert_input, topk_weights, local_ids, state)
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

# Projection component from Nathen Fernandes, upstream commit
# 86aff75e6aca72511bf9b41b6dc190bf1e763373. Namespace-only adaptation.
# MIT License
# 
# Copyright (c) 2026 Nathen Fernandes
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
import base64 as _np__zsrc_b64, zlib as _np__zsrc_zlib

def _np__zsrc(payload: str) -> str:
    return _np__zsrc_zlib.decompress(_np__zsrc_b64.b64decode(payload)).decode()
"Dense projections: small_m (12-bit lossless or BF16 register-direct GEMV) for the shapes it claims, then the\nregister-direct swap-AB GEMV for small m, 12-bit lossless pooler weights at m = 1, and a TMA + wgmma router for\n64 < m <= 256. Shapes no kernel claims fall back to SGLang's own dispatch."
import hashlib as _np_hashlib
import zlib as _np_zlib
from typing import TypedDict as _np_TypedDict
import torch as _np_torch
import torch.nn.functional as _np_F
from torch.utils.cpp_extension import load_inline as _np_load_inline
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl as _np__bf16_gemm_dispatch_impl

class _np_Weights(_np_TypedDict):
    qkv_gate: _np_torch.Tensor
    o: _np_torch.Tensor
    pooler: _np_torch.Tensor
    unpooler: _np_torch.Tensor
    router: _np_torch.Tensor

class _np_Config(_np_TypedDict):
    m: int
    n: int
    k: int
    layer: str

class _np_Inputs(_np_TypedDict):
    x: _np_torch.Tensor
    config: _np_Config
_np_SM_CUDA_TEMPLATE = _np__zsrc('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')
_np_SM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_SM_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_np_SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_SM_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_np_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_SM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_SM_ESC_CAP = 32

def _np_sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _np_sm_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _np_SM_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _np_sm_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _np_load_inline(name, _np_SM_CPP + extra_cpp, cuda_sources=_np_sm_cuda_source(instances) + extra_cuda, functions=['sm_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_sm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np__to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np__to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np__to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_sm_c12_unpack(data, base, esc):
    """Inverse of sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_SM_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_sm_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_sm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_sm_units(hi), _np_sm_units(lo)]
    else:
        mats = [_np_sm_units(w)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_sm_c12_bytes(v)
        if not _np_torch.equal(_np_sm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_sm_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_sm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_sm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_sm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_SM_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_SM_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np__to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.sm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_SM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_np_SM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_np_SM_MAX_M = 32
_np_SM_TABLE_M.update({('router', 8): _np_SM_TABLE_MT['router', 1], ('qkv_gate', 8): _np_SM_TABLE_MT['qkv_gate', 1], ('pooler', 8): _np_SM_TABLE_MT['pooler', 1], ('o', 8): _np_SM_TABLE_MT['o', 1], ('unpooler', 1): _np_SM_TABLE_MT['unpooler', 1], ('unpooler', 8): _np_SM_TABLE_MT['unpooler', 1]})
_np_SM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_np_SM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_np_SM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_np_SM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _np_sm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_sm_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_sm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_sm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _np_sm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _np_sm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_sm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _np_sm_verify_onehot(ext, inst_id[inst], sp, _np_sm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _np_sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _np_prepare_small_m(weights, table_m=None, table_mt=None, name='small_m_gemv'):
    table_m = dict(_np_SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(_np_SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = _np_sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, _np_SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if _np_sm_bf16_twin(inst):
            instances.append(_np_sm_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = _np_sm_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = _np_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and _np_sm_bf16_twin(inst):
            plan = _np_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, _np_sm_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def _np_run_small_m(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= _np_SM_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _np_sm_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != _np_torch.bfloat16 or (x.shape[1] != _np_SM_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = _np_torch.empty((m, n), dtype=_np_torch.float32 if layer == 'router' else _np_torch.bfloat16, device=x.device)
    if _np_sm_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
_np_COMPRESSED_PAIRS = {('pooler', 1)}
_np_M32_CUDA_TEMPLATE = _np__zsrc('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')
_np_M32_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_M32_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_np_M32_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_M32_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_np_M32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_M32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_M32_ESC_CAP = 32

def _np_m32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _np_m32_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,))))
        cases.append(f'    case {i}: return m32_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _np_M32_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _np_m32_build(instances, name='m32_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _np_load_inline(name, _np_M32_CPP + extra_cpp, cuda_sources=_np_m32_cuda_source(instances) + extra_cuda, functions=['m32_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_m32_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_m32_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_m32_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_m32_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_m32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_m32_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_M32_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_m32_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_m32_c12_unpack(data, base, esc):
    """Inverse of m32_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_M32_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_m32_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_m32_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_m32_units(hi), _np_m32_units(lo)]
    else:
        mats = [_np_m32_units(w, layout)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_m32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_m32_c12_bytes(v)
        if not _np_torch.equal(_np_m32_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_m32_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_m32_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_m32_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_m32_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_M32_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_M32_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_m32_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_m32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_M32_TABLE_M = {}
_np_M32_MAX_M = 32
_np_M32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def _np_m32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_m32_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_m32_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_m32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_m32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_m32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_m32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_m32_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_m32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * key[2]
    rows = _np_m32_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = _np_m32_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_m32_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = _np_m32_verify_onehot(ext, inst_id[inst], sp, _np_m32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and _np_m32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _np_prepare_m32(weights, table_m=None, table_mt=None, name='m32_gemv'):
    table_m = dict(_np_M32_TABLE_M if table_m is None else table_m)
    table_mt = dict(_np_M32_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = _np_m32_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, _np_M32_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if _np_m32_bf16_twin(inst):
            instances.append(_np_m32_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = _np_m32_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = _np_m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and _np_m32_bf16_twin(inst):
            plan = _np_m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, _np_m32_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    st['outputs'] = {layer: _np_torch.empty((_np_M32_MAX_M, w.shape[0]), dtype=w.dtype, device=w.device) for layer, w in weights.items() if any((key[0] == layer for key in st['plans']))}
    return st

def _np_run_m32(x, layer, n, st):
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if not 9 <= m <= _np_M32_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _np_m32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != _np_torch.bfloat16 or (x.shape[1] != _np_M32_LAYER_K[layer]) or (not x.is_cuda):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = _np_torch.empty((m, n), dtype=st['outputs'][layer].dtype, device=x.device)
    if _np_m32_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out

def _np_mma(n):
    regs = ','.join((f'%{i}' for i in range(n // 2)))
    outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
    return f'__device__ void mma{n}(float* a,uint64_t x,uint64_t y) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(y));\n    }}'
_np_CUDA = _np__zsrc('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')
_np_CPP = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\nint launch_ss(const void*,const void*,void*,int,int,int,int,int,int,int,int,cudaStream_t);\nint launch(torch::Tensor x,torch::Tensor w,torch::Tensor y,int bn,int split,int stages,int hint,int plain) {\n  return launch_ss(x.data_ptr(),w.data_ptr(),y.data_ptr(),x.size(0),y.size(1),x.size(1),bn,split,stages,hint,plain,at::cuda::getCurrentCUDAStream());\n}\n'

def _np_wide2_build():
    source = _np_CUDA.replace('MMA_FUNCTIONS', '\n'.join((_np_mma(n) for n in [64, 128, 256])))
    return _np_load_inline('wide2_final_v1', _np_CPP, cuda_sources=source, functions=['launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])
_np_CANDIDATE_CONFIGS = {(128, 'qkv_gate'): (128, 3, 8, 1), (128, 'o'): (128, 1, 8, 1), (128, 'pooler'): (128, 2, 8, 1), (128, 'unpooler'): (128, 1, 9, 1), (256, 'qkv_gate'): (256, 3, 5, 1), (256, 'o'): (256, 1, 5, 1), (256, 'pooler'): (256, 2, 5, 1), (256, 'unpooler'): (256, 1, 5, 1)}

def _np_prepare_wide2(weights) -> dict:
    device = weights['qkv_gate'].device
    props = _np_torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0):
        return {'configs': {}}
    h200_keys = [(128, 'qkv_gate'), (128, 'pooler'), (256, 'pooler')]
    keys = h200_keys
    configs = {key: _np_CANDIDATE_CONFIGS[key] for key in keys}
    return {'ext': _np_wide2_build(), 'weights': {name: weight.contiguous() for name, weight in weights.items() if name != 'router'}, 'configs': configs}

def _np_run_wide2(x, layer, n, st):
    m = x.shape[0]
    cfg = st['configs'].get((m, layer))
    if cfg is None or m > 256 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
        return None
    weight = st['weights'].get(layer)
    if weight is None or weight.shape != (n, x.shape[1]) or weight.dtype != _np_torch.bfloat16:
        return None
    out = _np_torch.empty((m, n), dtype=_np_torch.bfloat16, device=x.device)
    rc = st['ext'].launch(x, weight, out, *cfg, 0)
    if rc:
        raise RuntimeError(f'wide2 launch failed: {rc}')
    return out

def _np__mma(n):
    regs = ', '.join((f'%{i}' for i in range(n // 2)))
    outs = ', '.join((f'"+f"(d[{i}])' for i in range(n // 2)))
    return f'__device__ __forceinline__ void rr_mma_{n}(float* d, uint64_t a, uint64_t b) {{\n    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));\n}}'
_np_ROUTER_CUDA = _np__zsrc('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')
_np_ROUTER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rr_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\nint rr_scalar(int64_t,int64_t,int64_t,int,int64_t,int);\nint rr_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

def _np_build_router(configs):
    src = _np_ROUTER_CUDA.replace('@@MMA@@', '\n'.join((_np__mma(n) for n in sorted({c[0] for c in configs}))))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join((f'if constexpr(BN=={n}) rr_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs}))))
    cases = '\n'.join((f"case {i}: return rr_launch_t<{','.join(map(str, c))}>(w,x,out,m,stream,query);" for i, c in enumerate(configs)))
    src = src.replace('@@CASES@@', cases).replace('@@SCALAR@@', _np_ROUTER_SCALAR + _np_ROUTER_SMALL)
    name = 'router_' + _np_hashlib.sha256(src.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_ROUTER_CPP, cuda_sources=src, functions=['rr_launch', 'rr_scalar', 'rr_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_pack_router(w, nr):
    hi = w.to(_np_torch.bfloat16)
    lo = (w - hi.float()).to(_np_torch.bfloat16)
    return _np_torch.stack([p.view(256 // nr, nr, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)

def _np_prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext': _np_build_router(configs), 'configs': configs, 'weight': w, 'packed': {nr: _np_pack_router(w, nr) for nr in {64 if c[3] else c[0] for c in configs}}}

def _np_run_candidate(x, st, index):
    out = _np_torch.empty((x.shape[0], 256), dtype=_np_torch.float32, device=x.device)
    stream = _np_torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rr_scalar(st['weight'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream, 0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rr_launch(index, w.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream, 0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out
_np_ROUTER_CONFIGS = [(32, 16, 4, 1, 2, 1), (32, 8, 4, 0, 2, 1), (64, 4, 8, 1, 2, 3)]

def _np_prepare_router(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256, 8192) or w.dtype != _np_torch.float32 or (not w.is_cuda):
        return {'enabled': False}
    if _np_torch.cuda.get_device_capability(w.device) != (9, 0):
        return {'enabled': False}
    st = _np_prepare_candidates(weights, _np_ROUTER_CONFIGS)
    st['small'] = {(16, 8): _np_pack_small_router(w, 16, 8)}
    st['enabled'] = True
    for m in (1, 8, 16, 32, 128, 256):
        x = _np_torch.zeros((m, 8192), dtype=_np_torch.bfloat16, device=w.device)
        _np_run_router(x, 'router', 256, st)
    _np_torch.cuda.synchronize(w.device)
    return st

def _np_run_router(x, layer, n, st) -> _np_torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return _np_run_candidate(x, st, -1)
    if m <= 16:
        return _np_run_small_candidate(x, st)
    return _np_run_candidate(x, st, 0 if m <= 32 else 1 if m <= 128 else 2)
_np_ROUTER_SCALAR = _np__zsrc('eNqdVm1v4zYM/p5focOAwo59PdtNi2tt+bABu2Ho2g9LvgWGodhyLzi/ZJIcOy3630fKL0matrtbPiQSRfIhH0pkJooXm5wpHqxLRe5sgj/3i+53vggncfyQVyuWxzHZVuuUCBHLhOVMxN+5KHluJFUpFcnyiqkpiWPBpRLrRIF+Y5PuMI7LbbzSKu7VC6XWnpD/+Lzme9dFWJjkCew1DG83Qgtv6Wf32vNBjju1Tqn6JjhL/0zb89YmomroKq+S7/0+ZyWnoHV24dqkYWKDmzC8RAcamrAkWd5Fy1lEn55H6Yw0zXK+iFBQA5BH2hb3oIkiVquK5CldnkUGhiHzStm4WHchEzRHYUQFBzkXG8FVnDCpggNKZ9PQaCwIeXprLtfaMyG/bAR7KBipS1HluRZlldAwijq+Cu58ZVkDDtCQGSoAqiBAjbh8G1RnApitpQ4Rn+Fbp34CPQBLAJbBfOFLRM5TQ9pAoyWn9wvz1JC4B6YrJnUBfFwEt59memHR+QKN+yzeyfkYeki6L9KWIs99GkOlWF+kNxy/QSdD3pBDJNDfc6ujNaQ1X5gQMCbQE3B8YP484lC4p/GNHD0ljzw6dHpSyWMdKOYZRn7ejgGcunF/3M3uwI0+9Ejm0Hiv6XVS49Ex7cx99cQ98IGPC66kE9GsYJmxhSeZOfA1yk913VF3h7o7e5Sf6nqj7iNEs/frvaJ7Meo2qDv6vdjrPu+fwwRplN+Y4Ck0pK5TyLqQ2CruF58uvOjdB/NOoTtfW2qMHFj7DC1jzMB6Gd87jySl7pWf+tDYqGuSrUUx9iyP06opY7krE8Nps/5jb+209wiB6f5IHbNLDuCwR0Z06+9JAPOuxUpD24GVbqRo9fRTLevljR+owCACzeqXIQwURTfOefZD7+p/5n/MwG6ppt7lFfbjnoDhLjxPJogyTkfEvJrFCobgsGrHVVV3o6AYJTDXONNb8k/Nxa4jQM8QHLP05dQNXPvS9Wwv7NnWRpR6A3FJnbKvdZn8qmBgrmrFJcnYWNHh9A+u9grGWcZs9G5+oKgwr5OES2kSaAi1KMlH1+kc9PuMnZd18Td/kFPXcRwLMzalSm9uinUZyPUjj1VogBaMWpbPYf/bDmDs6+trc7g6EEvxAesxYOihi0EEQQBE6yQdWwc81xTFiIGLMAyNo/8eZmMbr/3dMFvb6DWQ90KD93joF0j4C9rd70LATTHpYe5fnJuPn32o7b8VE84w')
_np_ROUTER_SMALL = _np__zsrc('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')

def _np_pack_small_router(w, s=16, wk=8):
    hi = w.to(_np_torch.bfloat16)
    lo = (w - hi.float()).to(_np_torch.bfloat16)

    def units(v):
        v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        return v.permute(0, 1, 3, 2, 4)
    data = _np_torch.cat([units(hi), units(lo)], dim=2).contiguous()
    ku = 128 // s // wk
    return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

def _np_run_small_candidate(x, st, s=16, wk=8):
    out = _np_torch.empty((x.shape[0], 256), dtype=_np_torch.float32, device=x.device)
    rc = st['ext'].rr_small(st['small'][s, wk].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], s, wk, _np_torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'router small launch failed: {rc}')
    return out
_np_V9_SELECTED_ROUTES = []

def _np_qkv_mma(n):
    regs = ','.join((f'%{i}' for i in range(n // 2)))
    outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
    return f'__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));\n    }}'
_np_QKV_CUDA = _np__zsrc('eNq9Gv1z2rjy9/wVam7IyNglQCiTw7j5aLlMp4TkgGvvXSbjMbYIToxNbPGVPP73tyvZxgaStne9dzcBsVqtVvu9Uvd+cX3bmzqMNO2pY5VG7/fyEDOc+twdsx0zg2GlngdH3HF9nodZ3l0Qunw0BqDDhq7PyO8fiWk6bObazDRhOAxCm7m+B3OmuQezU6ByVDU5eTQtxwmpHfgRJ7PAdYpkopDnkPFp6BOa4Cmmac+4Zd4xn4WubfLAjEZWyBw6UfQVUsS1QM31XZ6uIhMNRsQHglY0BhTP4q7H6P54YIWhy8ISopckpUbD5lZpUK+Rm0L5VitU9P1GYz/chx00/PKVxv6YjYNwuZ/fki0mzP6RTfFrxkpynckXWwyY2g+yMLe2Tz0ZWREDJvYIyfPxXArZHSlNYEsy0Um7QVLOeLgUtEoTCzS6LFn249QNWQk0HXGY32R0kjJKTt9MyCC0SFtf5ZmWfGQY38tyHi19m26JKuEn2TeWWMg8BsRKlufe+czRySae4D1hOsECdjbFljG/sTXZlFxo+Q/AUgp1dOAO9GdNrLUE4h2nR1VSKIMEtEIVdto34MSOkj2/oKbosUU7Ob1NptEou/vQCyxeIzMthQ12aTCCQ6LgtrhJFAmQYDzxGGdgXo3GYMlZVJrVSkPgVmjsWTCsFY60Qm0FSnx3mzW14T6dlRbJYJkMnpLBXJ5ssK1V5LteE4J1WGRnzrZ26gSH0slBeXE0hP+U9+9ryn9pOlOvKc3mUTULqgCkXs3oL94GQxeNtSVAXN9wu2BWmoKtgpZIoXDnBQPLw0WhUJe3T7mStY+YSZ7TEx9bWTVloxUfC5t5kJaTUZyWcjQJPNde7lKkPYkVOZh6DyXO/CgIS1VnS6+S69fV264C2LJHzBwhK0LNN6DmZ6HnFf6ogau+yyoaTz+WynyIrTVRrZiMOc+r+fT08vLs9HSPM+DB4qyJ23W6QgAX4rN/Kb568rMvvq4/XbfEoNv6qA2CwCPX3avfPrVb7/dMU55PpArPmvr2yBwEU9+JTJNSXHhQVU6q7+qNSvVYSXTywEKfeXHmABKh65jih+VzoPThDynOS2tC+FwD0RPyXbgLzTT9GeQ+9MVKvUiCKReMj7WpH4mQQrzAvxMfRQI0xpNIKleQhLAeEsT/3Dck80fIfaOikXbXoNWjck29eFtRDi808vXc6HSLcCqN/Hlu9C/lsHdufO4X6ddz9c9zkPgm3c4XXNS/PKxp5OprJ96kUldOaOeL2kPSvQbFMVArit9UjBQF6W8T7J4bvX6xd67CZ6WugYY+fDFATYZxdFJudM/VSh0XsQWYog/ik9YptCViLGipUq7WFJLKxwYMEo1vbtPNxEbcdQw+CpnlfHIWpYVGPMtnBkAPjkA6GCiNgRfYD2K20NPIXRhMJxnYYW+DYJBZsQSZAJVgbohlRZQ0ChkgkUEFTK0oMfgtQPO0HgYGiunwc18pIiuHsL+fBYpIrqJ03z4MNBLMfWPs+hRUoHW+vMXJIozXCltnmQEkLSOudCLwN8xbkYFQVQheI+BKfGkgGCFgA+FAznfPkV4aSgbszvUNiGQRBzGOTeY7Rhkx3OFaqTT2LCVGT4LkFl672+x0pekSAjUaRTm4qBDdbQIftNN92+4qxaOq7qpS0B/dMWSGeAkRggMPuGOGe0jX6HDEe8MtZCF6vIKmgbSogDBUsRqNry0cQVFu7m/lkQhZic98yBwy32ZQvASL5WYS5NZGrhfLTVFjSLOLpBBWWZFO5tpkoW+HZsDnTAbA0jC0bO4GPsRgDLNY2HJz6IaQiAdxaqmUyklCmcwV5e8Q9KwX6C1ielMeQNSxHOPm4FbqKtHE2oxjbRR6fXn8tD4VxiWlfaxhfJECxxgTSypnGxC9jKoiU5+wxERR2gGE07ImHUpQoA8D1VWK6COQBnL7gCgkbeZFjLxELSZQr2mSaHsXlV8moXU3tiDIhIHn5Uz2EQzmsfm5rz+q6i6WVTguRNPH4p+430KjKcfqo9gW4sjGjgtpKNJlKAYuY52K4NCNsoKyf4UpF5hCLwLXUQE1bk3ELi7sUFH0vM9K2jGaiAcCr4Ylz5Z2eu8rStLthAMk9jNcZXuL59R8YBeKuaQIcQ+yhKLv2ispUARjabkeVzFbhfh3Cw804DYhFito/NRNfXiHa28egRwcYMXxxjhSyPf2GNaCOS90D1uRVmot9kJpKe8NUaakITI2HwFcnw/PhmdKzUP0cFLvFN1XAeVT6h7CCMR+ACpOTr9KKW+pK57ZmJO5/PsFkGuydtjJtqn9jUbsRVbfQNxJO8MELTmzLM/X6QH/RKVGrBsKNRF0Dpi2DqvK7Y6QkYh/BOY1agp0fQTyf80U7wH3vokk9XtU1XZ3MZA9IJEBGzok62YUM6He3ypbsYtU9jYtfW0I3wznwk5ywUoaCVrIa+f4njN/76nhnOrWObeUm18zvxuPrZKMEyIYvWAWrxzhAZh6aNYgbusPa2HtcsnjrC+kqd5ZGHFfuis90IdDMJ4/cVCoySrGme9eIFG/ZlDX5nx62vt6dg39UWy4MvU97xb3y+rJ8P/jJxgVjyu/Vv/hQfAoH87a7fQoay9cvVieSS1Dezp2uSmy+avK3q29irKTKlr+DpqksouqkPqPUCn/gCH+/30J6TgBdEvuW5oI6aQCVUgiQ4qz740yJjwcQh8hUuYrToJ5CxaKJswoK6/fVu6+o0wuE+K8hRvL1JW/qZQcCJUk6bAsk3nMqpKP6P9Iby81Q5mWKdsRiaqagbmmVfUsuYd7ShM79prGrCDa83uDzg4xmxdQX8fQ64wSyKGErFUWhsYICkyV8kPwLaibYFSAEfbutm3cF48RAJ5XlWtkKpvNbmq3xvMTtMhPpSX8PcHffPWqSdpgZHazptu5wIh8h9rc1l+NlDw0oAJW40OseVKpjVllbhvb/MMc2GrxOK/dhFQYpvOw2rZjSqtMTRQ2x2irc7uJDboSTPkND0V3rgJAndu3hmkKeVSTuxgz9CnIxr7Nmst21Qe11jdq87/pv8fCf9FWKJXeegzeKm5VVDBqbWw9MMlyLefPxdqtlv+tVrYg1S3I0a3szNIU8q8Xdv+8qPu3hJ4tbdFFjZ0KcCLw0kO8kfEC2/LAY2G8zmiJWxs/X1F6rtqQ98TxDU7rwxeVJtdDqmAMPSlbAFNk3BAvBUUq+ZK3Iy8tVoyn9fq4txaPCfJBY4KSULQnTf6E3g1/bxTTOxPClgWIYEsqGsGrw7yyxb47erC0SoV9kwSVqJWL2yXehC5S52q+U4rVE03HOQWVNfH/buNaUw7BYMJmTw/z1pLqfKdghVS5kOhansBAaaEaEH51HC5xuBTDJxw+ieEch/PN9kTEhlRdXANMJXuX9c1eZ/Vi+spk6vh46XVrNHadXa8ecV9SKAiE+HUKx0omRblQU1KauUXF23FxzaeuL14VsHJQnEgCNblWXnzfwPJbQ9BfA8BpDHH1mIVVAS3NwNmJo9tcPl7trfamkevfkd9bvh04zHCY7fHlhNEDe9pPLurlXB8O68AycQtkylt8YfKZK/3kjUbe9omrfNf3WShfbCD3iMEgiK81gTPu2snmhEke/KnnTXgY3we9kVDlGZ/JP2J0DVs+D5fXAZD6fcrCZZdFUw940hHjgvFNpPPlFxZGbuDT/d2H2teoeGAqKgdyM61SLZfLGtJr+dbAYx/Z0II9tINHvEpKeYofsN5WymU9d9HpuOPoBrTwnD6qKVIO698ojZUGanIddlPZRi1WV/r6VlsQq9e09Ss9CHGlMblLRauI3BzzAyhUckhBHR/+MPutTu+qa16eXZsfz/pnZv8/1y3z/Lf21Vm/Uteq8fEVCGXAd8yTNgDqG4s/dfqtbrt19qVldq46LfnSk0fpff3011/tlgnR5nxjdbtqgoddXvU/XXXwR/VdfRNFsGReXZ2b4IhtsYkiHsJ+2gOYtN745YvTRGHz1GoX6Wj9GJVApFMJ4CNaXmbC4jsfpjafoTIPSj/lPWnz4WsbQ7xY4qPTZ/HIoPYvZQ4XD1AqVpm5q7uTHoZTmIr7HVGyD/FhQz4CQpmgXWgg9J4GwhaCRiEn8tXXXi3kjqlqaQwtSF+JPwuQdOdWGEIM4YQZ+Os30EmP8TMO9jeYckaHvpbAU+Cltfi49K2xa/dEl3QpMmTPfQKDxZMK/2RvBMHe1LZZFKVuitpWWNyzfv+WncC/DkKOceCDLM9wuzPPC+bMkZfL39hRSoGHU6bLpEOF+eCt//PWdhEZWnoCvcgwF9GDoaUN/fThfGiV/Om4y+6iIkSgsgq/RdWC7J2jMMRuSKot7P1D4A/dO5C3Pbwznlc6fJXiDGSA5x/Ri2JPo2MwETQ8MBSBkbxFSZStN2KB48QqgXpFbGwIXYgp6TMGFeKRSYkrEqjnuEvPSSwYIX/4XYIGdgdOrAhgC+I5hHaJO7O8pIJGhp97moiMIG+8sn1GdhAvMg7wS7AHAjwToMqGagBfVr9lPW+qQOoEIVe2PZ1Yvr0EkzyzOeScmCnQE7S1vnYA9JXGbtRzFGp0zcJLSCvuJAzQbKC6kiuzYofGNDbsWOtQlmRM7WTWSKwM+d94iucLHQ+BL6KZhI3vQHNNvDqd1GsNvL+Kf11gaCh2uo0LfBiSMIgbAGh3pZ0nps30LaILbaEJWmMNbSePnc9Oa41+FlGltaAoLTy74Fqj+X8YgC0rkKU7/lWAImOvyBLZ4C5vfB3t50f4aO5ypO8oz3hr12v1Tk+JIyuERuL5v/6qY3X1P1TFe10=')

def _np_build_qkv(configs, profile=False):
    ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
    source = _np_QKV_CUDA.replace('@@MMA@@', '\n'.join((_np_qkv_mma(n) for n in ns)))
    source = source.replace('@@CALL@@', ' '.join((f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);' for n in ns)))
    source = source.replace('@@SWAP@@', ' '.join((f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns)))
    source = source.replace('@@CASES@@', '\n'.join((f"case {i}: return q_launch_t<{','.join(map(str, c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);" for i, c in enumerate(configs))))
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint q_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
    return _np_load_inline('qkv_' + _np_hashlib.sha256(source.encode()).hexdigest()[:16], cpp, cuda_sources=source, functions=['q_launch'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def _np_qkv_pack(w, groups, kt=1):
    rows = (2304 + groups - 1) // groups
    if kt == 2:
        return w.view(groups, rows, 128, 64).permute(0, 2, 1, 3).contiguous().reshape(-1, 64)
    if 2304 % groups == 0:
        return w.contiguous()
    out = _np_torch.zeros((groups, rows, 8192), dtype=_np_torch.bfloat16, device=w.device)
    for g in range(groups):
        lo, hi = (g * 2304 // groups, (g + 1) * 2304 // groups)
        out[g, :hi - lo].copy_(w[lo:hi])
    return out.reshape(-1, 8192)

def _np_prepare_qkv_candidates(weights, configs, profile=False):
    w = weights['qkv_gate']
    return {'ext': _np_build_qkv(configs, profile), 'configs': list(configs), 'packed': {(g, kt): _np_qkv_pack(w, g, kt) for g, kt in {(c[1], 2 if c[5] & 32 else 1) for c in configs}}, 'device': w.device}

def _np_run_qkv_candidate(x, st, index, stats=None, query=0):
    out = _np_torch.empty((x.shape[0], 2304), dtype=_np_torch.bfloat16, device=x.device)
    cfg = st['configs'][index]
    rc = st['ext'].q_launch(index, st['packed'][cfg[1], 2 if cfg[5] & 32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], _np_torch.cuda.current_stream().cuda_stream, query, stats.data_ptr() if stats is not None else 0)
    if query:
        return rc
    if rc:
        raise RuntimeError(f'qkv launch {cfg} failed: {rc}')
    return out
_np_QKV_SM_CUDA_TEMPLATE = _np__zsrc('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')
_np_QKV_SM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint qkv_sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_QKV_SM_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_np_QKV_SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_QKV_SM_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_np_QKV_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_QKV_SM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_QKV_SM_ESC_CAP = 32

def _np_qkv_sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _np_qkv_sm_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,))))
        cases.append(f'    case {i}: return qkv_sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _np_QKV_SM_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _np_qkv_sm_build(instances, name='qkv_sm_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _np_load_inline(name, _np_QKV_SM_CPP + extra_cpp, cuda_sources=_np_qkv_sm_cuda_source(instances) + extra_cuda, functions=['qkv_sm_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_qkv_sm_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_qkv_sm_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_qkv_sm_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_qkv_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_qkv_sm_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_qkv_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_QKV_SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_qkv_sm_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_qkv_sm_c12_unpack(data, base, esc):
    """Inverse of qkv_sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_QKV_SM_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_qkv_sm_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_qkv_sm_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_qkv_sm_units(hi), _np_qkv_sm_units(lo)]
    else:
        mats = [_np_qkv_sm_units(w, layout)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_qkv_sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_qkv_sm_c12_bytes(v)
        if not _np_torch.equal(_np_qkv_sm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_qkv_sm_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_qkv_sm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_qkv_sm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_qkv_sm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_QKV_SM_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_QKV_SM_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_qkv_sm_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.qkv_sm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_QKV_SM_TABLE_M = {}
_np_QKV_SM_MAX_M = 32
_np_QKV_SM_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def _np_qkv_sm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_qkv_sm_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_qkv_sm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_qkv_sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_qkv_sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_qkv_sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_qkv_sm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_qkv_sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * key[2]
    rows = _np_qkv_sm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = _np_qkv_sm_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_qkv_sm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = _np_qkv_sm_verify_onehot(ext, inst_id[inst], sp, _np_qkv_sm_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and _np_qkv_sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _np_prepare_qkv_sm(weights, table_m=None, table_mt=None, name='qkv_sm_gemv'):
    table_m = dict(_np_QKV_SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(_np_QKV_SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = _np_qkv_sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, _np_QKV_SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if _np_qkv_sm_bf16_twin(inst):
            instances.append(_np_qkv_sm_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = _np_qkv_sm_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = _np_qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and _np_qkv_sm_bf16_twin(inst):
            plan = _np_qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, _np_qkv_sm_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def _np_run_qkv_sm(x, layer, n, st):
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if not 9 <= m <= _np_QKV_SM_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _np_qkv_sm_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != _np_torch.bfloat16 or (x.shape[1] != _np_QKV_SM_LAYER_K[layer]) or (not x.is_cuda):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = _np_torch.empty((m, n), dtype=_np_torch.bfloat16, device=x.device)
    if _np_qkv_sm_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
_np_QKV_LARGE_CONFIGS = [(64, 36, 128, 3, 8, 27, 2), (64, 36, 128, 3, 8, 25, 2), (64, 36, 128, 3, 4, 59, 2)]
_np_QKV_SMALL_INST = (1, 2, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 8, 8, 2, 7, 1)
_np_QKV_SMALL_GEOMETRY = (1, 8, 8, 2)

def _np_prepare_qkv(weights) -> dict:
    w = weights.get('qkv_gate')
    if w is None or w.shape != (2304, 8192) or w.dtype != _np_torch.bfloat16 or (not w.is_cuda):
        return {'is_enabled': False}
    if _np_torch.cuda.get_device_capability(w.device) != (9, 0):
        return {'is_enabled': False}
    small = _np_prepare_qkv_sm({'qkv_gate': w}, table_m={}, table_mt={('qkv_gate', 2): (_np_QKV_SMALL_INST, _np_QKV_SMALL_GEOMETRY)}, name='qkv_small_final')
    large = _np_prepare_qkv_candidates(weights, _np_QKV_LARGE_CONFIGS)
    warm = _np_torch.zeros((128, 8192), dtype=_np_torch.bfloat16, device=w.device)
    for index in range(len(_np_QKV_LARGE_CONFIGS)):
        _np_run_qkv_candidate(warm, large, index)
    _np_torch.cuda.synchronize(w.device)
    return {'is_enabled': 'H100' in _np_torch.cuda.get_device_name(w.device), 'device': w.device, 'small': small, 'large': large, 'large_index': 0}

def _np_run_qkv(x, layer, n, st) -> _np_torch.Tensor | None:
    if not st.get('is_enabled') or layer != 'qkv_gate' or n != 2304:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda):
        return None
    if x.device != st['device'] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if 9 <= m <= 16:
        return _np_run_qkv_sm(x, layer, n, st['small'])
    if m == 128:
        return _np_run_qkv_candidate(x, st['large'], st['large_index'])
    return None

def _np_rt2_mma(n):
    regs = ', '.join((f'%{i}' for i in range(n // 2)))
    outs = ', '.join((f'"+f"(d[{i}])' for i in range(n // 2)))
    return f'__device__ __forceinline__ void rt2_mma_{n}(float* d, uint64_t a, uint64_t b) {{\n    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));\n}}'
_np_rt2_ROUTER_CUDA = _np__zsrc('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')
_np_rt2_ROUTER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rt2_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\nint rt2_scalar(int64_t,int64_t,int64_t,int,int64_t,int);\nint rt2_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

def _np_rt2_build_router(configs):
    src = _np_rt2_ROUTER_CUDA.replace('@@MMA@@', '\n'.join((_np_rt2_mma(n) for n in sorted({c[0] for c in configs}))))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join((f'if constexpr(BN=={n}) rt2_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs}))))
    cases = '\n'.join((f"case {i}: return rt2_launch_t<{','.join(map(str, c))}>(w,x,out,m,stream,query);" for i, c in enumerate(configs)))
    src = src.replace('@@CASES@@', cases).replace('@@SCALAR@@', _np_rt2_ROUTER_SCALAR + _np_rt2_ROUTER_SMALL)
    name = 'router2_' + _np_hashlib.sha256(src.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_rt2_ROUTER_CPP, cuda_sources=src, functions=['rt2_launch', 'rt2_scalar', 'rt2_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_rt2_pack_router(w, nr):
    hi = w.to(_np_torch.bfloat16)
    lo = (w - hi.float()).to(_np_torch.bfloat16)
    return _np_torch.stack([p.view(256 // nr, nr, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)

def _np_rt2_prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext': _np_rt2_build_router(configs), 'configs': configs, 'weight': w, 'packed': {nr: _np_rt2_pack_router(w, nr) for nr in {64 if c[3] else c[0] for c in configs}}}

def _np_rt2_run_candidate(x, st, index):
    out = _np_torch.empty((x.shape[0], 256), dtype=_np_torch.float32, device=x.device)
    stream = _np_torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rt2_scalar(st['weight'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream, 0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rt2_launch(index, w.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream, 0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out
_np_rt2_ROUTER_CONFIGS = [(32, 16, 4, 1, 2, 1, 256, 256), (32, 8, 4, 0, 2, 1, 256, 256), (64, 8, 6, 1, 2, 1, 64, 64), (64, 4, 8, 1, 2, 3, 256, 256)]

def _np_prepare_router2(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256, 8192) or w.dtype != _np_torch.float32 or (not w.is_cuda):
        return {'enabled': False}
    if _np_torch.cuda.get_device_capability(w.device) != (9, 0):
        return {'enabled': False}
    st = _np_rt2_prepare_candidates(weights, _np_rt2_ROUTER_CONFIGS)
    st['small'] = {(16, 8): _np_rt2_pack_small_router(w, 16, 8)}
    st['enabled'] = True
    for m in (1, 8, 16, 32, 64, 128, 256):
        x = _np_torch.zeros((m, 8192), dtype=_np_torch.bfloat16, device=w.device)
        _np_run_router2(x, 'router', 256, st)
    _np_torch.cuda.synchronize(w.device)
    return st

def _np_run_router2(x, layer, n, st) -> _np_torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return _np_rt2_run_candidate(x, st, -1)
    if m <= 16:
        return _np_rt2_run_small_candidate(x, st)
    return _np_rt2_run_candidate(x, st, 0 if m <= 32 else 1 if m <= 64 else 2 if m <= 128 else 3)
_np_rt2_ROUTER_SCALAR = _np__zsrc('eNqdVm1v4zYM/p5focOAwo59PdtNi2tt5bABu2Ho2g9LvgWGodhyLzi/ZJIcOy3630fKb0nTdnfLh0SiSD7kQ4nMRPF8mzHFg02hyJ1N8Od+2f4ulvNJFD1k5ZplUUR25SYhQnmRjFnGRPSdi4JnRlwWUpE0K5makigSXCqxiRUY1DZpD6Oo2EVrreJevVBq7An5j89rvvdtiLlJnsBew/BmK7Twln52rz0f5LhTm4Sqb4Kz5M+kOW9sIsqarrMy/t7tM1ZwClpnF65Naia2uJnPL9GBhiYsjld34WoW0qfnQTojdb1aLEMUVADkkabBPWiiiFWqJFlCV2ehgWHIrFQ2LjZtyATNURhSwUHOxVZwFcVMquCA0tl0btQWhDy9NVcb7ZmQX7aCPeSMVIUos0yL0lJoGEUdXwV3vrKsHgdoSA0VAFUQoEZcvQ2qMwHMxlKHiM/wrVM/ge6BJQDLYLH0JSJniSFtoNGS0/uleWpI3APTNZO6AD4ugttPM72w6GKJxl0W7+R8DN0n3RVpR5HnLo2+Uqwr0huO36CTIW/IIRLoj9zqaA1pLZYmBIwJdAQcH5g/j9gX7ml4I0dPySOPDp2eVPJYB4p5hpGfN0MAp27cH3ezP3CjDz2SOjQaNb1Wajw6pp26r564Bz7wccGVdEKa5iw1dvAkUwe+Bvmprjvo7lF3bw/yU11v0H2EaEa/3iu6F4NujbqD34tR93l8DhOkUX5jgifQkNpOIatcYqu4X3668MJ3H8w7hW597agxcGCNGVrGkIH1Mr53HklC3Ss/8aGxUdckO4ti7GkWJWVdRHJfxIbTpN3H3tlJ5xEC0/2ROmabHMBhjwzpzh9JAPO2xUpD24GVbqRo9fRTLevlje+pwCACzeqXPgwUhTfOefpD7+p/5n/MwH6lpt7lFfbjjoD+LjxPJogyjkcEvZpFCqZgv2qGVVm1syAfJDDYONNb8k/Fxb5lQA8RnLP0ZO4Grn3perY37/jWVpR6PXVxlbCvVRH/qmBkrivFJUnZUNP+9A+uRgXjLGU2ejc/UFRYVHHMpTQJtIRKFOSj67QOun3Kzosq/5s/yKnrOI6FKZtSJTc3+aYI5OaRR2pugBYMW5YtYP/bHmDs6+trs788EEv+ASvSY+ixi0EEQQBU6yQdWwe80BxFiIGL+XxuHP37MGvbeO0Ph9nYRqeBxOcavMNDv0DCX9DwfhcC7opJD3P/4tx8/OxDdf8FLxbOzA==')
_np_rt2_ROUTER_SMALL = _np__zsrc('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')

def _np_rt2_pack_small_router(w, s=16, wk=8):
    hi = w.to(_np_torch.bfloat16)
    lo = (w - hi.float()).to(_np_torch.bfloat16)

    def units(v):
        v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        return v.permute(0, 1, 3, 2, 4)
    data = _np_torch.cat([units(hi), units(lo)], dim=2).contiguous()
    ku = 128 // s // wk
    return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

def _np_rt2_run_small_candidate(x, st, s=16, wk=8):
    out = _np_torch.empty((x.shape[0], 256), dtype=_np_torch.float32, device=x.device)
    rc = st['ext'].rt2_small(st['small'][s, wk].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], s, wk, _np_torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'router small launch failed: {rc}')
    return out
_np_OLAYER_CUDA = _np__zsrc('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')
_np_OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def _np_olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return ol_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns))
    return _np_OLAYER_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def _np_build_olayer(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (0, 1, 2, 3)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = _np_olayer_source(configs)
    name = 'olayer_' + _np_hashlib.sha256(source.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_OLAYER_CPP, cuda_sources=source, functions=['ol_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

def _np_pack_olayer(weight) -> _np_torch.Tensor:
    columns = _np_torch.arange(64, device=weight.device)[None, :] ^ (_np_torch.arange(64, device=weight.device)[:, None] & 7) * 8
    tiled = weight.reshape(128, 64, 16, 64).permute(0, 2, 1, 3)
    return tiled.gather(-1, columns.expand(128, 16, 64, 64)).contiguous()
_np_OLAYER_PRIMARY_CONFIGS = [(16, 256, 3, 3, 1, 1), (32, 128, 8, 1, 1, 1), (64, 128, 4, 3, 1, 1), (128, 128, 4, 3, 1, 1), (128, 64, 5, 3, 1, 1, 2)]
_np_OLAYER_NEARBY_CONFIGS = {bn: (bn, 128, 4, 3, 1, 1) if bn <= 128 else (bn, 64, min(8, 227000 // ((64 + bn) * 128 + 16)), 3, 1, 1) for bn in range(72, 249, 8)}
_np_OLAYER_CONFIGS = list(dict.fromkeys(_np_OLAYER_PRIMARY_CONFIGS + list(_np_OLAYER_NEARBY_CONFIGS.values())))

def _np_prepare_olayer_candidates(weights, configs) -> dict:
    weight = weights['o'].contiguous()
    return {'enabled': True, 'ext': _np_build_olayer(configs), 'configs': configs, 'weight': weight, 'packed': _np_pack_olayer(weight) if any((c[4] == 1 for c in configs)) else None, 'tiled32': weight.reshape(256, 32, 16, 64).permute(0, 2, 1, 3).contiguous() if any((c[4] == 2 for c in configs)) else None}

def _np_run_olayer_candidate(x, st, index) -> _np_torch.Tensor:
    pack = st['configs'][index][4]
    weight = st['tiled32'] if pack == 2 else st['packed'] if pack == 1 else st['weight']
    out = _np_torch.empty((x.shape[0], 8192), device=x.device, dtype=_np_torch.bfloat16)
    rc = st['ext'].ol_launch(index, x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.shape[0], _np_torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f"olayer launch failed: {rc}, config={st['configs'][index]}")
    return out

def _np_prepare_olayer(weights) -> dict:
    weight = weights.get('o')
    if weight is None or weight.shape != (8192, 1024) or weight.dtype != _np_torch.bfloat16 or (not weight.is_cuda):
        return {'enabled': False}
    if _np_torch.cuda.get_device_capability(weight.device) != (9, 0):
        return {'enabled': False}
    st = _np_prepare_olayer_candidates(weights, _np_OLAYER_CONFIGS)
    st['token_configs'] = {bn: _np_OLAYER_CONFIGS.index(cfg) for bn, cfg in _np_OLAYER_NEARBY_CONFIGS.items()}
    for index, cfg in enumerate(_np_OLAYER_CONFIGS):
        m = cfg[0] * (cfg[6] if len(cfg) > 6 else 1)
        _np_run_olayer_candidate(_np_torch.zeros((m, 1024), dtype=_np_torch.bfloat16, device=weight.device), st, index)
    _np_torch.cuda.synchronize(weight.device)
    return st

def _np_run_olayer(x, layer, n, st) -> _np_torch.Tensor | None:
    if not st.get('enabled') or layer != 'o' or n != 8192:
        return None
    if x.ndim != 2 or x.shape[1] != 1024 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16 or (not 1 <= x.shape[0] <= 256):
        return None
    m = x.shape[0]
    if 65 <= m <= 224:
        index = st['token_configs'][(m + 7) // 8 * 8]
        return _np_run_olayer_candidate(x, st, index)
    return None
_np_CPS32_CUDA_TEMPLATE = _np__zsrc('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')
_np_CPS32_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint cps32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_CPS32_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_np_CPS32_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_CPS32_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_np_CPS32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_CPS32_ESC_CAP = 32

def _np_cps32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _np_cps32_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_cps32_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_cps32_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_cps32_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_cps32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_cps32_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_CPS32_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_cps32_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_cps32_c12_unpack(data, base, esc):
    """Inverse of cps32_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_CPS32_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_cps32_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_cps32_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_cps32_units(hi), _np_cps32_units(lo)]
    else:
        mats = [_np_cps32_units(w, layout)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_cps32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_cps32_c12_bytes(v)
        if not _np_torch.equal(_np_cps32_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_cps32_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_cps32_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_cps32_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_cps32_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_CPS32_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_CPS32_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_cps32_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.cps32_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_CPS32_TABLE_M = {}
_np_CPS32_MAX_M = 32
_np_CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def _np_cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_cps32_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_cps32_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_cps32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_cps32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_cps32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_cps32_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_cps32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * key[2]
    rows = _np_cps32_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = _np_cps32_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_cps32_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = _np_cps32_verify_onehot(ext, inst_id[inst], sp, _np_cps32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and _np_cps32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
_np_CPSSM_CUDA_TEMPLATE = _np__zsrc('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')
_np_CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_CPSSM_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_np_CPSSM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_CPSSM_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_np_CPSSM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_CPSSM_ESC_CAP = 32

def _np_cpssm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _np_cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_CPSSM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_cpstage___to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_cpssm_c12_unpack(data, base, esc):
    """Inverse of cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_CPSSM_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_cpssm_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_cpssm_units(hi), _np_cpssm_units(lo)]
    else:
        mats = [_np_cpssm_units(w)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_cpssm_c12_bytes(v)
        if not _np_torch.equal(_np_cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_cpssm_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_CPSSM_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_CPSSM_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_cpstage___to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_np_CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_np_CPSSM_MAX_M = 32
_np_CPSSM_TABLE_M.update({('router', 8): _np_CPSSM_TABLE_MT['router', 1], ('qkv_gate', 8): _np_CPSSM_TABLE_MT['qkv_gate', 1], ('pooler', 8): _np_CPSSM_TABLE_MT['pooler', 1], ('o', 8): _np_CPSSM_TABLE_MT['o', 1], ('unpooler', 1): _np_CPSSM_TABLE_MT['unpooler', 1], ('unpooler', 8): _np_CPSSM_TABLE_MT['unpooler', 1]})
_np_CPSSM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_np_CPSSM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_np_CPSSM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_np_CPSSM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _np_cpssm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_cpssm_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _np_cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _np_cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _np_cpssm_verify_onehot(ext, inst_id[inst], sp, _np_cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _np_cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _np_cpstage___to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_cpstage___to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)
_np_CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def _np_cps_build(family, instances, depth):
    is32 = family == 'm32'
    prefix = 'cps32_' if is32 else 'cpssm_'
    template = _np_CPS32_CUDA_TEMPLATE if is32 else _np_CPSSM_CUDA_TEMPLATE
    cpp = _np_CPS32_CPP if is32 else _np_CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == (16 if is32 else 14) else inst + (0, 1) if is32 and len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + _np_hashlib.sha256(source.encode()).hexdigest()[:12]
    return _np_load_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_prepare_cps(weights, depth=None, layers=_np_CPS_LAYERS, buckets=(2, 4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
        for bucket, selected_depth in ((2, 2), (4, 1)):
            if bucket in buckets and selected_layers:
                part = _np_prepare_cps(weights, selected_depth, selected_layers, (bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0, 1, 2, 3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm', 'm32'):
        table = _np_CPSSM_TABLE_MT if family == 'sm' else _np_CPS32_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and (layer in _np_CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and ((layer, bucket) in _np_CPS32_TABLE_MT):
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0, rows, _np_CPSSM_LAYER_K[layer], *cfg)
            entries.append(((layer, 'mt', bucket), (inst, cfg)))
        if include_policy and family == 'sm':
            for (layer, m), (inst, cfg) in _np_CPSSM_TABLE_M.items():
                if layer not in _np_CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m, rows, _np_CPSSM_LAYER_K[layer], *cfg)
                entries.append(((layer, 'm', m), (inst, cfg)))
        if not entries:
            continue
        twin = _np_cpssm_bf16_twin if family == 'sm' else _np_cps32_bf16_twin
        instances = list(dict.fromkeys((inst for _, (inst, _) in entries for inst in (inst, twin(inst)) if inst is not None)))
        ext = _np_cps_build(family, instances, depth)
        ids = {inst: i for i, inst in enumerate(instances)}
        plan_fn = _np_cpssm_plan if family == 'sm' else _np_cps32_plan
        packs, streams = ({}, {})
        for key, (inst, cfg) in entries:
            sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
            if sp is not None:
                st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
    return st

def _np_cps_launch(sp, x, out, n, policy=0, query=0):
    fn = _np_cps32_launch_raw if sp['family'] == 'm32' else _np_cpssm_launch_raw
    return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

def _np_run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _np_cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != _np_torch.bfloat16 or (x.shape[1] != _np_CPS32_LAYER_K[layer]) or (not x.is_cuda):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = _np_torch.empty((m, n), dtype=_np_torch.float32 if layer == 'router' else _np_torch.bfloat16, device=x.device)
    rc = _np_cps_launch(sp, x, out, n, st.get('policy', 0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out

def _np_cps_resources(st, layer, m, policy=0):
    sp = st['plans'].get((layer, 'm', m)) or st['plans'][layer, 'mt', _np_cps32_mt(m)]
    x = _np_torch.empty((m, _np_CPS32_LAYER_K[layer]), dtype=_np_torch.bfloat16, device=sp['wp'].device)
    out = _np_torch.empty((m, sp['rows']), dtype=_np_torch.float32 if layer == 'router' else _np_torch.bfloat16, device=x.device)
    attrs = _np_cps_launch(sp, x, out, sp['rows'], policy, 2)
    return {'registers': attrs // 1000, 'local_bytes': attrs % 1000, 'active_clusters_or_blocks': _np_cps_launch(sp, x, out, sp['rows'], policy, 1), 'blocks_per_sm': _np_cps_launch(sp, x, out, sp['rows'], policy, 3), 'dynamic_smem': _np_cps_launch(sp, x, out, sp['rows'], policy, 4), 'fmt': sp['fmt'], 'inst': sp['inst'], 'cfg': sp['cfg']}
'BF16 cuBLASLt projections with offline descriptors and a cold-L2 tuning probe.'
import statistics as _np_statistics
import time as _np_time
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_tt_lt__LT_CPP = _np__zsrc('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')
_np_tt_lt__LT_EXT = None
_np_tt_lt__LT_WORKSPACE_BYTES = 128 << 20
_np_tt_lt__LT_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')
_np_tt_lt__LT_H200_FIXED = {(256, 'o'): (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (128, 'unpooler'): (2, 0, [66, 18, 1, 0, 0, 3, 35, 0, 5]), (256, 'unpooler'): (3, 0, [66, 20, 1, 0, 0, 4, 35, 0, 2]), (128, 'qkv_gate'): (0, 0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (256, 'qkv_gate'): (0, 0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (128, 'pooler'): (2, 0, [66, 15, 1, 0, 0, 4, 35, 0, 2]), (256, 'pooler'): (3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2])}
_np_tt_lt__LT_H200_VERSION = 130101
_np_tt_lt__LT_SEARCH_SEEDS = {(128, 'qkv_gate'): [(2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 1, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 4])], (128, 'o'): [(2, 0, [66, 18, 1, 0, 0, 4, 35, 0, 2]), (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 2, 35, 0, 5]), (3, 0, [66, 18, 1, 0, 0, 5, 35, 0, 2])], (128, 'pooler'): [(2, 0, [66, 15, 1, 0, 0, 1, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 2, 35, 0, 5])], (128, 'unpooler'): [(2, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 18, 1, 0, 0, 6, 35, 0, 5]), (2, 0, [66, 28, 1, 0, 0, 2, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 5, 35, 0, 2])], (256, 'qkv_gate'): [(0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 9]), (1, 0, [66, 134, 1, 0, 0, 2, 35, 0, 9]), (0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 2])], (256, 'o'): [(1, 0, [66, 20, 1, 0, 0, 7, 35, 0, 5]), (2, 0, [66, 20, 1, 0, 0, 4, 35, 0, 5]), (3, 0, [66, 20, 1, 0, 0, 5, 35, 0, 3]), (2, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5])], (256, 'pooler'): [(3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 4, 35, 0, 3]), (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 8]), (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 9])], (256, 'unpooler'): [(2, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5]), (2, 0, [66, 20, 1, 0, 0, 5, 35, 0, 2]), (3, 0, [66, 20, 1, 0, 0, 0, 35, 0, 3])]}

def _np_tt_lt__lt_build():
    global _np_tt_lt__LT_EXT
    if _np_tt_lt__LT_EXT is None:
        _np_tt_lt__LT_EXT = _np_load_inline(name='muse_cublaslt_v1', cpp_sources=_np_tt_lt__LT_CPP, with_cuda=True, extra_cflags=['-O3', '-std=c++20'], extra_ldflags=['-l:libcublasLt.so.13'], verbose=False)
    return _np_tt_lt__LT_EXT

def _np_tt_lt__lt_state(weights: dict) -> dict:
    ext = _np_tt_lt__lt_build()
    device = weights['qkv_gate'].device
    workspace = _np_torch.empty(_np_tt_lt__LT_WORKSPACE_BYTES, device=device, dtype=_np_torch.uint8)
    return {'ext': ext, 'version': ext.version(), 'context': ext.Context(workspace), 'configs': {}, 'weights': {name: weights[name].contiguous() for name in _np_tt_lt__LT_LAYERS}, 'packed': {}, 'workspace': workspace}

def _np_tt_lt__lt_plan(st: dict, layer: str, m: int, layout: int, epilogue: int):
    w = st['weights'][layer]
    n, k = w.shape
    if layout >= 2:
        if layer not in st['packed']:
            st['packed'][layer] = w.t().contiguous()
        w = st['packed'][layer]
    return st['ext'].Plan(st['context'], w, m, n, k, layout, epilogue)

def _np_tt_lt_prepare_cublaslt(weights: dict) -> dict:
    device = weights['qkv_gate'].device
    props = _np_torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0) or not _np_tt_lt__LT_H200_FIXED:
        return {'configs': {}}
    st = _np_tt_lt__lt_state(weights)
    if _np_tt_lt__LT_H200_VERSION is not None and st['version'] != _np_tt_lt__LT_H200_VERSION:
        return {'configs': {}}
    for key, descriptor in _np_tt_lt__LT_H200_FIXED.items():
        m, layer = key
        layout, epilogue, config = descriptor
        plan = _np_tt_lt__lt_plan(st, layer, m, layout, epilogue)
        index = plan.restore(list(config))
        if index < 0 or plan.metadata()[index][:9] != list(config):
            continue
        x = _np_torch.zeros((m, st['weights'][layer].shape[1]), device=device, dtype=_np_torch.bfloat16)
        out = _np_torch.empty((m, st['weights'][layer].shape[0]), device=device, dtype=_np_torch.bfloat16)
        if plan.run(x, out, index) == 0:
            st['configs'][key] = (plan, index)
    _np_torch.cuda.synchronize(device)
    return st

def _np_tt_lt_run_cublaslt(x: _np_torch.Tensor, layer: str, n: int, st: dict) -> _np_torch.Tensor | None:
    entry = st['configs'].get((x.shape[0], layer))
    if entry is None or not x.is_cuda or x.dtype != _np_torch.bfloat16 or (not x.is_contiguous()):
        return None
    w = st['weights'][layer]
    if x.device != w.device or w.dtype != _np_torch.bfloat16 or tuple(w.shape) != (n, x.shape[1]) or x.data_ptr() % 256:
        return None
    out = _np_torch.empty((x.shape[0], n), device=x.device, dtype=_np_torch.bfloat16)
    rc = entry[0].run(x, out, entry[1])
    if rc:
        raise RuntimeError(f'cuBLASLt launch failed: status {rc}')
    return out
'Experimental lossless BF16 register-A WGMMA projections, one launch per call.'
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline

def _np_tt_mw_mma(n):
    nr = n // 2
    regs = ','.join((f'%{i}' for i in range(nr)))
    outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
    inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
    ops = ['wgmma.fence.sync.aligned;']
    for kk in range(4):
        ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
_np_tt_mw_CUDA_PREFIX = '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t l0,uint32_t l1,uint32_t c,uint32_t t0,uint32_t t1,uint32_t* a) {\n  uint32_t cm=c&0x77777777u,lo=prmt(t0,t1,cm),hi=prmt(t0,t1,cm>>16);\n  a[0]=prmt(l0,lo,0x5140u)|((c<<12)&0x80008000u);\n  a[1]=prmt(l0,lo,0x7362u)|((c<<8)&0x80008000u);\n  a[2]=prmt(l1,hi,0x5140u)|((c<<4)&0x80008000u);\n  a[3]=prmt(l1,hi,0x7362u)|(c&0x80008000u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
_np_tt_mw_CUDA_CORE = _np__zsrc('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')
_np_tt_mw_CUDA_MAP = '\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n'
_np_tt_mw_CUDA_HOST = '\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;\n  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {DISPATCH}\n  return -1;\n}\nint info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}\n'
_np_tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def _np_tt_mw_cuda_source(instances):
    cases = []
    attrs = []
    for i, inst in enumerate(instances):
        bn, r, wk, pf, q, c, cap = inst
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    return _np_tt_mw_CUDA_PREFIX + '\n'.join((_np_tt_mw_mma(n) for n in (16, 32, 64))) + _np_tt_mw_CUDA_CORE + _np_tt_mw_CUDA_MAP + _np_tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def _np_tt_mw_build(instances):
    src = _np_tt_mw_cuda_source(instances)
    name = 'm32w_' + _np_hashlib.sha256(src.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])
_np_tt_mw_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

def _np_tt_mw__to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_tt_mw__to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_tt_mw_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_tt_mw_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_tt_mw_SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_tt_mw__to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_tt_mw_sm_c12_unpack(data, base, esc):
    """Inverse of sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_tt_mw_SM_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_tt_mw_sm_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_tt_mw_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != _np_torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = _np_torch.full((n // r * total * rw, 64), -1, dtype=_np_torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = _np_tt_mw_sm_c12_bytes(v)
        if not _np_torch.equal(_np_tt_mw_sm_c12_unpack(data, b, entries), v):
            return None
        table = _np_tt_mw_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = _np_torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = _np_torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(_np_torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = _np_torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = _np_torch.arange(len(gs), device=w.device) - _np_torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = _np_tt_mw__to_i32(entry)
    else:
        data = _np_tt_mw_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def _np_tt_mw_launch_raw(ext, idx, pk, x, out):
    return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], _np_torch.cuda.current_stream().cuda_stream)

def _np_tt_mw_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = _np_torch.eye(k, device=w.device, dtype=_np_torch.bfloat16)
    out = _np_torch.empty((k, n), device=w.device, dtype=_np_torch.bfloat16)
    for start in range(0, k, m):
        if _np_tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(_np_torch.equal(out.view(_np_torch.int16), w.T.contiguous().view(_np_torch.int16)))
_np_tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def _np_tt_mw_prepare_m32w(weights, configs=None) -> dict:
    configs = _np_tt_mw_CONFIGS if configs is None else configs
    device = next(iter(weights.values())).device
    state = {'plans': {}, 'device': device}
    if device.type != 'cuda' or _np_torch.cuda.get_device_capability(device) != (9, 0):
        return state
    instances = []
    for inst, split in configs.values():
        for c in (inst[5], False):
            twin = inst[:5] + (c,) + inst[6:]
            if twin not in instances:
                instances.append(twin)
    ext = _np_tt_mw_build(instances)
    state['ext'] = ext
    packs = {}
    for (m, layer), (inst, split) in configs.items():
        if layer not in weights:
            continue
        w = weights[layer]
        for compressed in (inst[5], False):
            actual = inst[:5] + (compressed,) + inst[6:]
            idx = instances.index(actual)
            if ext.info(idx) % 10000:
                continue
            key = (layer, inst[1], inst[2], split, compressed)
            if key not in packs:
                packs[key] = _np_tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
            pk = packs[key]
            if pk is None or not _np_tt_mw_verify_onehot(ext, idx, pk, w, m):
                continue
            if m == 32 and (not _np_tt_mw_verify_onehot(ext, idx, pk, w, 33)):
                continue
            state['plans'][m, layer] = (idx, pk)
            break
    return state

def _np_tt_mw_run_m32w(x: _np_torch.Tensor, layer: str, n: int, st: dict) -> _np_torch.Tensor | None:
    if x.ndim != 2 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
        return None
    m = x.shape[0]
    if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    key = (16 if m <= 16 else 32, layer)
    plan = st['plans'].get(key)
    if plan is None:
        return None
    idx, pk = plan
    if n != pk['n'] or x.shape[1] != pk['k']:
        return None
    out = _np_torch.empty((m, n), device=x.device, dtype=_np_torch.bfloat16)
    rc = _np_tt_mw_launch_raw(st['ext'], idx, pk, x, out)
    if rc:
        raise RuntimeError(f'm32w launch failed: {rc}')
    return out
_np_V9_SELECTED_ROUTES = [{'m': 32, 'layer': 'o', 'name': 'ol_1', 'module': 'ol_v8', 'config': [32, 128, 8, 1, 1, 1]}, {'m': 32, 'layer': 'pooler', 'name': 'cps', 'module': 'cps_v8'}, {'m': 32, 'layer': 'unpooler', 'name': 'mw', 'module': 'mw'}, {'m': 128, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 128, 'layer': 'unpooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'qkv_gate', 'name': 'lt', 'module': 'lt'}]

def _np_prepare_v9(weights: _np_Weights) -> dict:
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = weights['router'].to(_np_torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = (weights['router'] - hi.float()).to(_np_torch.bfloat16).t().contiguous()
    state['router_weight'] = weights['router']
    state['sm'] = _np_prepare_small_m(weights)
    state['m32'] = _np_prepare_m32(weights)
    state['wide2'] = _np_prepare_wide2(weights)
    state['router'] = _np_prepare_router(weights)
    state['qkv'] = _np_prepare_qkv(weights)
    if state['qkv'].get('large') is not None:
        state['qkv']['is_enabled'] = True
    state['router2'] = _np_prepare_router2(weights)
    state['olayer'] = _np_prepare_olayer(weights)
    state['cps'] = _np_prepare_cps(weights)
    state['new_lt'] = _np_tt_lt_prepare_cublaslt(weights)
    state['new_mw'] = _np_tt_mw_prepare_m32w(weights)
    for row in _np_V9_SELECTED_ROUTES:
        m, layer = (row['m'], row['layer'])
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        x = _np_torch.zeros((m, weight.shape[1]), dtype=_np_torch.bfloat16, device=weight.device)
        _np_candidate_v9({'x': x, 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}, state)
    _np_torch.cuda.synchronize(weights['qkv_gate'].device)
    return state

def _np_candidate_sglang(x: _np_torch.Tensor, layer: str, n: int, state: dict) -> _np_torch.Tensor:
    if layer != 'router':
        return _np__bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= 8:
        return _np_F.linear(x.float(), state['router_weight'])
    output = _np_torch.mm(x, state['router_hi_t'], out_dtype=_np_torch.float32)
    output += _np_torch.mm(x, state['router_lo_t'], out_dtype=_np_torch.float32)
    return output

def _np_candidate_v8(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    m = x.shape[0]
    output = None
    if layer == 'router':
        if 2 <= m <= 8:
            output = _np_run_small_m(x, layer, n, state['sm'])
        elif 65 <= m <= 192:
            output = _np_run_router2(x, layer, n, state['router2'])
        if output is None:
            output = _np_run_router(x, layer, n, state['router'])
    elif layer == 'qkv_gate' and m == 128:
        output = _np_run_qkv(x, layer, n, state['qkv'])
    elif layer == 'o' and 65 <= m <= 224:
        output = _np_run_olayer(x, layer, n, state['olayer'])
    elif layer in ('pooler', 'unpooler') and 9 <= m <= 16:
        output = _np_run_cps(x, layer, n, state['cps'])
    if output is None and 9 <= m <= 32:
        output = _np_run_m32(x, layer, n, state['m32'])
    if output is None and m <= 32:
        output = _np_run_small_m(x, layer, n, state['sm'])
    if output is None and 64 < m <= 256:
        output = _np_run_wide2(x, layer, n, state['wide2'])
    return _np_candidate_sglang(x, layer, n, state) if output is None else output

def _np_candidate_base(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    if layer == 'router' and x.shape[0] <= 16:
        output = _np_run_router(x, layer, n, state['router'])
        return _np_candidate_sglang(x, layer, n, state) if output is None else output
    return _np_candidate_v8(data, state)

def _np_candidate_v9(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    m = x.shape[0]
    if layer == 'o' and m == 32 and (n == 8192) and (x.shape[1] == 1024) and (x.dtype == _np_torch.bfloat16) and x.is_contiguous() and (x.data_ptr() % 16 == 0):
        output = _np_run_olayer_candidate(x, state['olayer'], _np_OLAYER_CONFIGS.index((32, 128, 8, 1, 1, 1)))
        if output is not None:
            return output
    if layer == 'pooler' and m == 32:
        output = _np_run_cps(x, layer, n, state['cps'])
        if output is not None:
            return output
    if layer == 'unpooler' and m == 32:
        output = _np_tt_mw_run_m32w(x, layer, n, state['new_mw'])
        if output is not None:
            return output
    if layer == 'pooler' and m == 128:
        output = _np_tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'unpooler' and m == 128:
        output = _np_tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'pooler' and m == 256:
        output = _np_tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'qkv_gate' and m == 256:
        output = _np_tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    return _np_candidate_base(data, state)
'H200 small-m table refinements. Unclaimed shapes return None.'
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_tt_v10_sm2_SMALLM2_CUDA_TEMPLATE = _np__zsrc('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')
_np_tt_v10_sm2_SMALLM2_CPP = '\nint smallm2_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_tt_v10_sm2_SMALLM2_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_np_tt_v10_sm2_SMALLM2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_tt_v10_sm2_SMALLM2_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_np_tt_v10_sm2_SMALLM2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_tt_v10_sm2_SMALLM2_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_tt_v10_sm2_SMALLM2_ESC_CAP = 32

def _np_tt_v10_sm2_smallm2_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _np_tt_v10_sm2_smallm2_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
        cases.append(f'    case {i}: return smallm2_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _np_tt_v10_sm2_SMALLM2_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _np_tt_v10_sm2_smallm2_build(instances, name='smallm2_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _np_load_inline(name, _np_tt_v10_sm2_SMALLM2_CPP + extra_cpp, cuda_sources=_np_tt_v10_sm2_smallm2_cuda_source(instances) + extra_cuda, functions=['smallm2_launch', 'smallm2_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_tt_v10_sm2_smallm2_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_tt_v10_sm2_smallm2_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_tt_v10_sm2_smallm2_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_tt_v10_sm2_smallm2_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_tt_v10_sm2_smallm2_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_tt_v10_sm2_smallm2_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_tt_v10_sm2_smallm2_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_tt_v10_sm2_smallm2_c12_unpack(data, base, esc):
    """Inverse of smallm2_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_tt_v10_sm2_smallm2_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_tt_v10_sm2_smallm2_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_tt_v10_sm2_smallm2_units(hi), _np_tt_v10_sm2_smallm2_units(lo)]
    else:
        mats = [_np_tt_v10_sm2_smallm2_units(w)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_tt_v10_sm2_smallm2_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_tt_v10_sm2_smallm2_c12_bytes(v)
        if not _np_torch.equal(_np_tt_v10_sm2_smallm2_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_tt_v10_sm2_smallm2_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_tt_v10_sm2_smallm2_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_tt_v10_sm2_SMALLM2_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_tt_v10_sm2_SMALLM2_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_tt_v10_sm2_smallm2_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp['fmt'] == 4:
        return ext.smallm2_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], _np_torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smallm2_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_tt_v10_sm2_smallm2_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_tt_v10_sm2_smallm2_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_tt_v10_sm2_smallm2_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_tt_v10_sm2_smallm2_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_tt_v10_sm2_smallm2_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_tt_v10_sm2_smallm2_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_tt_v10_sm2_smallm2_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _np_tt_v10_sm2_smallm2_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _np_tt_v10_sm2_smallm2_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_tt_v10_sm2_smallm2_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _np_tt_v10_sm2_smallm2_verify_onehot(ext, inst_id[inst], sp, _np_tt_v10_sm2_smallm2_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _np_tt_v10_sm2_smallm2_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
_np_tt_v10_sm2_SMALLM2_ROUTES = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1)), ('pooler', 8): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1))}

def _np_tt_v10_sm2_prepare_smallm2(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in _np_tt_v10_sm2_SMALLM2_ROUTES.items():
        w = weights.get(layer)
        if w is None or w.dtype != _np_torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or _np_torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, _np_tt_v10_sm2_smallm2_bf16_twin(inst)))))
    ext = _np_tt_v10_sm2_smallm2_build(instances, 'smallm2_final_v3')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = _np_tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = _np_tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, _np_tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st

def _np_tt_v10_sm2_run_smallm2(x, layer, n, st) -> _np_torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != _np_torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != _np_tt_v10_sm2_SMALLM2_LAYER_K[layer]:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = _np_torch.empty((x.shape[0], n), device=x.device, dtype=_np_torch.bfloat16)
    rc = _np_tt_v10_sm2_smallm2_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None
_np_tt_v10_sm2_SMALLM2_ROUTES = {key: value for key, value in _np_tt_v10_sm2_SMALLM2_ROUTES.items() if key == ('pooler', 8)}
'Hopper BF16 projections tuned with cold-L2 H200 graph replays.'
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_tt_v10_tune_CUDA = _np__zsrc('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')
_np_tt_v10_tune_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,int,int,int,uint64_t,int);\n'

def _np_tt_v10_tune_source(configs: list) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'D void mma{n}(float* a,uint64_t w,uint64_t x) {{asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;":{outs}:"l"(w),"l"(x));}}')
    cases = '\n'.join((f"case {i}:return launch_t<{','.join(map(str, c))}>(x,w,y,m,n,k,stream,query);" for i, c in enumerate(configs)))
    return _np_tt_v10_tune_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CALL@@', '\n'.join((f'if constexpr(BN=={n}) mma{n}(a,aw,ax);' for n in ns))).replace('@@CASES@@', cases)

def _np_tt_v10_tune_build(configs: list):
    for c in configs:
        bn, wg, st, pack, split, push, pw, px, hint, order, sw, prod = c
        assert bn in (128, 192, 256) and wg in (1, 2) and (st >= 2) and (split in (1, 2, 3, 4))
        assert st * ((64 * wg + bn) * 128 + 16) + 16 <= 227328
        assert prod in (0, 1) and (prod or wg == 1)
        assert pack in (0, 1) and push in (0, 1) and (pw in range(4)) and (px in range(4))
        assert hint in range(4) and order in (0, 1) and (sw in (0, 5))
        if push and split > 1:
            assert split * ((bn // 8 + split - 1) // split) * (wg * 128) * 16 <= st * (64 * wg + bn) * 128
    src = _np_tt_v10_tune_source(configs)
    name = 'm256tune_' + _np_hashlib.sha256(src.encode()).hexdigest()[:16]
    return _np_load_inline(name, _np_tt_v10_tune_CPP, cuda_sources=src, functions=['launch'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=True, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def _np_tt_v10_tune_pack_weight(w: _np_torch.Tensor, rows: int) -> _np_torch.Tensor:
    n, k = w.shape
    col = _np_torch.arange(64, device=w.device)[None, :] ^ (_np_torch.arange(rows, device=w.device)[:, None] & 7) * 8
    return w.reshape(n // rows, rows, k // 64, 64).permute(0, 2, 1, 3).gather(-1, col.expand(n // rows, k // 64, rows, 64)).reshape(n, k).contiguous()

def _np_tt_v10_tune_run_candidate(x: _np_torch.Tensor, w: _np_torch.Tensor, ext, index: int, n: int) -> _np_torch.Tensor:
    out = _np_torch.empty((x.shape[0], n), device=x.device, dtype=_np_torch.bfloat16)
    rc = ext.launch(index, x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0], n, x.shape[1], _np_torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'm256tune launch failed {rc}')
    return out
_np_tt_v10_tune_CONFIGS = []
_np_tt_v10_tune_ROUTES = {}

def _np_tt_v10_tune_prepare_m256tune(weights: dict) -> dict:
    state = {'routes': {}}
    if not _np_tt_v10_tune_CONFIGS:
        return state
    device = next(iter(weights.values())).device
    if device.type != 'cuda' or _np_torch.cuda.get_device_capability(device) != (9, 0):
        return state
    state.update(ext=_np_tt_v10_tune_build(_np_tt_v10_tune_CONFIGS), weights={}, routes=_np_tt_v10_tune_ROUTES)
    for layer, index in _np_tt_v10_tune_ROUTES.items():
        w = weights[layer].contiguous()
        cfg = _np_tt_v10_tune_CONFIGS[index]
        state['weights'][layer] = _np_tt_v10_tune_pack_weight(w, 64 * cfg[1]) if cfg[3] else w
    return state

def _np_tt_v10_tune_run_m256tune(x: _np_torch.Tensor, layer: str, n: int, st: dict) -> _np_torch.Tensor | None:
    index = st['routes'].get(layer)
    if index is None or x.ndim != 2 or (not 129 <= x.shape[0] <= 256):
        return None
    if x.dtype != _np_torch.bfloat16 or not x.is_cuda or (not x.is_contiguous()) or x.data_ptr() % 16:
        return None
    w = st['weights'][layer]
    if x.device != w.device or tuple(w.shape) != (n, x.shape[1]) or w.dtype != _np_torch.bfloat16:
        return None
    return _np_tt_v10_tune_run_candidate(x, w, st['ext'], index, n)
_np_tt_v10_tune_RL_CUDA = _np__zsrc('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')
_np_tt_v10_tune_RL_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rl_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def _np_tt_v10_tune_rl_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void rl_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return rl_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) rl_mma{n}(a,aw,ax);' for n in ns))
    return _np_tt_v10_tune_RL_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def _np_tt_v10_tune_build_rl(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (2, 3, 6, 7, 14, 15)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported rl configuration: {cfg}')
    _np_tt_v10_tune_source = _np_tt_v10_tune_rl_source(configs)
    name = 'rl_' + _np_hashlib.sha256(_np_tt_v10_tune_source.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_tt_v10_tune_RL_CPP, cuda_sources=_np_tt_v10_tune_source, functions=['rl_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)
'Lossless BF16 projection refinements; only confirmed exact shapes are claimed.'
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_tt_v10_warps_W0_DC2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_tt_v10_warps_W0_DC2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_tt_v10_warps_W0_DC2_ESC_CAP = 32

def _np_tt_v10_warps_wp0_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_tt_v10_warps_wp0_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_tt_v10_warps_wp0_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_tt_v10_warps_wp0_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_tt_v10_warps_wp0_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_tt_v10_warps_wp0_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else _np_tt_v10_warps_W0_DC2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_tt_v10_warps_wp0_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_tt_v10_warps_wp0_c12_unpack(data, base, esc, decode=0):
    """Inverse of wp0_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else _np_tt_v10_warps_W0_DC2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_tt_v10_warps_wp0_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_tt_v10_warps_wp0_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_tt_v10_warps_wp0_units(hi), _np_tt_v10_warps_wp0_units(lo)]
    else:
        mats = [_np_tt_v10_warps_wp0_units(w, layout)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_tt_v10_warps_wp0_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_tt_v10_warps_wp0_c12_bytes(v, decode)
        if not _np_torch.equal(_np_tt_v10_warps_wp0_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(_np_tt_v10_warps_wp0_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_tt_v10_warps_wp0_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_tt_v10_warps_wp0_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_tt_v10_warps_wp0_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_tt_v10_warps_W0_DC2_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_tt_v10_warps_W0_DC2_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_tt_v10_warps_wp0_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
_np_tt_v10_warps_W0_CUDA = _np__zsrc('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')
_np_tt_v10_warps_W0_CPP = '\n\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint wp_bde7122f9b4f_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n\n'
_np_tt_v10_warps_W0_INSTANCES = [(1, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (0, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1)]
_np_tt_v10_warps_WARPS_FAMILY_SPECS = {0: {'name': 'wp_bde7122f9b4f', 'launch': 'wp_bde7122f9b4f_launch', 'family': 'dc5', 'decode': 5}}
_np_tt_v10_warps_WARPS_ROUTES = {('qkv_gate', 16): {'family': 0, 'inst': (1, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), 'cfg': (1, 4, 8, 4), 'rows': 2304, 'k': 8192}}
_np_tt_v10_warps_WARPS_PACKERS = {0: (_np_tt_v10_warps_wp0_pack_format, _np_tt_v10_warps_wp0_stream)}
_np_tt_v10_warps_WARPS_SOURCES = {0: (_np_tt_v10_warps_W0_CPP, _np_tt_v10_warps_W0_CUDA, _np_tt_v10_warps_W0_INSTANCES)}

def _np_tt_v10_warps_warps_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0) -> int:
    return ext[sp['family']](inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), x.shape[0], n, x.shape[1], *cfg, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_tt_v10_warps_warps_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8) -> bool:
    n, k = exact.shape
    x = _np_torch.zeros((rows_per_launch, k), device=exact.device, dtype=_np_torch.bfloat16)
    out = _np_torch.empty((rows_per_launch, n), device=exact.device, dtype=_np_torch.bfloat16)
    for column in range(0, k, rows_per_launch):
        count = min(rows_per_launch, k - column)
        x.zero_()
        x[_np_torch.arange(count, device=x.device), _np_torch.arange(column, column + count, device=x.device)] = 1
        if _np_tt_v10_warps_warps_launch_raw(ext, inst_id, sp, x, out, n, cfg):
            return False
        if not _np_torch.equal(out[:count], exact[:, column:column + count].t()):
            return False
    return True

def _np_tt_v10_warps_prepare_warps(weights) -> dict:
    state = {'plans': {}, 'ext': {}}
    entries = []
    for (layer, m), route in _np_tt_v10_warps_WARPS_ROUTES.items():
        weight = weights.get(layer)
        if weight is None or weight.dtype != _np_torch.bfloat16 or (not weight.is_cuda):
            continue
        if weight.ndim != 2 or weight.shape[0] < route['rows'] or weight.shape[1] != route['k']:
            continue
        if _np_torch.cuda.get_device_capability(weight.device) != (9, 0):
            continue
        entries.append(((layer, m), route))
    for family in sorted({r['family'] for _, r in entries}):
        spec = _np_tt_v10_warps_WARPS_FAMILY_SPECS[family]
        cpp, cuda, instances = _np_tt_v10_warps_WARPS_SOURCES[family]
        ext = _np_load_inline(spec['name'], cpp, cuda_sources=cuda, functions=[spec['launch']], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        state['ext'][family] = getattr(ext, spec['launch'])
    packs = {}
    streams = {}
    for key, route in entries:
        layer, m = key
        family = route['family']
        spec = _np_tt_v10_warps_WARPS_FAMILY_SPECS[family]
        weight = weights[layer][:route['rows']].contiguous()
        n, k = weight.shape
        pack_format, stream = _np_tt_v10_warps_WARPS_PACKERS[family]
        cfg = route['cfg']
        original = route['inst']
        for fmt in (1, 0):
            inst = (fmt, *original[1:])
            pack_key = (family, layer, fmt, n)
            if pack_key not in packs:
                packs[pack_key] = pack_format(weight, fmt, decode=spec['decode']) if spec['decode'] is not None else pack_format(weight, fmt)
            packed = packs[pack_key]
            if packed is None:
                continue
            stream_key = pack_key + (inst[2], *cfg)
            if stream_key not in streams:
                if spec['family'].startswith('rag_'):
                    rows, columns, size = packed['data'].shape
                    target = cfg[1] * cfg[2] * cfg[3]
                    padded = _np_torch.zeros((rows, target, size), device=weight.device, dtype=packed['data'].dtype)
                    padded[:, :columns].copy_(packed['data'])
                    packed = dict(packed, data=padded)
                streams[stream_key] = stream(packed, inst[2], *cfg)
            sp = streams[stream_key]
            if sp is None:
                continue
            index = _np_tt_v10_warps_WARPS_SOURCES[family][2].index(inst)
            plan = dict(sp, family=family, inst_id=index, cfg=cfg, rows=n, k=k, inst=inst)
            x = _np_torch.empty((m, k), device=weight.device, dtype=_np_torch.bfloat16)
            out = _np_torch.empty((m, n), device=weight.device, dtype=_np_torch.bfloat16)
            attr = _np_tt_v10_warps_warps_launch_raw(state['ext'], index, plan, x, out, n, cfg, 2)
            occupancy = _np_tt_v10_warps_warps_launch_raw(state['ext'], index, plan, x, out, n, cfg, 1)
            required = n // (16 * inst[2] * cfg[0]) if cfg[3] > 1 else (n // (16 * inst[2] * cfg[0]) + 131) // 132
            if attr < 0 or attr % 1000 or occupancy < required:
                continue
            if not _np_tt_v10_warps_warps_verify_onehot(state['ext'], index, plan, weight, cfg, inst[1] * 8):
                continue
            state['plans'][key] = plan
            break
    return state

def _np_tt_v10_warps_run_warps(x, layer, n, st) -> _np_torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != _np_torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != plan['k']:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = _np_torch.empty((x.shape[0], n), device=x.device, dtype=_np_torch.bfloat16)
    rc = _np_tt_v10_warps_warps_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None
'Lossless m32 pooler and qkv projections, with an optional unpooler route.'
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_tt_v10_m32x_M32X_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_tt_v10_m32x_M32X_ESC_CAP = 32
_np_tt_v10_m32x__CFG = (1, 8, 16, 1)
_np_tt_v10_m32x__CUDA = _np__zsrc('eNrNPWtz27ay3/MrkHScQ9qMIkqK7fjVcZ20p+PEyVhO0zu+PhqaoizFFCmT1MNu89/v7gIgQRCU5KS9p8nIAkFgsVjsLnbxWP3QDwajKGBvTlq9N29PPrx5y149efLDKPLDaT9gB/607/WSaZSNxkFjeKS/uR642+XsNOuPoqyc54U3cTLKhmM1M7ufBL0s8UZZWgIbx5Mg8bLRLOjdJPF0kiKsJz8IPH/9+Q3r9frBbOQHvR4kB3HiB6MohJe93pMn+B7a77Bxu7Xohf255cdRmlHeJpvY7I8njBeY7UPKS8dsFofQXBhYz8J+4yaMr72wEfmNd+7eXhT3vDCMfS8LGrNOI2232B8bTWfDdTZazkb7q8MuNzpX+88AUunfHnt2mDyzZo2F7cjkfZF8KJJzGwuHz6yJbSNCSZBNkwiR+8o7M1V7sxC9mcWjft6b6VrdAfSndej/1fi2W71MohwIlGV+CW0quApzRBuwBlxdFdfHITJJxpmVZ3lO8fpaSacaan2BmvUM6zeuBSZAQAYUZBvtAp8+4QMJD4kF39fiOy3h18/xwzHkuI3HnjUIYy9j1nPfvuxcOUynWQnhpvrgcpQJSQDUSO+BZF44uomCfmPsbke7tyCiSTxv+HHYGLRbDZJZ+jOoMsQfGx1n45Wzse1s7NDjrrPxmhJKsZzhocdbg2eWf9m8wt7ytKukW0q6fWUX1ZBQohYl3SLZKpJtmbxuyoSL5AQSvnzJdtk8GN0Ms5TFAxaDcoCesjQLJnssjOfs+j4LUpB19q7JrJkXTuGp+aJtMy/qs3duntd5sQPAOy+uRxnQvc/rnDS0QeoHfi8O+wUPvVOH4Z2rPJwo6Qu11IVSCsaUj1x5qJk/ZofshD1nzcWO+DfdrxYbhjGUK1gb20Hw/tg2lR6OakqzoyPmblMdHI9SKewhtOMAKq/cTnNqsz+ZZZ2wgwPmtmxCcbfZbOJnKiC4tRB22tstFcKuGUBLAwBIAvImFDpmAO1aAAUGJ5WaRp0xnoZmlZH6XhjUqgqo1kgikrCFqi+W6AoO0KwnNJy89LZ3PfJSBbNaTMJ40i4pLej0zmCAH0x3oPP4wXTgGdFbpbpAKv5qifhhNFCtkcND1gQcSjKIDSF8ACs52QNMfwjCSl13SV2gQLvT6iBf0cNOZ7vzqtOZCmiTxLsZe2waJXEYAhiwNJgFuLI74LDmPnwBE+6zra07UCmXdznfIdvgMyLHsUoDvVdHh6yjjtlwlA9aOhw0ksY88Sbl+QY+nWKMhiM5SCd23khV7lHsUV5AyoW8NHf4f6Mo8f9TAhj1R4M6rXBSCGSt3J8UMlcv2iMdULuukIT0qGExcVOL/fmnltWheUkfQ03e8LVdGlOlEr3lRBQCBvTdgsIKJb/KJEhRyYSNYmm/KpIVBbMgsQpDT7OQwHCe7D+z9xmfCs+wNMu82yBy2PU0Y5Ns4aXM96IohockngVslO0xQCBl2TBg11468tk12La3LI1hvvTgxShNpwGbD4ME/oLBngWRPg3GST9IehPAOCOij52qMUqsl1WtunE8y425jQ0f2y4YOuO6BobLsjKgZK5W7LFNSmABRKV/QFmFQpM67Z2Og3EPmquYzFKfFS30ev4s83o3QRQkI7+Xxb106CVBH4EzI/DEi24trcfJ6h5PwTpJen7mYf2i70lJzya1qn/imecjhFYzCajIQP0G79jensClwG7VFEVtrLRkr72kN4pGiqkNOQqmfgyuZG6xqshBuWQECGHtAsvMa1xvd8D0b14hjojfHpmBXiLwEhABWRjvOLl/ZtfhFSwmgV+LWbZYihZ+zYIGh9HLFhUUe84yLBH6chQHQQTqgGhXlXZ62ZDIUKlGEoSBlwYNMZLUqNoEMG65BT/s8V4YGpDdlGwhugtteAvwI4Q/sVYbc8/YBb0FLLYMrgY4zXoeujYdS+F6h5HbBM6vKgxIdcNQptAcOUe6CEiyQk48noRBFsD47u2R+4BuM3pJfGj/KBw/+HTIf35lcv/3uMTg4A9yt3qQe9WD3KkecJ9aiBhhbmKT6zgOFU7OkvsSG2uyH99Whf+P/40YDOcNa0yg82yyD885cwM8GrbGxAOtf9/w/LvpKMlZq8LrE+6Mw7QMYCq9f8bSIJzkmgXNToca/Jprl/hWqpdql3MFE9+ypzijq9pwuyO14Q2uSCkqmN7UTTqAM1fBfFUBqyZC14X51COazaR7+RNolj50m3nsOolhZmUw6Uap52ejOOKajKXDeJ6y6QQahWJ8TmcDbxriihPQzuujb4oz0HQ8im7AahmA8KZDGIK+B9TUuBwHoQckMYwuToxlDsBXtsC67PNxWjSlMSNIhWXmQyAKs54aINEw5q1I8r5AOEes1eT/pmFog8WC1odl0mJev9/haxmd5zhFcZxkxjWphcaCbR2y68ZiH9L3PH2P6QeefsD0nKfnuSoQks6VwXAQ9hZx0rHK8EENoE0Sc4qJ8RyDUdTjJaxeXrWHqsBqLgbSqAA7prFwoK7DVpS6p1I61/N/K6o+rNXAHEuZqJtmcRJAJ6JZ75p65G5vojDyxaMZUndzAqPe4/1tyUK9JLJmBm3N4VEhI5wZUT8LQCt6WcAOkLg/v784YmmWTP2MvQcgP49BYIoi+ruD5hFASzOQRp9zA0yhCQ3T2TtoouOws2P4dqGppXDc5XDa68JpLYezK+C0VsFpL4ezXYJjoGHVkY7ztVJa0LWez+1L2RrW2Ns7e3elTHTW8+NKgWO2ydztqyXutolv8/ef1MKf3FzzFB20oCEyx9GLEukWOVq6X6Z4ZiH3zELwzPQOoaMW8nbw3/FlB3oQgutEXmdhos8vwyvQGJVirqHYfbVYy1DsoVqsbSg258W+ogfH0O9b0dkh7+yw2tlj7Oyw6Ky6MzGFWnPEADAZ7pcKFCYqavQh+xGGCaYuHNjMlTku5riyYrE2A947qjX6vsfvFj1nVNlhx0jCbWwTSW6b6z+I+nNR/76mfsdU3xXtu0X7DzX1d2vqP4j6sv15TX23ZStjJWbxt6nvTQJwfGGi2wPL5I6WL3fZnyz0ooAe2vAAptEIbR1amoTnoRcOML0D6R1aGv79wznO5eg8D0c3Q1pe/lfKQCriCKAzKJM2NEE/O9flfOJl/tAqy/DZuSqwgXFZGE03YM0AV3Jo6bTtmpaFFwkvRcvHP4OnDF2wdoE2Fq+6g1VdyN/kq771qykJZ2MgCPQCGTcpLARCRhUUsFGOL5Mr9p9DwEDaT8f9L56PlEFCp9ADMIY8mSem8zSMs9Rh81E2ZIHnD/mggLFE1lI2hOkIiwCd00k4yl6cMjBRs5EXlkjth16asrfjUaaTG8zWqR+UzIRNliY+NxTOfuXfH8VX9+Kcp7JRnyeizCG4LIA/fGByCo36C6ARFN2nJNDpV54Cy0U6uSz3UNAYUyyRZgOXYNU/gndz8BM+ABMA/BHpP7EVEwssHejFpZWOHsBbsScwnog9DDQgcCVAIc4WPINNQTlSJEBrR7ss4yYt2IMwAOcXwA8vknjOM9gEJGHuJWATdC+AOYC1cVNkEOJ2C72URRvsza/HvzALzEUcsTgK7+095rIoZu/fHzvYXAsfFnyJyWEdfKJhQTPaAWHja09Y00FZhlHN7tltkERB6DBwIiziDmB6GG1AIp5mkymwBdouAA9M+7yAB9BPf8I2U9xYBlxf4Loa7eqItS5gKJTgIIIBdlu7om7HRqxKkBvVOZuzhPw+F99d+fz2DU8gQQR/Xbw//p0nT96L7zPxfSq+P5/LhMw5/SQSXf79u6h6IaCefDx60utxT0YuGfZ6oTeN/GHvGp2WtNezqG3BMjfBeGY90aecHkgH0GnkZwBlPpHGumZelkot5GpfKRfp5jypTlhasSAVYjfuecmNoFCRPC2Sn8+VtJJ/+omnlxs3y62ZnAKMNtrew+QJf/Y4UvulAmdY4AwLnEEBgeopZp5i5ilknlZrfT7HEvD3R/q7l/fmM1X9THU/Y2XZtdNPBPMTAf2EUD+VwJYtS92CEoambms4wKgXmH8BuuHilzIwWk84+/AZ3pP4PmetV9uGIiduC+cTYee5is3XhonkOXsKMAzV/v3ruw9QD4seoQlsahu0Q9E6zORnH35XsMFnEKgiB/yE8w/FI5jpb99/vPgfBYQJ//MPve4FACqKtVsIqMdX/mXmNnohH7SSoB0MED9+6v4bSULIic0EkW4r6Y4kDz5jCpDHL0IawU5TVJcfphmOER6FibwxTHwZrrLEUX+EytELD5CSwi9zypIJg4zVEBZC2UQpBFBJAIwSJBNwfnu+l2YH9PLIQhmlSSBYwOuIXFBc1KENCFqGA50BJgEo+5TW5JgP7xkuol9eYT1jBdC5+XpDigsXZVFY8C14INfv73Nq8WS7SL6ygfGRWTdhMthk1il7yayTLkpDF4ThJhn134zGjXubTEQo0YLcJseppKrShYkA5UJHFnbJLuOZeHOJ6clHaADEdhPFdRPnxU0Uu032ym2VK+VV8tpbsp8dIfubJOgSCvpmrWZnF7Hnex2khtO5CWt6x5EFuJyQcoNE8caIn6SpUSwF5Y5KSeejrYLcNkRT7Nf+Ag3zyZzbMGgcvnJozocMyH1J8012w5826ImMM14cDdB9DXyUlTSf7P3FL/CnjYNGUzCO5UKveoaqSJAdsEDjk1DZYDg/zm/l80t41ut2sdUqu+jFfvldUefAYdQajQyyfVGxgtsvUA8qb0JDMIqf9dc3SEGLuoY0vYdCUHqL5TkLm+pBFnZBr56QV1fUX3D+g8KJTQjqFfxmpT0gM5S/xfKnn/TyF7+h6TmKLHRyQcwcZiFLwXSIRz/sfcPEjauOcvEw30Z7TgIuype2nshQlc6BwpyoKwtvV5RUts+kt0cuBb3FdQVQk8SnkC4qVzBENyCtUcQdVCiFc2JZXfYC+4ujgCNtfPX5PMeH6ZtZtFPkGt+LTSUqwXESTCUR6+TKrYMu+m/kerV2pzo0df9HvvqqusPK9o30cg00X6FoK0sP7BapiCq3K1vNPZCZdG9muJaRK2io8ZLtYm7J0dFUDS4fopjx0jZYl5Szkefsa7W4YQleDZRboC4Vvg2yLkx+YzlQp6ps3XN8tgA6IFeA5MD6KaKRIrQMh4RK7trV4iYamglnYMXUq8gKNKz0T9+S8CdiJ8q/EVsr4rSj2Gbieytg0OC2GD8fRFtJISQSv7JdIniFq/zNykwyFVMJ4iQRxZOkCEpntNV0qOtJDB57xg/sVnbzBHDCr0cr3nwKSi1lvUbRKeP7Hk4zzcXP4t/UpF3QNuWWlS1r5EdOwdVQOAg0NE1AWzSBQZsVV2iOZ0fmE72OVUz/7ZYt6qsKkxdmKfo4fcRAVv+lUlutptkti6bG8qDiN9GxhRyacp+jub1JAwL29GvQKGhE0AIOJwx1AuyQBI3Kx1gTwJrTLMZ17JugJ09THrLL51ekAXDthbte6aiQAW0cPrIjrrbRG1AlBfV6OgLdcfqpZoVYUzhfDpv77MsBzHlsa+uLXV9DXVvGOuFBZSW5RlhBCg7LsirJhmtkFnZ48+zzFs7U9ub5BWCxefYO4G62W1uCe0zwhSmX+Ifzuy05jOloU3DGlvUlB6MD+F7dgHK9SjmoMq6mvkOaOZCv+0vXwu9wcO4OTj6K81glPrPunDsBjjY85pfdi6vLc/icvbsqzIwOW8zozXv4tK4UrsWFIzOz1rJbwWp8de0LcOcqdjNtY+TsVpEFacNwYUARAKl9inYAqNHsRtg180vE9uryy9VleFVorrkFmqjQA2hMSN0CuV+4OgE1ZKs6RRfIp+hKFz6nUOPNQo1vF8nXRdJt2kukNCdCxIkQkUmARIhMk6Ou4dCpADVXntUtIAeapJAd2WRcSIUHlmnL1ub8lHSiytPAF5yK0ZVySBHntsXErivoagWF1yb0Krkpu7Y6rz6G31NOnBSI0yXiACcil1qpw6TCrRyjGzss984fY4rUCC+dISDRBVRqhLdmFi4WrJnn+5dnx1wcUfA6V/vr7nrJfa6/QQINzLfGtOLzWr48GApaErs3JPEDlvCRJZqNgebaDOKwX7g2eXa0yIzGCTQUTOpcIdVY4QDEGYDq/j/aMg6tEGhdo7PM6v4D7xafYPEbzPHuxWrVZ2JSkwmfYgMjlLySoWx1+QIg9geKgODg9H4NrHRbZ1Bjr9ea/9dniVWKWZtkaSz/o25ZWaCChf4Fn/s/THm8Lz8+lB/nJdUibJwt3Cfhhk4u7zy3VHqF7Ct2vKpSC7ecRHH3SkPAoPuF4tgpFMeuoulde+l4PELb/70af1RR+WWLrsMemoeq0nfYg6tmbC1V7fzfNZ8+Dh+aeCLpms8R+HQvnlr09CCe2vQ0N0LpwDs3h/KKniSUbXqSUHboaV5noa3JK+Uq65rl680dG01hYkbPANYLV7Ms12GeR5jzjzDoq/43WN1Gk37LQot+tHHy0X6cTc8N0udsdljIfbVM5cqisNdnHbq4sfqmZXHRsrhnWbpmmayy6csMsD4zlO1wIpGTjraAUPsmvXPMrXJxuGj/2zS5NpBlBAWLrschHnCId3B2vAlukLe1hWcPYIi8q8Pmoj3YbeJnqkkWV6p11rhyAov2z+Tc4BDk4pIRbih+cjUBXIMaq7WoTo3f16dGwIEHAHwXYQc2VzEBWTWrKaK0+0hXwahHUe3PkHz0pKhU7UWhXbUXhaLVXtTpXLWYe6WoX+1FoYm1F4VS1l4s0c9f/04bwGA9rjthrxa8iv6c4qGow2LvChfgb+joNa6j54bu22l56Zxm53Ocyr9UtBLd4kE7F7cYdun+DrZiUub92JDJIfDtpkPqmF3Aa4vjT7atHKk6ENrpyOIiC4UNap2hfExM+dwoD/C0DejuH1eZ58EEtXLFCShLmDhsbSLEMjW+rlIX42VQJVau2dZVIo+0vMuLPMW9OxoEcfnOQOR0fmlZtI+Xc47cGMXFzjtbXacFYTSesaZTVDg/00hf3m12hJ7G5JarPrTUh/aVvczcMrrEGk2f/rdpahQV5cww7tg8jsgmWcjpejhDbarQFjLu91X6QsbDvkpjyJgbZOGb2Xvb7ME019WH1Uuha47eo5Ya1hh5AyPL2A+4GNEUixGcYXGccLwcMUdu4m1imYRxKJsgpqlcWS36NjKtzeaROKgYoe60yLcDZ+ElnnwHok3qDXjaJoxxT1h6hdwhjCakyZXdjx3ce7HkISSYk3Amkm5jhxxJg7JXtsxjXCGw5re0QU6epS13WO7yxnKINMGQF7q7v8r58Pr96lY533GMb+Vplm6+7RiHBkTVyB7XrnPdcq7bBpekNL6CGCa+0v2SsZclo4UWGGQ32m0sWtJluQYKcH+FPJWW4qnwyBuHPPSGvBMKnbYNHclvRX8rQp0qQjUOVAktmWrlqbaO6neIrlHpmoQX+bZVlWBwGq5dozFSGVI8kFEPmq47VMC34NO2V+ncpTp4ybWGv2iFbp11pEIMFtoal3FyOEJdALYqJA+E2fpUOE6r9M0jlqLqFEBluWuVwC4/mGFgDlpBS9VtYZNGMSu10rYxPxlZlVTDGh1vYcnaG8vX3XCp7fv6bJSrB9wLrx5iUNb8No/07Z28rrtW3dJNE4MU1CBV2lzibT1iG6lOKhlye7H6uFDWHhfKyuOibt2R6hfrjgtl1XGhrDkuzCuOFJRirVneaCAsH30647vSjli5i2TyAeRyPhmexf0j+L67Yv8xZfOrYnWvWvDKfKFTNYkr1dpG8znfa1iqpX6kAeYY7wnbjp5swHLdqrxXanXKeRx3L7dDqa/ON3fG+e6+rDFz4ilpbd4EfwfUt94ZMvT+iT36+leuB9S6stpetun8oL5S/M2bzKYdQPPeIB3G188N8YPI2hlQxUbAM3coaGD8iBSIjTvlxwzFYwse+8Vju7RIo9nxt6WLn2P2J8P7dasNIkUvbpsWPvQQLP1GGBdBV/ADNtzGKwrHoLwUAZ/ML1siCIV8+axedT0rVWzLyBVUkWzpLRniZUuGLMSELxN5GJhbEaPirsy9Yo/cA/Jew8eHT7+O3wwjT1cu9KFXjBdiAH3tlN9ax+sOW3R6uuQsCuOt18Ox7Hkpv29nEZ7PaTlcBhEqTjLpqhGR4B3DBfTXQXtn5/X1axl2KG/eYW5+a6/oY3EIvwKZn6xIVp2p+KeckUjU4xF0pedHpk4R+GKLVZQtZu/pBc0nlXILYSXlEwTWFPsLrjyqkesw1cQ//0leJRCH+/crRZJ4jhZd6cz9+U/qcXR+cQeMVXmMm1Qi3uBCp3udc1KP8pS+ddVp2fD98WS5E2RygHCSUsQPhMqnpZDaQ490DxRgoRyKlUZdGNHcRrC+et+gvLQq7179KCNXoOCiAFsFD9oaTwCPYT8O2LhqIlbURNm/OUPFEc8dhcNrl4ZVca5TZPyal0rxly/Zz9Mw5Neb98RNYQvngBLJ82XYBjuJxxPP54cYU1oLcG294sVvuKwkx+qmURzjxNOQeNu3N8BWiyORdPlX3kA2ndOjDXdkOmwJd1WAKAGdmscMficoohPFkLnBr48QNiILLxNfVKdVEUhkxnfn/qDIKhQ4hUKgzBrzr/vfppPKTK2zdFRi42gpE+ssTDphiwlSCyaO1mBh2hsQnLg25wFpNKZTjufuGwbWF/zxyLG9yYd2Rx9ZMASyYmQvftMH9uK3f8y44nhm3zGQN/+1AdQsHjmXFHeV3HLfZYgCvEpkOsgv3mtH+fH2l3qmXyfM2a+lKZFf5cGT6P+vC36amsYuXtIBHaJvZccKsdxi2iS1YndQjbAglTuempdpV0m3lLS2N7hkL9A8oDoDq7EnsJ8OhZrAq434TdElMLJErrJN+zlliJIx/HSK99aJPwrmPPtcHdtvwKdOtVCrlxhZQgSFqoigHrM/j64nnEQfZyVTsWw4SnuirL7r6oe0R2DVCzzZMXg6AIqSNafdLNQrYOQKqAHysMXEzT/l4pteOnDFWjESyYLGtviuEEDRWtBigiS3vBSpoTw8SOAawoPoI0xbuJvQnbE3kTeuqU808Ft8aJoVK6rkj7rcH+2KdbpS1JBVoO+qqxM5l6qxROqWKoxj9nXZ6VZVTeHMoNybrKosCqcgM8Ee0rboyvcPpSHr0DSYW6T7Bn2bBP7se/UtGH0Xw4CFoDSC5F8pi+dRHq0GT32I+CWc8iLKyD7FPYF2x3EWsJOL45RNpumQB8TJI1au0j/a5VeDLsrjclanvpzeeCIO2AavqK44MZQo94MpKi1df22u2mb7S05GrLOTpLFCZPB2cJckq1lZLJMkqqGJ3gwXfYV7LWuePGJGwz1tvQa3+HP37GbfvBoumJhfdyVWLqYGfrO6dEcc5YzEfX9Jl2riFn3HrFqmcX7hGzDiiM/MhUlhaDFgC86r3rtFRnQwFGRSXFdftQfzdYlGy5Hlp3/KDuEaZ4fKgT2rOBklumOX5mwcU5yrfYeCEPCJ219pSajEexQk4W2sb5bk9mqJ97qc2dbVOf8V5VGjAoz6gmNq5FJuzUpb9jaXMrNIG6HnUl538u0vtHAfdfpt2Qyy9sywyhz1HQMzruHrrlSAQv/VKr01ldFfqYa+Gqbhv1PF1CmA9WW/dgtJRJUTgVAprFZOiniSgfHMh0spgFPKC1cG+8fYErn/hK/7way4RYc/MvZLkL2hwMrWc3hnFy94Lrw+zrJkdD3NoAT0TbzCzPfeokvm1nvam/oYJD+hl/ABEXOYhPZV//kosOTupkFyD546D6uGhporyqQggwuGUaBnAROeS8oskRIVbAYqhzwSHkuv+34frDQJIAlu0k232Wxu4W+LhfyXiv7ZAef0ARbh5uhHICj8EwaQk+lFkaR9EvlQRIAT6IqwbjLqm4z4JqO9OXVqkMhQwM2abvEwVR/SDARGtEZDKiLD47oahhjMQ2QH49kBURw/SGmkMlGYU1dQFqmKFEVqEiWJikRBpB5SDqkGFCsHi5rE4ci/h9Y4X71kMPpYgj9uHMpnlAieJ+PqSmb/Geid83nKBl6+zWrJ16okpNbzgedQH21UK1imO/X9IE2lggEhFFImngdeI5qOz4E5ceEM2BO1KHrXFJYMHOEDrlKPLChJnNuF55+QeR32+vXrXJhoAj15j1smY2z85L1NeyknZ5h1RllnIusUs04p69RWUGvlvwaCsaRw8+WcCn0+FxU/n4otGco9leA+EbxPHOAnkdvFzC7ldSuN1EWhonBWtLIiIlg55r0lRBLyDnh8PEBIpKD+ERNhGPMWZYUztgHQihdFdwXr2l0Zgww6Q8e6APvTokI7r9C9wFdt0e8NugMMzUP6gNKyRkfWGEsMx4CgiA9IZ2jzoq8KulDUO1q+QaPtqdDMCOYp5EqyukXd1/I9f211oTVyULuMfmpMK6kFqpKsJqJVEW5FxKqXjAc9kHFXxkzFx5LRA3fz60mKsQmTDJUHmh7RKWLT4pmkxSs3DzLHirtjxZotQcpdLAruhYxpgmOGobnsJXg8zpg5kJQEXAompTRRRgyPCaJobOqrJqKSFNYVB/UA6FYBVeEY7TjiJsvZOD+MUq6rRdYzRC/HCHk0xKV6FJZPGSgtBmZBaXWe8ha9/n1UsjZknmJzLFGuqxSrCOiQt1M1fV5wBTqQP7yTK8xyA13VhkHQThUtsGXe3EfeeOSrJg0CdCQKtjInhPajoJ/F0cc4ybzrMDjhlgxCPg7DeB70ZfCzr7qOSlGDyMZzueaROKGFd2QknMTRYHQD4hri8t4fX8tvcwyYl4nwLqHfEMH4oHx/NG5bGLDv/CeyWDkmUEQGE5Rl0IB1lfd9QSwYjp9EpMZ0LN5xswAlheZFesLuUKLQX5JFALPmVYOCJxoQF/QCVIIoHcXCcOd1Zl4of0WFYgvS6mz9e7QQukvePxT1oRceYIDd8rK8irsSzS5Ys/1pOIpuPpJN8jEJBkGCwedUIEq79RVIXaO4iXO20GZNI0cWN4DsHHcwNI4F+i2VtXLjp437DuQu4H1whP02SeIErchDfPrg+9OJF/n3IBfHZIyTaZ+Cjf9+GmajSRKjsRMn6BMQ0wODpGN7X7JpcHioGEU/zvZeuPjjEToenZyvuSk0rphpruq9aL5LjjOOSQga3Iy6IFtK/guX0Oehb9eZv3vs0RQQUFFI0rFdcq7eweARntKtlATCzikkAuxn0DSSSQxYwWWnFK777cICtGVbNQexxTnsOUZrqCmih2xd1P3cSqU+xf2jaLf10PPw1EcYJw5P+jvoiYDFhwY9Wn61nody/RV8DjpQJq+G4rlQJa6kDItrYzncbW+VC0wr1Ve2WMCc5jDVHy6qDmd1APHm1ItdcnI1V46vswB9Cufp/8Ol+15PLp2PMn+IspnSnhqMc8Cae/K3f1RX9cB1Oo4L/5vwv/VqG/52mq+3nV33dQtydx13G752sAiyp7NwsKfYwbFz5pw60BnoB3Sh6yCaiJ1AiuOzzxt3zY03/+7G6TJ1QL89tVeY2K9l4P//A0c9BRc=')
_np_tt_v10_m32x__CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def _np_tt_v10_m32x_m32x_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_tt_v10_m32x_m32x_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_tt_v10_m32x_m32x_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_tt_v10_m32x_m32x_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_tt_v10_m32x_m32x_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else _np_tt_v10_m32x_M32X_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_tt_v10_m32x_m32x_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
    """Inverse of m32x_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else _np_tt_v10_m32x_M32X_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_tt_v10_m32x_m32x_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_tt_v10_m32x_m32x_units(hi), _np_tt_v10_m32x_m32x_units(lo)]
    else:
        mats = [_np_tt_v10_m32x_m32x_units(w, layout)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_tt_v10_m32x_m32x_c12_bytes(v, decode)
        if not _np_torch.equal(_np_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(_np_tt_v10_m32x_m32x_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_tt_v10_m32x_M32X_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_tt_v10_m32x_M32X_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_tt_v10_m32x_m32x_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32x_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_tt_v10_m32x_m32x_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_tt_v10_m32x_m32x_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_tt_v10_m32x__build():
    name = 'm32x_pooler_' + _np_hashlib.sha256(_np_tt_v10_m32x__CUDA.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_tt_v10_m32x__CPP, cuda_sources=_np_tt_v10_m32x__CUDA, functions=['m32x_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def _np_tt_v10_m32x__make_plan(ext, weights, fmt):
    pack = _np_tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = _np_tt_v10_m32x_m32x_stream(pack, 1, *_np_tt_v10_m32x__CFG)
    if stream is None:
        return None
    return dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=4096, cfg=_np_tt_v10_m32x__CFG)

def _np_tt_v10_m32x__run(x, plan):
    out = _np_torch.empty((x.shape[0], 4096), dtype=_np_torch.bfloat16, device=x.device)
    rc = _np_tt_v10_m32x_m32x_launch_raw(plan['ext'], plan['idx'], plan, x, out, 4096, _np_tt_v10_m32x__CFG)
    if rc:
        raise RuntimeError(f'm32x launch failed: {rc}')
    return out
_np_tt_v10_m32x__W_CUDA = _np__zsrc('eNrtO2lT40iW3/kVGjogJCvbWAeGwpZ7uWqioilDATPEDsEqZCuNVehCkrENzX/f9zJ1pGxBV/V0z2zvLtWgp8zUu/IdmS+zf3DpxAupdHKs2yenx+cnp9LuxsbGD1449mculfrjmeu0p4OVFjuZhZkX0Iae0UTr1pvTzPXCjLUV1CTbdumTN6a2DeAkSsbUC33osu2NE2kGww3dzqTA0Bf23HZcN5GfIs9tSbEivUgJzWZJWA6TbXv8lDn2PQ1p4o3tLLLTqZNQV44VpSe9Akb8uMA2d7xs5CRySWVEAGKYNyTJSQMY7TuZ51N586Wd0HupHQMyKe5JNwdSAJ8mHk3aWbJkqNqxk3jZss1JHhyMM6c96ppSTG63OndkS+tJ//GXWBoljnTTe92UDg6kzWRTHikEH0AV3gMaRMlyU+ltrDKLxJ6owCvKX+ex5IiPXWPEZoz0BMo1kmv6gVlLM5rY6TIcy+vkCmr5sIJqQn3qpLTt+N59SN2etDqO6coZP868pBqFTK0xg8J2zWr6XZqOKw04KxaAI2XZ2e4sjAn8KIOBqUi/SHLZ1zWVft/Q640atHX1BumzwKlouWQchWkmcdsLnJhZygP7m0RzIphQyXQc+U2WNI7bDmq0PZr5D+2MhmmUtHW3mq1cTfd+NHL8djGp0BEFsU8zameLg4PRMqNp+0yHZmc8pfYUOWF2drulkZctnWwZr/higuXtVnPugrX5mzJIwM3ugT9ABg6MeD/jfc0eUTSz0I/vzmWuFNYMHlkT3RsvufT8o6femiJ8t5ARlHGmHRyEke34fjR2MroiWfvJbM8MXXrZ6oAfES6dKJy0aQHzT+0FsM+hZQk9l9CcyYTilVIimwrylhvSUyWpGHjiJMgEy1udbwamlbzcZnogr7yJn7ZHwLzAe2/zgLHkKgeodKfSPTxSYCjnxl2PAy4d25HvVsycdUhF80wTXo4F+FocdS2MajE/Ar6ruWRjxoFkScfSttRZ7OU/s976sKkfwThRS0gJCYwDpWn81HtzvDQYSFqXfeXcdu5WxqGcQI0AQ7ua2ZkxN5aPpX5f0sCnkdH9TqeDv7Mch/YOjj2jq4s49ptR6GsogFUQookNsxmF8Q6KiovjtW9fNxoMMZj5oh1W/enY8emaBXKng9Qw89tJ2MaUvGCWKIEpSlt65Tsu8w1ujOzJEYq+4b7Fk5M+2CPPSeuxuZkTP4qNwh0YEyD03mSCvwibIDz+IkydRvaaWFrxkN/bO37wJuLCyLKkDvCw4pFICikA4sKmHeD1B+qvfa29+zVowTB1E62LveyZXXPXNGc5vjhx7gNHmoVJ5PuACBZNkozh/xHsrNODB5hiT1LVRwWs71GwPjQebEEGOWcpXZVtYEmmOHNTr5y6dDppQ/pOnLg2gfhrVjM19YqpOlZKIutxAMMA+g34fO43nT3+r9Gl+L8ZQxi63uTtGHFcueY7UeC48r73HN1bRWa8PazA9l0T1GRbuvTLLytNOCVSw2yueSAOUWrzK3zIerlCc5cDXaswWNDqawGu+VXsZOOpLHhGlfpWAw+slS2ZDgb7yrahzcgigbdtTYep7fflfVXGvj1lW5sprTzgNymN6SyxOr2kr3V7iaoqkjeREbeVKM5tcvdf1iJpCABB4GhdeeJHDnDpklWO2QoFl8/15cj8Hr5rT2g4hpUzrs/KBSzvgV+bL9zE3nbQNUOt+wBbnYmhswDL/tSWKmTLhHUK2eqSrb1XAl37ZOsD9OII7RXjsEkkrfiv80dQ1KBVg2YN2rVdRnP3D6cJkLYHvyCu9oHR7P7RNHXo07ETenWD0dxrogmL6cDL7PskmsWN0427lIZuwMAinTqBnAQBCJZsHNQqUK9AowLNCtytwG4F7t2VOY4jZpBWQnoJGSVkltBuCXVLaK+E9kvoQ4VZIFJR0SoyWkVHqwhpjBKsnkdqpwD0AjALoFvfPryuu6ih/ytd1NC/3ZBIzT/Juu80mbahf6dp/84crRu+YfybOQIQggw6PTgh48j8N3MEI3QYYsAQgwVeY/fPExwKcL8CPwgkRHICPU0gqAkUNYEkuvT/pdDTNf+Voadr/l5GTFbiDlnxerLic2Td4l8wUGFoQGcE80cnMPe/0y3/p0kEXQZ0GdBlsGBsfvhzS2QCaMInJnxismC+2/mTSwRDTBhiwhCTJYNd7f9Dr5uHtBwWiGoCVU0gqwl0dYGuLsop0NUFurpAVxfo6gJdXaCrC3R1ga4h0DW0/9WpYyOjQQw5gPZxL3o0ZAX/S/b35mf2uPjIHl/IKIp86RhcB1+PDy8GG7bN69vsaMt3ZuF4ao+iWeimti1r+n4LUMBARUhPwTy1nV2n49A9faTtdu17GgQy28PzAgrgTDzXZi9OmAHq47/xM4TPTixlC1Irydt2QtMs8cY4cE4EPFXaq42h6ZjYdvhkj1iC1LotKZplTKSA/Q2Fg4809r2sKgJkHQHWeEWALjKahECDn24wTTCPRQ10dFNhjOxj/TK4vSvrxXQRJxJT9Y11uQPhZ3ht5Qq7vLAuVZNcHVlHwxa0kZsrS/7Skq+O1H1FBZyGsoOYW/iHDM8smJKfjAMTcSPGzHOtbJpQx/3kLtoLMr+3oGkHMc2dJMYXiH/7O5AmfSek7N3Qe2J1Y+SktGOJR5JpoBBstViXOr9v3VyRycz384abn6EBm7+09gtGEid8sEZ+NH5gjGxxbYaZ0LbD2wKhbVl8/jDqWPLDTtdUWjKiAt0AAWVHZt/Am0LCh1HDEFUTB/0IeArh+NHNnMSL9cOaMWgso/zApD1JnHHmRaHjs3MaPL/N7ImXpBk7Z4Qso7U77IwDD1nmvGL7vfh8pxndgqPzJnI+VZbV4cZWFY1Sq9NL+196qaoqb5yReqGXrZ2QsgM0rTotwylU09b+ylkYK6nV0PKFYYHcZsiLg9D8QG/tiJMV2yT0Dchj3CRTmatqlkUS+J+7sG6375hEXiFiTW5pe1vy+jDPRS+3jdTytr708ob3z4jB0ygoO1u8cVqMx9br6mBB9eqoQS2SeHyKxg/jr47INkQmGUxN9ZRW1wSLbkEsLdCBweVfoz5e3y0HijPLNCSnSs05wTM7i0n+MyNUq72Gi0x87yFKGlud3KIgVqBKMRD0L28Kpa5GTOpbECdVWQ6zVuFIquCDrcsbFVGgqFwuYIr6txhP7nrAUQ6rhn7XQ44wPE58fs5eY75DOqWd8HPUxJnfXny8ux2e3VkvrzVjKW0l9SMeskWjYWbSIJwoHmSMNBlbcxDNj8J7BQXkAUTN566UbHjWMnQVpSjmfW3KRIf0QcV+f3jW82HeUAbk8e7Wv7OE02SgrfqAlkDI+FZ7wLmL+xcfe3FuEHJMYvY1S2CSMx7fHg139Lt3sXwFLF/7OK73dT1kRE/FeQePQrDwQbRf75SG0jU72ynweiNA7I1Q9fBQrYuPhdobtNUoUs2vPQuQxD2czYGFXj+CoPFQTEB1+IYryaLVm1QpFe27wigeQck4J/EdrOAgJVbwsoB1bIcEn2nEKT39DQTPAoK5gGBZIID11nsoNIEHrc7Dc4li/9dQPAsoRC7mJQrQRYHjVcKjEkEzjca8as4ms2bhKzxoMVtg1Dk5HyTpsSZVExuXeaMuNj7njYbYOK+kfN2oP3Nj91Qwqm+Z7PkUrVmGeDMYaPuKZZW3pjylLgTYF1v6WDCg+MBgJzmKUjsLcjCeChMhSapKY/G9CstPFo37hv4T7RxQTRzybgB8IjQGwjUawB6NBxZEpdVo/qau1pLiyg2wMhPJ3s4XZbui925QezdqQKBQq0BxcLCWJ2tzdTS0LK2r1A6w4FPiEPHqU5lPlbexGLpSq7H/Niyg3Fq57Juw/IHK4qneU7+IiaF58ZRfybIwT7W0LstROybsi3xLbzHD3jKVb8sG+4zVX4/YU/hg2td701rAnsjIwsCCXAvKzbxwVuZKOa8/KrBxuJVhX3A0VIFBdb/1FXLshQr8Azy9s1AjZuurqsPLN3wMa/u3PocI9Ouqm/CNwUBTGq8A9gQlPeLOiK96hte9R1BXS77cAc0+qtbwmq+LyuUH7ruiBxpajztsEMEJkh+32IvSyldJQJ4tDFU2dmAFK3pjYptSOguswHmgNn+XIS2TYqEkZFGWRuN+sUADeDWdplYc70DPav5MwPiyYi8ny3G8Bfu3gq1CwSXXNbWtrByc2Fm71zcTroIV98A4RX4ZLIfLq2CFH3Hhn5qX9f76/cEnE6t/DVfmijVMfltuUt6Wm5S35Sb8tpzIz7pPwjy0F6oFaHoILhFcMvAZwWcGzhGcV5aHdieUF3TwbIj+7C3Vy1Y7CWWGnjDMCpl674x6ZqOKNWOLXVHQW4oczTI1X8gKdqW0QhXWtZdsErkhsQ/kVnW3Qdn2I1J7n3rlQvzXveR1Y2NjlnrhvXQajiOXfgwtWJz42TKm8vZ4dl3Ua3jvNcygC1+xIktxnyOWhbpOS8qC2gXUOEuKq6epUJAZRQtu4WkGdjEuqUuUAVYIaQ4+zV39L7wVvsDr2ie4HUxOwyxZXkSA7MuMJstLms588PUejvgrzVYHHS3/TpMUdu6woW8Ua5Pw69qgQU6NaDre7kJ8p6Ez8ukJnThAg2w/Yr1NKES4XpDewianvKn7oJQHNHhfNVVeCdatXLoyqqW/9qpaDXbCjrNc7KCOXgllqDWivQqXyXC5w7mUQd/Hf7OvT4dX55f258ML++Tw+tC+/s+LU/vo49n54bXWJXoum4Kzgdzm7JARoF/5/NPw+vTy7PTw76f28Hx4utJ7dfPpH/84O7VhU3+00nWm2xeX55/Prz+dD/FF3+2uDmH82OfnR/bHT2dnDL+SG+E/U70szXGtJpmXMjNZNMlFzUDntTdK+PNbSols7jNnxEzkKoPsFGB1MM8l9WIn277DhlXwGiw0LEhAHsjRUMGdUjJW8ulNxuv1RceC0C5j/ZBXExtqiAoZWSz+Q36DnUvLJClEQkt2BqOfnIORgsW9vKzHtuIPNAmtt4u5kCjJJQH1g+q/ENQ4artXOS2bCUzKS2viwK6k8FbWVCQwEBqtFXX0ESbjimaHGRjfaJZRGRkgRU/Z/NlZnCxDJ/DGVyxRfGbR/Mp7pkwepVFZv4XQMAovoiRD3z7mcRGJHPp+NKcu0d4kxCTOEp7sMRIg5jNmasdROPHu8VLy5B6rHvBoYxn8xAsscDwDNik7lwpfcxA5UI+GP2rKDhoAjmS103IoLyPzHjdXCMh/hPfpLdQE60mZ4Vlp1qsxUgopOQAhK/hse67VMCaXHiiDyUKM5GOfHL+oCCJPLzPowwMsnlMUUsQkIcu8IEf4cWpt44NxGM6CQ9aksbAphLCKlZ9hgqh/uoCkM7knbLqwDicWfJS5+M6THSVy/QxAQccNSAheVa/5yzDNSu1tMIANCA8/Qj7jEYOXRNzKzRcVOK9AWoHfFTDKlyJYpHMPd6meiykOFnNS56DQ0/uBDfapBHKGCSFeJ2iSRBvIebBfFFF/XgC0AFa1hEzJYhhTUtwtM1a038IKiwe/Jy+C5fyorcyZF06ifMaUlzUvT6WJg/Zf6bhQcTH0r0KoSOXtSb59bA6KDQpXenl1K9fXP4230F6B+DWXfOKgM13S+xTiPfyo0ID/F4qPcYvFBdDTfwMynA59')
_np_tt_v10_m32x__W_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_w_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint m32x_w_info(int);\n'

def _np_tt_v10_m32x__w_c12_bytes(v):
    return _np_tt_v10_m32x_m32x_c12_bytes(v, 5)

def _np_tt_v10_m32x__w_c12_unpack(data, base, esc):
    return _np_tt_v10_m32x_m32x_c12_unpack(data, base, esc, 5)

def _np_tt_v10_m32x__w_table(base):
    return base << 8 | base << 24

def _np_tt_v10_m32x__w_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != _np_torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = _np_torch.full((n // r * total * rw, 64), -1, dtype=_np_torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = _np_tt_v10_m32x__w_c12_bytes(v)
        if not _np_torch.equal(_np_tt_v10_m32x__w_c12_unpack(data, b, entries), v):
            return None
        table = _np_tt_v10_m32x__w_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = _np_torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = _np_torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(_np_torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = _np_torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = _np_torch.arange(len(gs), device=w.device) - _np_torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = _np_tt_v10_m32x_m32x_to_i32(entry)
    else:
        data = _np_tt_v10_m32x_m32x_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def _np_tt_v10_m32x__w_launch_raw(ext, idx, pk, x, out):
    return ext.m32x_w_launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], _np_torch.cuda.current_stream().cuda_stream)

def _np_tt_v10_m32x__w_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = _np_torch.eye(k, device=w.device, dtype=_np_torch.bfloat16)
    out = _np_torch.empty((k, n), device=w.device, dtype=_np_torch.bfloat16)
    for start in range(0, k, m):
        if _np_tt_v10_m32x__w_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(_np_torch.equal(out.view(_np_torch.int16), w.T.contiguous().view(_np_torch.int16)))

def _np_tt_v10_m32x__w_build():
    name = 'm32x_unpooler_' + _np_hashlib.sha256(_np_tt_v10_m32x__W_CUDA.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_tt_v10_m32x__W_CPP, cuda_sources=_np_tt_v10_m32x__W_CUDA, functions=['m32x_w_launch', 'm32x_w_info'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def _np_tt_v10_m32x__w_make_plan(ext, weights, fmt):
    pack = _np_tt_v10_m32x__w_pack_weight(weights, 64, 4, 1, fmt == 1)
    if pack is None:
        return None
    return dict(pack, ext=ext, idx=0 if fmt == 1 else 1, fmt=fmt)

def _np_tt_v10_m32x__w_run(x, plan):
    out = _np_torch.empty((x.shape[0], 8192), dtype=_np_torch.bfloat16, device=x.device)
    rc = _np_tt_v10_m32x__w_launch_raw(plan['ext'], plan['idx'], plan, x, out)
    if rc:
        raise RuntimeError(f'm32x unpooler launch failed: {rc}')
    return out

def _np_tt_v10_m32x__w_verify_random(plan, weights):
    for m in (9, 16, 17, 24, 32):
        x = _np_torch.randn((m, 4096), device=weights.device, dtype=_np_torch.bfloat16)
        x[-1].zero_()
        x[-1, m * 131 % 4096] = 16
        ref = (x.double() @ weights.double().T).to(_np_torch.bfloat16).double()
        out = _np_tt_v10_m32x__w_run(x, plan).double()
        diff = out - ref
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > 0.0025:
            return False
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if float(rows.max()) > 0.01 or float(rows[-1]) > 0.00025:
            return False
    return True
_np_tt_v10_m32x__Q_CFG = (1, 4, 8, 4)
_np_tt_v10_m32x__Q_CUDA = _np__zsrc('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')
_np_tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def _np_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_tt_v10_m32x__q_build():
    name = 'm32x_qkv_' + _np_hashlib.sha256(_np_tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_tt_v10_m32x__Q_CPP, cuda_sources=_np_tt_v10_m32x__Q_CUDA, functions=['m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def _np_tt_v10_m32x__q_make_plan(ext, weights, fmt):
    pack = _np_tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = _np_tt_v10_m32x_m32x_stream(pack, 1, *_np_tt_v10_m32x__Q_CFG)
    if stream is None:
        return None
    plan = dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=2304, cfg=_np_tt_v10_m32x__Q_CFG)
    x = weights[:32]
    out = _np_torch.empty((32, 2304), dtype=_np_torch.bfloat16, device=weights.device)
    capacity = _np_tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, _np_tt_v10_m32x__Q_CFG, 1)
    attr = _np_tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, _np_tt_v10_m32x__Q_CFG, 2)
    if capacity < 144 or attr < 0 or attr % 1000:
        return None
    return plan

def _np_tt_v10_m32x__q_run(x, plan):
    out = _np_torch.empty((x.shape[0], 2304), dtype=_np_torch.bfloat16, device=x.device)
    rc = _np_tt_v10_m32x__q_launch_raw(plan['ext'], plan['idx'], plan, x, out, 2304, _np_tt_v10_m32x__Q_CFG)
    if rc:
        raise RuntimeError(f'm32x qkv launch failed: {rc}')
    return out

def _np_tt_v10_m32x_prepare_m32x(weights, *, include_unpooler: bool=False) -> dict:
    state = {'plans': {}, 'device': None}
    configs = [('pooler', (4096, 8192), _np_tt_v10_m32x__build, _np_tt_v10_m32x__make_plan), ('qkv_gate', (2304, 8192), _np_tt_v10_m32x__q_build, _np_tt_v10_m32x__q_make_plan), ('unpooler', (8192, 4096), _np_tt_v10_m32x__w_build, _np_tt_v10_m32x__w_make_plan)]
    for layer, shape, build, make_plan in configs:
        if layer == 'unpooler' and (not include_unpooler):
            continue
        w = weights.get(layer)
        if w is None or w.dtype != _np_torch.bfloat16 or tuple(w.shape) != shape or (not w.is_cuda):
            continue
        if _np_torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        if state['device'] is not None and state['device'] != w.device:
            continue
        state['device'] = w.device
        ext = build()
        for fmt in (1, 0):
            plan = make_plan(ext, w, fmt)
            if plan is None:
                continue
            if layer == 'pooler':
                ok = _np_tt_v10_m32x_m32x_verify_onehot(ext, plan['idx'], plan, w, _np_tt_v10_m32x__CFG, 32) and _np_tt_v10_m32x_m32x_verify_random(ext, plan['idx'], plan, w, _np_tt_v10_m32x__CFG, [9, 16, 17, 24, 32])
            elif layer == 'qkv_gate':
                ok = _np_tt_v10_m32x__q_verify_onehot(ext, plan['idx'], plan, w, _np_tt_v10_m32x__Q_CFG, 32) and _np_tt_v10_m32x__q_verify_random(ext, plan['idx'], plan, w, _np_tt_v10_m32x__Q_CFG, [9, 16, 17, 24, 32])
            else:
                ok = _np_tt_v10_m32x__w_verify_onehot(ext, plan['idx'], plan, w, 32) and _np_tt_v10_m32x__w_verify_random(plan, w)
            if ok:
                state['plans'][layer] = plan
                break
    return state

def _np_tt_v10_m32x_run_m32x(x, layer, n, st) -> _np_torch.Tensor | None:
    expected = {'pooler': (4096, 8192), 'qkv_gate': (2304, 8192), 'unpooler': (8192, 4096)}.get(layer)
    if expected is None or x.ndim != 2 or x.shape[0] != 32:
        return None
    if (n, x.shape[1]) != expected or x.dtype != _np_torch.bfloat16 or (not x.is_cuda):
        return None
    if x.device != st.get('device') or not x.is_contiguous():
        return None
    plan = st.get('plans', {}).get(layer)
    if plan is None:
        return None
    if layer == 'pooler':
        return _np_tt_v10_m32x__run(x, plan)
    if layer == 'qkv_gate':
        return _np_tt_v10_m32x__q_run(x, plan)
    return _np_tt_v10_m32x__w_run(x, plan)
_np_V10_ROUTES = [{'m': 8, 'layer': 'pooler', 'name': 'smallm2', 'module': 'v10_sm2'}, {'m': 16, 'layer': 'qkv_gate', 'name': 'warps', 'module': 'v10_warps'}, {'m': 256, 'layer': 'o', 'name': 'm256tune', 'module': 'v10_tune', 'config': [256, 64, 5, 7, 1, 1, 1]}, {'m': 32, 'layer': 'qkv_gate', 'name': 'm32x', 'module': 'v10_m32x'}, {'m': 32, 'layer': 'pooler', 'name': 'm32x', 'module': 'v10_m32x'}]
_np_V10_SELECTED_ROUTES = [r for r in _np_V9_SELECTED_ROUTES if (r['m'], r['layer']) not in {(v['m'], v['layer']) for v in _np_V10_ROUTES}] + _np_V10_ROUTES

def _np_prepare_v10_tune(weights: _np_Weights) -> dict:
    weight = weights.get('o')
    if weight is None or not weight.is_cuda or weight.dtype != _np_torch.bfloat16 or (tuple(weight.shape) != (8192, 1024)):
        return {}
    if _np_torch.cuda.get_device_capability(weight.device) != (9, 0):
        return {}
    return {'ext': _np_tt_v10_tune_build_rl([(256, 64, 5, 7, 1, 1, 1)]), 'weight': _np_tt_v10_tune_pack_weight(weight.contiguous(), 64)}

def _np_run_v10_tune(x: _np_torch.Tensor, layer: str, n: int, state: dict) -> _np_torch.Tensor | None:
    if layer != 'o' or n != 8192 or (not state) or (x.ndim != 2) or (tuple(x.shape) != (256, 1024)):
        return None
    if x.dtype != _np_torch.bfloat16 or not x.is_cuda or (not x.is_contiguous()) or x.data_ptr() % 16 or (x.device != state['weight'].device):
        return None
    output = _np_torch.empty((256, 8192), device=x.device, dtype=_np_torch.bfloat16)
    rc = state['ext'].rl_launch(0, x.data_ptr(), state['weight'].data_ptr(), output.data_ptr(), 256, _np_torch.cuda.current_stream(x.device).cuda_stream, 0)
    if rc:
        raise RuntimeError(f'm256tune O launch failed: {rc}')
    return output

def _np_prepare_v10(weights: _np_Weights) -> dict:
    state = _np_prepare_v9(weights)
    state['v10_sm2'] = _np_tt_v10_sm2_prepare_smallm2(weights)
    state['v10_warps'] = _np_tt_v10_warps_prepare_warps(weights)
    state['v10_tune'] = _np_prepare_v10_tune(weights)
    state['v10_m32x'] = _np_tt_v10_m32x_prepare_m32x(weights, include_unpooler=False)
    for route in _np_V10_ROUTES:
        m, layer = (route['m'], route['layer'])
        weight = weights[layer]
        data = {'x': _np_torch.zeros((m, weight.shape[1]), dtype=_np_torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': weight.shape[0], 'k': weight.shape[1], 'layer': layer}}
        _np_candidate_v10(data, state)
    _np_torch.cuda.synchronize(weights['qkv_gate'].device)
    return state

def _np_candidate_v10(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    m = x.shape[0]
    output = None
    if layer == 'pooler' and m == 8:
        output = _np_tt_v10_sm2_run_smallm2(x, layer, n, state['v10_sm2'])
    elif layer == 'qkv_gate' and m == 16:
        output = _np_tt_v10_warps_run_warps(x, layer, n, state['v10_warps'])
    elif layer == 'o' and m == 256:
        output = _np_run_v10_tune(x, layer, n, state['v10_tune'])
    elif layer in ('qkv_gate', 'pooler') and m == 32:
        output = _np_tt_v10_m32x_run_m32x(x, layer, n, state['v10_m32x'])
    return _np_candidate_v9(data, state) if output is None else output
_np_V11_ROUTES = [{'m': 32, 'layer': 'unpooler', 'name': 'm32x_unpooler', 'module': 'v11_m32x'}]
_np_V11_SELECTED_ROUTES = [r for r in _np_V10_SELECTED_ROUTES if (r['m'], r['layer']) != (32, 'unpooler')] + _np_V11_ROUTES

def _np_prepare_v11(weights: _np_Weights) -> dict:
    state = _np_prepare_v10(weights)
    state['v11_m32x'] = _np_tt_v10_m32x_prepare_m32x({'unpooler': weights['unpooler']}, include_unpooler=True)
    weight = weights['unpooler']
    data = {'x': _np_torch.zeros((32, weight.shape[1]), dtype=_np_torch.bfloat16, device=weight.device), 'config': {'m': 32, 'n': weight.shape[0], 'k': weight.shape[1], 'layer': 'unpooler'}}
    _np_candidate_v11(data, state)
    _np_torch.cuda.synchronize(weight.device)
    return state

def _np_candidate_v11(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    if layer == 'unpooler' and x.shape[0] == 32:
        output = _np_tt_v10_m32x_run_m32x(x, layer, n, state['v11_m32x'])
        if output is not None:
            return output
    return _np_candidate_v10(data, state)
'Lossless BF16 small-token projection routes validated on H200.'
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE = _np__zsrc('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')
_np_tt_v12_smtrace1_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_tt_v12_smtrace1_SMTRACE_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_np_tt_v12_smtrace1_SMTRACE_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_tt_v12_smtrace1_SMTRACE_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_np_tt_v12_smtrace1_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_tt_v12_smtrace1_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_tt_v12_smtrace1_SMTRACE_ESC_CAP = 32

def _np_tt_v12_smtrace1_smtrace_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _np_tt_v12_smtrace1_smtrace_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _np_tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _np_tt_v12_smtrace1_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _np_load_inline(name, _np_tt_v12_smtrace1_SMTRACE_CPP + extra_cpp, cuda_sources=_np_tt_v12_smtrace1_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_tt_v12_smtrace1_smtrace_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_tt_v12_smtrace1__smtrace_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_tt_v12_smtrace1__smtrace_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_tt_v12_smtrace1_smtrace_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_tt_v12_smtrace1__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_tt_v12_smtrace1_smtrace_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_tt_v12_smtrace1_SMTRACE_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_tt_v12_smtrace1__smtrace_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_tt_v12_smtrace1_smtrace_c12_unpack(data, base, esc):
    """Inverse of smtrace_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_tt_v12_smtrace1_SMTRACE_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_tt_v12_smtrace1_smtrace_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_tt_v12_smtrace1_smtrace_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_tt_v12_smtrace1_smtrace_units(hi), _np_tt_v12_smtrace1_smtrace_units(lo)]
    else:
        mats = [_np_tt_v12_smtrace1_smtrace_units(w)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_tt_v12_smtrace1_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_tt_v12_smtrace1_smtrace_c12_bytes(v)
        if not _np_torch.equal(_np_tt_v12_smtrace1_smtrace_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_tt_v12_smtrace1_smtrace_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_tt_v12_smtrace1_smtrace_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_tt_v12_smtrace1_smtrace_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_tt_v12_smtrace1_smtrace_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_tt_v12_smtrace1_SMTRACE_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_tt_v12_smtrace1_SMTRACE_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_tt_v12_smtrace1__smtrace_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_tt_v12_smtrace1_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_np_tt_v12_smtrace1_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_np_tt_v12_smtrace1_SMTRACE_MAX_M = 32
_np_tt_v12_smtrace1_SMTRACE_TABLE_M.update({('router', 8): _np_tt_v12_smtrace1_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): _np_tt_v12_smtrace1_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): _np_tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): _np_tt_v12_smtrace1_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): _np_tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): _np_tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 1]})
_np_tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_np_tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_np_tt_v12_smtrace1_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_np_tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _np_tt_v12_smtrace1_smtrace_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_tt_v12_smtrace1_smtrace_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_tt_v12_smtrace1_smtrace_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_tt_v12_smtrace1_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_tt_v12_smtrace1_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_tt_v12_smtrace1_smtrace_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _np_tt_v12_smtrace1_smtrace_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _np_tt_v12_smtrace1_smtrace_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_tt_v12_smtrace1_smtrace_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _np_tt_v12_smtrace1_smtrace_verify_onehot(ext, inst_id[inst], sp, _np_tt_v12_smtrace1_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _np_tt_v12_smtrace1_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _np_tt_v12_smtrace1_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
    table_m = dict(_np_tt_v12_smtrace1_SMTRACE_TABLE_M if table_m is None else table_m)
    table_mt = dict(_np_tt_v12_smtrace1_SMTRACE_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = _np_tt_v12_smtrace1_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, _np_tt_v12_smtrace1_SMTRACE_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if _np_tt_v12_smtrace1_smtrace_bf16_twin(inst):
            instances.append(_np_tt_v12_smtrace1_smtrace_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = _np_tt_v12_smtrace1_smtrace_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = _np_tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and _np_tt_v12_smtrace1_smtrace_bf16_twin(inst):
            plan = _np_tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, _np_tt_v12_smtrace1_smtrace_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def _np_tt_v12_smtrace1_smtrace_run_core(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= _np_tt_v12_smtrace1_SMTRACE_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _np_tt_v12_smtrace1_smtrace_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != _np_torch.bfloat16 or (x.shape[1] != _np_tt_v12_smtrace1_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = _np_torch.empty((m, n), dtype=_np_torch.float32 if layer == 'router' else _np_torch.bfloat16, device=x.device)
    if _np_tt_v12_smtrace1_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
_np_tt_v12_smtrace1_SMTRACE_ROUTES = {('unpooler', 1): ((1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1, 8, 2, 4), (1, 8, 2, 4)), ('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def _np_tt_v12_smtrace1_prepare_smtrace(weights):
    usable = {}
    for (layer, m), (inst, cfg) in _np_tt_v12_smtrace1_SMTRACE_ROUTES.items():
        w = weights.get(layer)
        if w is not None and w.is_cuda and (w.dtype == _np_torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (_np_torch.cuda.get_device_capability(w.device) == (9, 0)):
            usable[layer] = w
    routes = {key: value for key, value in _np_tt_v12_smtrace1_SMTRACE_ROUTES.items() if key[0] in usable}
    return _np_tt_v12_smtrace1_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_3c2c50fe46c8')

def _np_tt_v12_smtrace1_run_smtrace(x, layer, n, st):
    if (layer, x.shape[0]) not in _np_tt_v12_smtrace1_SMTRACE_ROUTES:
        return None
    return _np_tt_v12_smtrace1_smtrace_run_core(x, layer, n, st)
_np_V12_SM2_O1 = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1))}

def _np_prepare_v12_o1(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in _np_V12_SM2_O1.items():
        w = weights.get(layer)
        if w is None or w.dtype != _np_torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or _np_torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, _np_tt_v10_sm2_smallm2_bf16_twin(inst)))))
    ext = _np_tt_v10_sm2_smallm2_build(instances, 'smallm2_o1_v12')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = _np_tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = _np_tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, _np_tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st
_np_V12_ROUTES = [{'m': 1, 'layer': 'unpooler', 'name': 'smtrace1', 'module': 'v12_smtrace1'}, {'m': 1, 'layer': 'o', 'name': 'smallm2_o1', 'module': 'v12_o1'}]
_np_V12_SELECTED_ROUTES = [r for r in _np_V11_SELECTED_ROUTES if (r['m'], r['layer']) not in {(n['m'], n['layer']) for n in _np_V12_ROUTES}] + _np_V12_ROUTES

def _np_prepare_v12(weights: _np_Weights) -> dict:
    state = _np_prepare_v11(weights)
    state['v12_smtrace1'] = _np_tt_v12_smtrace1_prepare_smtrace(weights)
    state['v12_o1'] = _np_prepare_v12_o1(weights)
    for route in _np_V12_ROUTES:
        m, layer = (route['m'], route['layer'])
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        data = {'x': _np_torch.zeros((m, weight.shape[1]), dtype=_np_torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
        _np_candidate_v12(data, state)
    _np_torch.cuda.synchronize()
    return state

def _np_candidate_v12(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    if layer == 'unpooler' and x.shape[0] == 1:
        output = _np_tt_v12_smtrace1_run_smtrace(x, layer, n, state['v12_smtrace1'])
        if output is not None:
            return output
    if layer == 'o' and x.shape[0] == 1:
        output = _np_tt_v10_sm2_run_smallm2(x, layer, n, state['v12_o1'])
        if output is not None:
            return output
    return _np_candidate_v11(data, state)
'Lossless BF16 small-token projection routes validated on H200.'
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE = _np__zsrc('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')
_np_tt_v13_smtrace8_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_tt_v13_smtrace8_SMTRACE_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_np_tt_v13_smtrace8_SMTRACE_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_tt_v13_smtrace8_SMTRACE_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_np_tt_v13_smtrace8_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_tt_v13_smtrace8_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_tt_v13_smtrace8_SMTRACE_ESC_CAP = 32

def _np_tt_v13_smtrace8_smtrace_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _np_tt_v13_smtrace8_smtrace_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _np_tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _np_tt_v13_smtrace8_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _np_load_inline(name, _np_tt_v13_smtrace8_SMTRACE_CPP + extra_cpp, cuda_sources=_np_tt_v13_smtrace8_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_tt_v13_smtrace8_smtrace_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_tt_v13_smtrace8__smtrace_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_tt_v13_smtrace8__smtrace_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_tt_v13_smtrace8_smtrace_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_tt_v13_smtrace8__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_tt_v13_smtrace8_smtrace_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_tt_v13_smtrace8_SMTRACE_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_tt_v13_smtrace8__smtrace_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc):
    """Inverse of smtrace_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_tt_v13_smtrace8_SMTRACE_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_tt_v13_smtrace8_smtrace_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_tt_v13_smtrace8_smtrace_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_tt_v13_smtrace8_smtrace_units(hi), _np_tt_v13_smtrace8_smtrace_units(lo)]
    else:
        mats = [_np_tt_v13_smtrace8_smtrace_units(w)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_tt_v13_smtrace8_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_tt_v13_smtrace8_smtrace_c12_bytes(v)
        if not _np_torch.equal(_np_tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_tt_v13_smtrace8_smtrace_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_tt_v13_smtrace8_smtrace_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_tt_v13_smtrace8_SMTRACE_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_tt_v13_smtrace8_SMTRACE_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_tt_v13_smtrace8__smtrace_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_tt_v13_smtrace8_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_np_tt_v13_smtrace8_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_np_tt_v13_smtrace8_SMTRACE_MAX_M = 32
_np_tt_v13_smtrace8_SMTRACE_TABLE_M.update({('router', 8): _np_tt_v13_smtrace8_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): _np_tt_v13_smtrace8_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): _np_tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): _np_tt_v13_smtrace8_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): _np_tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): _np_tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 1]})
_np_tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_np_tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_np_tt_v13_smtrace8_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_np_tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _np_tt_v13_smtrace8_smtrace_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_tt_v13_smtrace8_smtrace_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_tt_v13_smtrace8_smtrace_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_tt_v13_smtrace8_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_tt_v13_smtrace8_smtrace_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _np_tt_v13_smtrace8_smtrace_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _np_tt_v13_smtrace8_smtrace_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_tt_v13_smtrace8_smtrace_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _np_tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id[inst], sp, _np_tt_v13_smtrace8_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _np_tt_v13_smtrace8_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _np_tt_v13_smtrace8_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
    table_m = dict(_np_tt_v13_smtrace8_SMTRACE_TABLE_M if table_m is None else table_m)
    table_mt = dict(_np_tt_v13_smtrace8_SMTRACE_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = _np_tt_v13_smtrace8_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, _np_tt_v13_smtrace8_SMTRACE_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if _np_tt_v13_smtrace8_smtrace_bf16_twin(inst):
            instances.append(_np_tt_v13_smtrace8_smtrace_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = _np_tt_v13_smtrace8_smtrace_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = _np_tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and _np_tt_v13_smtrace8_smtrace_bf16_twin(inst):
            plan = _np_tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, _np_tt_v13_smtrace8_smtrace_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def _np_tt_v13_smtrace8_smtrace_run_core(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= _np_tt_v13_smtrace8_SMTRACE_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _np_tt_v13_smtrace8_smtrace_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != _np_torch.bfloat16 or (x.shape[1] != _np_tt_v13_smtrace8_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = _np_torch.empty((m, n), dtype=_np_torch.float32 if layer == 'router' else _np_torch.bfloat16, device=x.device)
    if _np_tt_v13_smtrace8_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
_np_tt_v13_smtrace8_SMTRACE_ROUTES = {('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def _np_tt_v13_smtrace8_prepare_smtrace(weights):
    usable = {}
    for (layer, m), (inst, cfg) in _np_tt_v13_smtrace8_SMTRACE_ROUTES.items():
        w = weights.get(layer)
        if w is not None and w.is_cuda and (w.dtype == _np_torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (_np_torch.cuda.get_device_capability(w.device) == (9, 0)):
            usable[layer] = w
    routes = {key: value for key, value in _np_tt_v13_smtrace8_SMTRACE_ROUTES.items() if key[0] in usable}
    return _np_tt_v13_smtrace8_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_d46e5decdceb')

def _np_tt_v13_smtrace8_run_smtrace(x, layer, n, st):
    if (layer, x.shape[0]) not in _np_tt_v13_smtrace8_SMTRACE_ROUTES:
        return None
    return _np_tt_v13_smtrace8_smtrace_run_core(x, layer, n, st)
_np_V13_ROUTES = [{'m': 8, 'layer': 'unpooler', 'name': 'smtrace8', 'module': 'v13_smtrace8'}]
_np_V13_SELECTED_ROUTES = [r for r in _np_V12_SELECTED_ROUTES if (r['m'], r['layer']) not in {(n['m'], n['layer']) for n in _np_V13_ROUTES}] + _np_V13_ROUTES

def _np_prepare_v13(weights: _np_Weights) -> dict:
    state = _np_prepare_v12(weights)
    state['v13_smtrace8'] = _np_tt_v13_smtrace8_prepare_smtrace(weights)
    for route in _np_V13_ROUTES:
        m, layer = (route['m'], route['layer'])
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        data = {'x': _np_torch.zeros((m, weight.shape[1]), dtype=_np_torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
        _np_candidate_v13(data, state)
    _np_torch.cuda.synchronize()
    return state

def _np_candidate_v13(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    if layer == 'unpooler' and x.shape[0] == 8:
        output = _np_tt_v13_smtrace8_run_smtrace(x, layer, n, state['v13_smtrace8'])
        if output is not None:
            return output
    return _np_candidate_v12(data, state)
'QKV candidates. Production routes are enabled only after measured validation.'
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline

def _np_tt_v14_qkv_qbig_mma(n):
    regs = ','.join((f'%{i}' for i in range(n // 2)))
    outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
    return f'__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));\n    }}'
_np_tt_v14_qkv_QKVBIG_CUDA = _np__zsrc('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')

def _np_tt_v14_qkv_build_qkv(configs, profile=False):
    ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
    source = _np_tt_v14_qkv_QKVBIG_CUDA.replace('@@MMA@@', '\n'.join((_np_tt_v14_qkv_qbig_mma(n) for n in ns)))
    source = source.replace('@@CALL@@', ' '.join((f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);' for n in ns)))
    source = source.replace('@@SWAP@@', ' '.join((f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns)))
    source = source.replace('@@CASES@@', '\n'.join((f"case {i}: return qb_launch_t<{','.join(map(str, c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);" for i, c in enumerate(configs))))
    digest = _np_hashlib.sha256(source.encode()).hexdigest()[:16]
    namespace = 'qb_' + digest
    begin = source.index('#define QD')
    args = 'int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats'
    source = source[:begin] + f'namespace {namespace} {{\n' + source[begin:] + f'\n}}\nint qb_launch({args}) {{return {namespace}::qb_launch(id,w,x,out,m,stream,query,stats);}}'
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
    return _np_load_inline('qbig_' + _np_hashlib.sha256(source.encode()).hexdigest()[:16], cpp, cuda_sources=source, functions=['qb_launch'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def _np_tt_v14_qkv_qbig_pack(w, groups, kt=1):
    rows = (2304 + groups - 1) // groups
    if kt == 2:
        return w.view(groups, rows, 128, 64).permute(0, 2, 1, 3).contiguous().reshape(-1, 64)
    if 2304 % groups == 0:
        return w.contiguous()
    out = _np_torch.zeros((groups, rows, 8192), dtype=_np_torch.bfloat16, device=w.device)
    for g in range(groups):
        lo, hi = (g * 2304 // groups, (g + 1) * 2304 // groups)
        out[g, :hi - lo].copy_(w[lo:hi])
    return out.reshape(-1, 8192)

def _np_tt_v14_qkv_prepare_qbig_candidates(weights, configs, profile=False):
    w = weights['qkv_gate']
    return {'ext': _np_tt_v14_qkv_build_qkv(configs, profile), 'configs': list(configs), 'packed': {(g, kt): _np_tt_v14_qkv_qbig_pack(w, g, kt) for g, kt in {(c[1], 2 if c[5] & 32 else 1) for c in configs}}, 'device': w.device}

def _np_tt_v14_qkv_run_qbig_candidate(x, st, index, stats=None, query=0):
    out = _np_torch.empty((x.shape[0], 2304), dtype=_np_torch.bfloat16, device=x.device)
    cfg = st['configs'][index]
    rc = st['ext'].qb_launch(index, st['packed'][cfg[1], 2 if cfg[5] & 32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], _np_torch.cuda.current_stream().cuda_stream, query, stats.data_ptr() if stats is not None else 0)
    if query:
        return rc
    if rc:
        raise RuntimeError(f'qkv launch {cfg} failed: {rc}')
    return out
_np_tt_v14_qkv_EXPERIMENT_CONFIGS = [(64, 36, 128, 3, 8, 539, 5), (64, 36, 128, 3, 8, 6683, 5), (64, 36, 256, 3, 4, 219, 5)]
_np_tt_v14_qkv_ENABLED_ROUTES = {128: 0}

def _np_tt_v14_qkv_prepare_qkvbig(weights: dict) -> dict:
    w = weights.get('qkv_gate')
    if w is None or w.shape != (2304, 8192) or w.dtype != _np_torch.bfloat16 or (not w.is_cuda):
        return {'routes': {}}
    if _np_torch.cuda.get_device_capability(w.device) != (9, 0) or 'H200' not in _np_torch.cuda.get_device_name(w.device):
        return {'routes': {}}
    if not _np_tt_v14_qkv_ENABLED_ROUTES:
        return {'routes': {}}
    state = _np_tt_v14_qkv_prepare_qbig_candidates(weights, _np_tt_v14_qkv_EXPERIMENT_CONFIGS)
    state['routes'] = dict(_np_tt_v14_qkv_ENABLED_ROUTES)
    for m, index in state['routes'].items():
        x = _np_torch.zeros((m, 8192), device=w.device, dtype=_np_torch.bfloat16)
        _np_tt_v14_qkv_run_qbig_candidate(x, state, index)
    _np_torch.cuda.synchronize(w.device)
    return state

def _np_tt_v14_qkv_run_qkvbig(x: _np_torch.Tensor, layer: str, n: int, st: dict) -> _np_torch.Tensor | None:
    if layer != 'qkv_gate' or n != 2304 or x.ndim != 2 or (x.shape[1] != 8192):
        return None
    index = st.get('routes', {}).get(x.shape[0])
    if index is None or not x.is_cuda or x.dtype != _np_torch.bfloat16 or (not x.is_contiguous()):
        return None
    if x.device != st['device'] or x.data_ptr() % 16:
        return None
    return _np_tt_v14_qkv_run_qbig_candidate(x, st, index)
_np_V14_ROUTES = [{'m': 128, 'layer': 'qkv_gate', 'name': 'qkvbig', 'module': 'v14_qkv'}]
_np_V14_SELECTED_ROUTES = [r for r in _np_V13_SELECTED_ROUTES if (r['m'], r['layer']) not in {(n['m'], n['layer']) for n in _np_V14_ROUTES}] + _np_V14_ROUTES

def _np_prepare_v14(weights: _np_Weights) -> dict:
    state = _np_prepare_v13(weights)
    state['v14_qkv'] = _np_tt_v14_qkv_prepare_qkvbig(weights)
    for route in _np_V14_ROUTES:
        m, layer = (route['m'], route['layer'])
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        data = {'x': _np_torch.zeros((m, weight.shape[1]), dtype=_np_torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
        _np_candidate_v14(data, state)
    _np_torch.cuda.synchronize()
    return state

def _np_candidate_v14(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    if layer == 'qkv_gate' and x.shape[0] == 128:
        output = _np_tt_v14_qkv_run_qkvbig(x, layer, n, state['v14_qkv'])
        if output is not None:
            return output
    return _np_candidate_v13(data, state)
_np_V15_ROUTES = [{'m': 256, 'layer': 'unpooler', 'name': 'ltsearch_transposed', 'module': 'lt', 'descriptor': [66, 20, 1, 0, 0, 4, 35, 0, 2], 'layout': 3, 'epilogue': 0, 'algorithm_workspace_bytes': 0}]
_np_V15_SELECTED_ROUTES = [r for r in _np_V14_SELECTED_ROUTES if (r['m'], r['layer']) not in {(n['m'], n['layer']) for n in _np_V15_ROUTES}] + _np_V15_ROUTES

def _np_prepare_v15(weights: _np_Weights) -> dict:
    state = _np_prepare_v14(weights)
    for route in _np_V15_ROUTES:
        m, layer = (route['m'], route['layer'])
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        data = {'x': _np_torch.zeros((m, weight.shape[1]), dtype=_np_torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
        _np_candidate_v15(data, state)
    _np_torch.cuda.synchronize()
    return state

def _np_candidate_v15(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    x, layer, n = (data['x'], data['config']['layer'], data['config']['n'])
    if layer == 'unpooler' and x.shape[0] == 256:
        output = _np_tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    return _np_candidate_v14(data, state)
_np__bundle_base_prepare = _np_prepare_v15
_np__bundle_base_kernel = _np_candidate_v15
'Experimental lossless nine-instruction BF16 decode. No route is promoted. See FINDINGS.txt.'
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_dd_decodediet_CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_dd_decodediet_CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def _np_dd_decodediet_dd_cap2_b_cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0
_np_dd_decodediet_CPSSM_CUDA_TEMPLATE = _np__zsrc('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')
_np_dd_decodediet_CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_dd_decodediet_CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_dd_decodediet_CPSSM_ESC_CAP = 32

def _np_dd_decodediet_dd_cap2_b_cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_dd_decodediet_cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_dd_decodediet_cpstage___to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_dd_decodediet_dd_cap2_b_cpssm_table(base):
    return base * 16843009

def _np_dd_decodediet_dd_cap2_b_cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_dd_decodediet_dd_cap2_b_cpssm_units(hi), _np_dd_decodediet_dd_cap2_b_cpssm_units(lo)]
    else:
        mats = [_np_dd_decodediet_dd_cap2_b_cpssm_units(w)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v)
        if not _np_torch.equal(_np_dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_dd_decodediet_dd_cap2_b_cpssm_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_dd_decodediet_dd_cap2_b_cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_dd_decodediet_CPSSM_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_dd_decodediet_CPSSM_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_dd_decodediet_cpstage___to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.dd_cap2_b_cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_dd_decodediet_CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_np_dd_decodediet_CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_np_dd_decodediet_CPSSM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_np_dd_decodediet_CPSSM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_np_dd_decodediet_CPSSM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_np_dd_decodediet_CPSSM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _np_dd_decodediet_dd_cap2_b_cpssm_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_dd_decodediet_dd_cap2_b_cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_dd_decodediet_dd_cap2_b_cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _np_dd_decodediet_dd_cap2_b_cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _np_dd_decodediet_dd_cap2_b_cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_dd_decodediet_dd_cap2_b_cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _np_dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id[inst], sp, _np_dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _np_dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _np_dd_decodediet_cpstage___to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_dd_decodediet_cpstage___to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)
_np_dd_decodediet_CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def _np_dd_decodediet_cps_build(family, instances, depth):
    is32 = False
    prefix = 'dd_cap2_b_cpssm_'
    template = _np_dd_decodediet_CPSSM_CUDA_TEMPLATE
    cpp = _np_dd_decodediet_CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + _np_hashlib.sha256(source.encode()).hexdigest()[:12]
    return _np_load_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_dd_decodediet_prepare_cps(weights, depth=None, layers=_np_dd_decodediet_CPS_LAYERS, buckets=(2, 4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
        for bucket, selected_depth in ((2, 2), (4, 1)):
            if bucket in buckets and selected_layers:
                part = _np_dd_decodediet_prepare_cps(weights, selected_depth, selected_layers, (bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0, 1, 2, 3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm',):
        table = _np_dd_decodediet_CPSSM_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and (layer in _np_dd_decodediet_CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and ((layer, bucket) in _np_dd_decodediet_CPS32_TABLE_MT):
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0, rows, _np_dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
            entries.append(((layer, 'mt', bucket), (inst, cfg)))
        if include_policy and family == 'sm':
            for (layer, m), (inst, cfg) in _np_dd_decodediet_CPSSM_TABLE_M.items():
                if layer not in _np_dd_decodediet_CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m, rows, _np_dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
                entries.append(((layer, 'm', m), (inst, cfg)))
        if not entries:
            continue
        twin = _np_dd_decodediet_dd_cap2_b_cpssm_bf16_twin
        instances = list(dict.fromkeys((inst for _, (inst, _) in entries for inst in (inst, twin(inst)) if inst is not None)))
        ext = _np_dd_decodediet_cps_build(family, instances, depth)
        ids = {inst: i for i, inst in enumerate(instances)}
        plan_fn = _np_dd_decodediet_dd_cap2_b_cpssm_plan
        packs, streams = ({}, {})
        for key, (inst, cfg) in entries:
            sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
            if sp is not None:
                st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
    return st

def _np_dd_decodediet_cps_launch(sp, x, out, n, policy=0, query=0):
    fn = _np_dd_decodediet_dd_cap2_b_cpssm_launch_raw
    return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

def _np_dd_decodediet_run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _np_dd_decodediet_dd_cap2_b_cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != _np_torch.bfloat16 or (x.shape[1] != _np_dd_decodediet_CPS32_LAYER_K[layer]) or (not x.is_cuda):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = _np_torch.empty((m, n), dtype=_np_torch.float32 if layer == 'router' else _np_torch.bfloat16, device=x.device)
    rc = _np_dd_decodediet_cps_launch(sp, x, out, n, st.get('policy', 0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline

def _np_dd_decodediet_tt_mw_mma(n):
    nr = n // 2
    regs = ','.join((f'%{i}' for i in range(nr)))
    outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
    inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
    ops = ['wgmma.fence.sync.aligned;']
    for kk in range(4):
        ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
_np_dd_decodediet_tt_mw_CUDA_PREFIX = '// dd_cap2_b_\n' + '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t L0,uint32_t L1,uint32_t C,uint32_t T0,uint32_t T1,uint32_t* a) {\n  uint32_t rotated;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));\n  const uint32_t low_high = (C & 0x87878787u) + T0;\n  const uint32_t high_high = (rotated & 0x87878787u) + T0;\n  a[0] = prmt(L0, low_high, 0x5140u);\n  a[1] = prmt(L0, low_high, 0x7362u);\n  a[2] = prmt(L1, high_high, 0x5140u);\n  a[3] = prmt(L1, high_high, 0x7362u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
_np_dd_decodediet_tt_mw_CUDA_CORE = _np__zsrc('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')
_np_dd_decodediet_tt_mw_CUDA_MAP = '\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n'
_np_dd_decodediet_tt_mw_CUDA_HOST = '\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;\n  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {DISPATCH}\n  return -1;\n}\nint info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}\n'
_np_dd_decodediet_tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def _np_dd_decodediet_tt_mw_cuda_source(instances):
    cases = []
    attrs = []
    for i, inst in enumerate(instances):
        bn, r, wk, pf, q, c, cap = inst
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    return _np_dd_decodediet_tt_mw_CUDA_PREFIX + '\n'.join((_np_dd_decodediet_tt_mw_mma(n) for n in (16, 32, 64))) + _np_dd_decodediet_tt_mw_CUDA_CORE + _np_dd_decodediet_tt_mw_CUDA_MAP + _np_dd_decodediet_tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def _np_dd_decodediet_tt_mw_build(instances):
    src = _np_dd_decodediet_tt_mw_cuda_source(instances)
    name = 'dd_cap2_b_m32w_' + _np_hashlib.sha256(src.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_dd_decodediet_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])

def _np_dd_decodediet_tt_mw__to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_dd_decodediet_tt_mw__to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_dd_decodediet_tt_mw_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_dd_decodediet_tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_dd_decodediet_tt_mw_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_dd_decodediet_tt_mw__to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_dd_decodediet_tt_mw_sm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_dd_decodediet_tt_mw_sm_table(base):
    return base * 16843009

def _np_dd_decodediet_tt_mw_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != _np_torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = _np_torch.full((n // r * total * rw, 64), -1, dtype=_np_torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = _np_dd_decodediet_tt_mw_sm_c12_bytes(v)
        if not _np_torch.equal(_np_dd_decodediet_tt_mw_sm_c12_unpack(data, b, entries), v):
            return None
        table = _np_dd_decodediet_tt_mw_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = _np_torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = _np_torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(_np_torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = _np_torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = _np_torch.arange(len(gs), device=w.device) - _np_torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = _np_dd_decodediet_tt_mw__to_i32(entry)
    else:
        data = _np_dd_decodediet_tt_mw_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def _np_dd_decodediet_tt_mw_launch_raw(ext, idx, pk, x, out):
    return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], _np_torch.cuda.current_stream().cuda_stream)

def _np_dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = _np_torch.eye(k, device=w.device, dtype=_np_torch.bfloat16)
    out = _np_torch.empty((k, n), device=w.device, dtype=_np_torch.bfloat16)
    for start in range(0, k, m):
        if _np_dd_decodediet_tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(_np_torch.equal(out.view(_np_torch.int16), w.T.contiguous().view(_np_torch.int16)))
_np_dd_decodediet_tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def _np_dd_decodediet_tt_mw_prepare_m32w(weights, configs=None) -> dict:
    configs = _np_dd_decodediet_tt_mw_CONFIGS if configs is None else configs
    device = next(iter(weights.values())).device
    state = {'plans': {}, 'device': device}
    if device.type != 'cuda' or _np_torch.cuda.get_device_capability(device) != (9, 0):
        return state
    instances = []
    for inst, split in configs.values():
        for c in (inst[5], False):
            twin = inst[:5] + (c,) + inst[6:]
            if twin not in instances:
                instances.append(twin)
    ext = _np_dd_decodediet_tt_mw_build(instances)
    state['ext'] = ext
    packs = {}
    for (m, layer), (inst, split) in configs.items():
        if layer not in weights:
            continue
        w = weights[layer]
        for compressed in (inst[5], False):
            actual = inst[:5] + (compressed,) + inst[6:]
            idx = instances.index(actual)
            if ext.info(idx) % 10000:
                continue
            key = (layer, inst[1], inst[2], split, compressed)
            if key not in packs:
                packs[key] = _np_dd_decodediet_tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
            pk = packs[key]
            if pk is None or not _np_dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, m):
                continue
            if m == 32 and (not _np_dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, 33)):
                continue
            state['plans'][m, layer] = (idx, pk)
            break
    return state

def _np_dd_decodediet_tt_mw_run_m32w(x: _np_torch.Tensor, layer: str, n: int, st: dict) -> _np_torch.Tensor | None:
    if x.ndim != 2 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
        return None
    m = x.shape[0]
    if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    key = (16 if m <= 16 else 32, layer)
    plan = st['plans'].get(key)
    if plan is None:
        return None
    idx, pk = plan
    if n != pk['n'] or x.shape[1] != pk['k']:
        return None
    out = _np_torch.empty((m, n), device=x.device, dtype=_np_torch.bfloat16)
    rc = _np_dd_decodediet_tt_mw_launch_raw(st['ext'], idx, pk, x, out)
    if rc:
        raise RuntimeError(f'm32w launch failed: {rc}')
    return out
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_dd_DECODEDIET_ROUTES = (('pooler', 16), ('unpooler', 16))

def _np_dd_prepare_decodediet(weights) -> dict:
    routes = _np_dd_DECODEDIET_ROUTES
    state = {'routes': frozenset(routes)}
    if not routes:
        return state
    if any((m == 16 and layer in ('pooler', 'unpooler') for layer, m in routes)):
        layers = tuple((layer for layer, m in routes if m == 16 and layer in ('pooler', 'unpooler')))
        state['cps'] = _np_dd_decodediet_prepare_cps(weights, layers=layers, buckets=(2,))
    return state

def _np_dd_run_decodediet(x, layer, n, state) -> _np_torch.Tensor | None:
    m = x.shape[0] if x.ndim == 2 else 0
    if (layer, m) not in state.get('routes', ()):
        return None
    return _np_dd_decodediet_run_cps(x, layer, n, state['cps'])
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
import hashlib as _np_hashlib
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_rw_SELECTED = ()
_np_rw_SOURCES = {128: _np__zsrc('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'), 256: _np__zsrc('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')}

def _np_rw__prepare(weights: dict, arms: tuple) -> dict:
    w = weights.get('router')
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != _np_torch.float32 or (not w.is_cuda):
        return {'enabled': False}
    if _np_torch.cuda.get_device_capability(w.device) != (9, 0):
        return {'enabled': False}
    hi = w.to(_np_torch.bfloat16)
    lo = (w - hi.float()).to(_np_torch.bfloat16)
    packed = _np_torch.stack([p.view(4, 64, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)
    state = {'enabled': True, 'weight': w, 'packed': packed, 'arms': {}}
    for arm in arms:
        label = 'rwdl_' + str(arm)
        source = _np_rw_SOURCES[arm]
        cpp = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint ' + label + '_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);'
        ext = _np_load_inline('routerwide_' + _np_hashlib.sha256(source.encode()).hexdigest()[:14], cpp, cuda_sources=source, functions=[label + '_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        state['arms'][arm] = (ext, getattr(ext, label + '_launch'))
        x = _np_torch.zeros((arm, 8192), device=w.device, dtype=_np_torch.bfloat16)
        _np_rw_run_routerwide_candidate(x, state, arm)
    _np_torch.cuda.synchronize(w.device)
    return state

def _np_rw_prepare_routerwide(weights: dict) -> dict:
    return _np_rw__prepare(weights, _np_rw_SELECTED) if _np_rw_SELECTED else {'enabled': False}

def _np_rw_prepare_routerwide_candidates(weights: dict) -> dict:
    return _np_rw__prepare(weights, (128, 256))

def _np_rw_run_routerwide_candidate(x: _np_torch.Tensor, state: dict, arm: int) -> _np_torch.Tensor:
    out = _np_torch.empty((x.shape[0], 256), device=x.device, dtype=_np_torch.float32)
    rc = state['arms'][arm][1](0, state['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], _np_torch.cuda.current_stream(x.device).cuda_stream, 0)
    if rc:
        raise RuntimeError(f'routerwide launch failed: {rc}')
    return out

def _np_rw_run_routerwide(x: _np_torch.Tensor, layer: str, n: int, st: dict) -> _np_torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256 or (x.ndim != 2):
        return None
    if x.shape[1] != 8192 or x.shape[0] not in _np_rw_SELECTED or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    return _np_rw_run_routerwide_candidate(x, st, x.shape[0])
'Router completion experiment. No route passed H200 confirmation.'
import hashlib as _np_hashlib
_np_rt_SELECTED = ()
_np_rt__CUDA = _np__zsrc('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')
_np_rt__CPP = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint routertail_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

def _np_rt__pack(w, s=16, wk=8):
    import torch as _np_torch
    hi = w.to(_np_torch.bfloat16)
    lo = (w - hi.float()).to(_np_torch.bfloat16)

    def units(v):
        v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        return v.permute(0, 1, 3, 2, 4)
    data = _np_torch.cat([units(hi), units(lo)], dim=2).contiguous()
    ku = 128 // s // wk
    return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

def _np_rt_prepare_routertail_candidates(weights) -> dict:
    import torch as _np_torch
    from torch.utils.cpp_extension import load_inline as _np_load_inline
    w = weights.get('router')
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != _np_torch.float32 or (not w.is_cuda):
        return {'enabled': False}
    if _np_torch.cuda.get_device_capability(w.device) != (9, 0):
        return {'enabled': False}
    name = 'routertail_' + _np_hashlib.sha256(_np_rt__CUDA.encode()).hexdigest()[:12]
    ext = _np_load_inline(name, _np_rt__CPP, cuda_sources=_np_rt__CUDA, functions=['routertail_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
    st = {'enabled': True, 'ext': ext, 'weight': w.contiguous(), 'packed': _np_rt__pack(w)}
    for m in (8, 16, 17):
        x = _np_torch.zeros((m, 8192), device=w.device, dtype=_np_torch.bfloat16)
        _np_rt_run_routertail_candidate(x, st)
    _np_torch.cuda.synchronize(w.device)
    return st

def _np_rt_run_routertail_candidate(x, st):
    import torch as _np_torch
    if not st.get('enabled') or x.ndim != 2 or x.shape[1] != 8192 or (not 2 <= len(x) <= 32):
        return None
    if x.dtype != _np_torch.bfloat16 or not x.is_cuda or x.device != st['weight'].device:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = _np_torch.empty((len(x), 256), device=x.device, dtype=_np_torch.float32)
    rc = st['ext'].routertail_small(st['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), len(x), 16, 8, _np_torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'routertail launch failed: {rc}')
    return out

def _np_rt_prepare_routertail(weights) -> dict:
    return {'enabled': False}

def _np_rt_run_routertail(x, layer, n, st):
    return None
'Lossless BF16 projection refinements; only confirmed exact shapes are claimed.'
import torch as _np_torch
from torch.utils.cpp_extension import load_inline as _np_load_inline
_np_mt_W0_DC2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_np_mt_W0_DC2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_mt_W0_DC2_ESC_CAP = 32

def _np_mt_wp0_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_mt_wp0_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_mt_wp0_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_mt_wp0_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_mt_wp0_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_mt_wp0_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else _np_mt_W0_DC2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_mt_wp0_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_mt_wp0_c12_unpack(data, base, esc, decode=0):
    """Inverse of wp0_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else _np_mt_W0_DC2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_mt_wp0_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_mt_wp0_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_mt_wp0_units(hi), _np_mt_wp0_units(lo)]
    else:
        mats = [_np_mt_wp0_units(w, layout)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_mt_wp0_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_mt_wp0_c12_bytes(v, decode)
        if not _np_torch.equal(_np_mt_wp0_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(_np_mt_wp0_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_mt_wp0_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_mt_wp0_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_mt_wp0_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_mt_W0_DC2_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_mt_W0_DC2_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_mt_wp0_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
_np_mt_M16TRACE_ROUTES = {'o': {'cfg': (2, 4, 4, 1), 'inst': (1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), 'decode': 0, 'rows': 8192, 'k': 1024, 'variant': 'next_early'}}
_np_mt_M16TRACE_SOURCES = {'o': ('\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m16trace_59fcb5b72994_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n', _np__zsrc('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'), 'm16trace_59fcb5b72994_launch')}

def _np_mt__launch(fn, index, plan, x, out, query=0) -> int:
    return fn(index, plan['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), plan['esc'].data_ptr(), x.shape[0], plan['rows'], plan['k'], *plan['cfg'], plan['t01'], plan['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_mt__onehot(fn, index, plan, weight) -> bool:
    x = _np_torch.zeros((16, plan['k']), device=weight.device, dtype=_np_torch.bfloat16)
    out = _np_torch.empty((16, plan['rows']), device=weight.device, dtype=_np_torch.bfloat16)
    ids = _np_torch.arange(16, device=weight.device)
    for column in range(0, plan['k'], 16):
        x.zero_()
        x[ids, column + ids] = 1
        if _np_mt__launch(fn, index, plan, x, out):
            return False
        if not _np_torch.equal(out, weight[:, column:column + 16].t()):
            return False
    return True

def _np_mt_prepare_m16trace(weights) -> dict:
    state = {'plans': {}}
    for layer, route in _np_mt_M16TRACE_ROUTES.items():
        weight = weights.get(layer)
        if weight is None or not weight.is_cuda or weight.dtype != _np_torch.bfloat16:
            continue
        if weight.ndim != 2 or tuple(weight.shape) != (route['rows'], route['k']):
            continue
        if _np_torch.cuda.get_device_capability(weight.device) != (9, 0):
            continue
        cpp, cuda, name = _np_mt_M16TRACE_SOURCES[layer]
        ext = _np_load_inline(name, cpp, cuda_sources=cuda, functions=[name], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        fn = getattr(ext, name)
        for index, fmt in enumerate((1, 0)):
            pk = _np_mt_wp0_pack_format(weight, fmt, decode=route['decode'])
            if pk is None:
                continue
            sp = _np_mt_wp0_stream(pk, 1, *route['cfg'])
            if sp is None:
                continue
            plan = dict(sp, **route, fn=fn, index=index)
            x = _np_torch.empty((16, route['k']), device=weight.device, dtype=_np_torch.bfloat16)
            out = _np_torch.empty((16, route['rows']), device=weight.device, dtype=_np_torch.bfloat16)
            attr = _np_mt__launch(fn, index, plan, x, out, 2)
            if attr < 0 or attr % 1000:
                continue
            if not _np_mt__onehot(fn, index, plan, weight):
                continue
            state['plans'][layer] = plan
            break
    return state

def _np_mt_run_m16trace(x, layer, n, state) -> _np_torch.Tensor | None:
    if x.ndim != 2 or x.shape[0] != 16 or (not x.is_cuda) or (x.dtype != _np_torch.bfloat16) or (not x.is_contiguous()):
        return None
    plan = state['plans'].get(layer)
    if plan is None or n != plan['rows'] or x.shape[1] != plan['k']:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = _np_torch.empty((16, n), device=x.device, dtype=_np_torch.bfloat16)
    rc = _np_mt__launch(plan['fn'], plan['index'], plan, x, out)
    if rc:
        raise RuntimeError(f'm16trace launch failed: {rc}')
    return out
_np_BUNDLE_QKV_CUDA = _np__zsrc('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')
_np_BUNDLE_ROUTES = {'router128': (128, 'router'), 'router256': (256, 'router'), 'qkv8': (8, 'qkv_gate'), 'router8': (8, 'router'), 'router16': (16, 'router'), 'o16': (16, 'o'), 'pooler16': (16, 'pooler'), 'unpooler16': (16, 'unpooler')}
_np_dd_DECODEDIET_ROUTES = (('pooler', 16), ('unpooler', 16))

def _np_prepare_bundle(weights) -> dict:
    state = {'dd': _np_dd_prepare_decodediet(weights), 'rw': _np_rw_prepare_routerwide_candidates(weights), 'rt': _np_rt_prepare_routertail_candidates(weights), 'mt': _np_mt_prepare_m16trace(weights)}
    symbol = 'qs_literal_owner5_decode5_launch'
    cpp = _np_SM_CPP.replace('sm_launch', symbol)
    ext = _np_load_inline('bundle_owner_' + _np_hashlib.sha256(_np_BUNDLE_QKV_CUDA.encode()).hexdigest()[:12], cpp, cuda_sources=_np_BUNDLE_QKV_CUDA, functions=[symbol], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
    pk = _np_tt_v10_m32x_m32x_pack_format(weights['qkv_gate'], 1, 0, 5)
    sp = None if pk is None else _np_sm_stream(pk, 1, 1, 4, 8, 4)
    state['qkv'] = None if sp is None else dict(sp, fn=getattr(ext, symbol))
    return state

def _np_run_bundle(x, layer, n, st) -> _np_torch.Tensor | None:
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if layer in ('pooler', 'unpooler') and m == 16:
        return _np_dd_run_decodediet(x, layer, n, st['dd'])
    if layer == 'router' and m in (128, 256):
        if n != 256 or x.shape[1] != 8192 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
            return None
        if not st['rw'].get('enabled') or x.device != st['rw']['weight'].device or x.data_ptr() % 16:
            return None
        return _np_rw_run_routerwide_candidate(x, st['rw'], m)
    if layer == 'router' and m in (8, 16):
        return _np_rt_run_routertail_candidate(x, st['rt']) if n == 256 else None
    if layer == 'o' and m == 16:
        return _np_mt_run_m16trace(x, layer, n, st['mt'])
    if layer == 'qkv_gate' and m == 8:
        sp = st['qkv']
        if sp is None or n != 2304 or x.shape[1] != 8192 or (x.dtype != _np_torch.bfloat16) or (not x.is_cuda) or (not x.is_contiguous()):
            return None
        if x.device != sp['wp'].device or x.data_ptr() % 16:
            return None
        out = _np_torch.empty((m, n), device=x.device, dtype=_np_torch.bfloat16)
        rc = sp['fn'](1, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, 8192, 1, 4, 8, 4, sp['t01'], sp['u01'], _np_torch.cuda.current_stream(x.device).cuda_stream, 0)
        if rc:
            raise RuntimeError(f'bundle qkv launch failed: {rc}')
        return out
    return None

def _np_prepare(weights: _np_Weights) -> dict:
    state = _np__bundle_base_prepare(weights)
    state['bundle'] = _np_prepare_bundle(weights)
    return state

def _np_kernel_fn(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    output = None
    if 'bundle' in state:
        output = _np_run_bundle(data['x'], data['config']['layer'], data['config']['n'], state['bundle'])
    return _np__bundle_base_kernel(data, state) if output is None else output
_np_V16_ROUTES = [{'m': 128, 'layer': 'router', 'name': 'router128', 'module': 'v16_bundle', 'owner_family': 'routerwide'}, {'m': 256, 'layer': 'router', 'name': 'router256', 'module': 'v16_bundle', 'owner_family': 'routerwide'}, {'m': 8, 'layer': 'qkv_gate', 'name': 'qkv8', 'module': 'v16_bundle', 'owner_family': 'owner5_decode5'}, {'m': 8, 'layer': 'router', 'name': 'router8', 'module': 'v16_bundle', 'owner_family': 'routertail_local4'}, {'m': 16, 'layer': 'router', 'name': 'router16', 'module': 'v16_bundle', 'owner_family': 'routertail_local4'}, {'m': 16, 'layer': 'o', 'name': 'o16', 'module': 'v16_bundle', 'owner_family': 'm16trace_early'}, {'m': 16, 'layer': 'pooler', 'name': 'pooler16', 'module': 'v16_bundle', 'owner_family': 'decodediet'}, {'m': 16, 'layer': 'unpooler', 'name': 'unpooler16', 'module': 'v16_bundle', 'owner_family': 'm32x_round3_decoder9'}]
_np_SELECTED_ROUTES = [r for r in _np_V15_SELECTED_ROUTES if (r['m'], r['layer']) not in {(q['m'], q['layer']) for q in _np_V16_ROUTES}] + _np_V16_ROUTES
_np__b2_base_prepare = _np_prepare
_np__b2_base_kernel = _np_kernel_fn
_np_b2p_tt_v10_m32x_M32X_ESC_CAP = 32
_np_b2p_tt_v10_m32x__CFG = (1, 8, 16, 1)
_np_b2p_tt_v10_m32x__CUDA = _np__zsrc('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')
_np_b2p_tt_v10_m32x__CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def _np_b2p_tt_v10_m32x_m32x_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_b2p_tt_v10_m32x_m32x_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_b2p_tt_v10_m32x_m32x_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_b2p_tt_v10_m32x_m32x_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_b2p_tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_b2p_tt_v10_m32x_m32x_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_b2p_tt_v10_m32x_m32x_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
    """Inverse of r3_pool32_pooler_d9_cap2_a_m32x_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_b2p_tt_v10_m32x_m32x_table(base):
    return base * 16843009

def _np_b2p_tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_b2p_tt_v10_m32x_m32x_units(hi), _np_b2p_tt_v10_m32x_m32x_units(lo)]
    else:
        mats = [_np_b2p_tt_v10_m32x_m32x_units(w, layout)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_b2p_tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_b2p_tt_v10_m32x_m32x_c12_bytes(v, decode)
        if not _np_torch.equal(_np_b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base * 16843009)
        else:
            tabs.append(_np_b2p_tt_v10_m32x_m32x_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_b2p_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_b2p_tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_b2p_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_b2p_tt_v10_m32x_M32X_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_b2p_tt_v10_m32x_M32X_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_b2p_tt_v10_m32x_m32x_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.r3_pool32_pooler_d9_cap2_a_m32x_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_b2p_tt_v10_m32x_m32x_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_b2p_tt_v10_m32x_m32x_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_b2p_tt_v10_m32x__build():
    name = 'r3_pool32_pooler_d9_cap2_a_m32x_pooler_' + _np_hashlib.sha256(_np_b2p_tt_v10_m32x__CUDA.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_b2p_tt_v10_m32x__CPP, cuda_sources=_np_b2p_tt_v10_m32x__CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def _np_b2p_tt_v10_m32x__make_plan(ext, weights, fmt):
    pack = _np_b2p_tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = _np_b2p_tt_v10_m32x_m32x_stream(pack, 1, *_np_b2p_tt_v10_m32x__CFG)
    if stream is None:
        return None
    return dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=4096, cfg=_np_b2p_tt_v10_m32x__CFG)

def _np_b2p_tt_v10_m32x__run(x, plan):
    out = _np_torch.empty((x.shape[0], 4096), dtype=_np_torch.bfloat16, device=x.device)
    rc = _np_b2p_tt_v10_m32x_m32x_launch_raw(plan['ext'], plan['idx'], plan, x, out, 4096, _np_b2p_tt_v10_m32x__CFG)
    if rc:
        raise RuntimeError(f'm32x launch failed: {rc}')
    return out
_np_b2p_tt_v10_m32x__W_CUDA = _np__zsrc('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')
_np_b2p_tt_v10_m32x__W_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_w_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint r3_pool32_pooler_d9_cap2_a_m32x_w_info(int);\n'

def _np_b2p_tt_v10_m32x__w_c12_bytes(v):
    return _np_b2p_tt_v10_m32x_m32x_c12_bytes(v, 5)

def _np_b2p_tt_v10_m32x__w_c12_unpack(data, base, esc):
    return _np_b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, 5)

def _np_b2p_tt_v10_m32x__w_table(base):
    return base << 8 | base << 24

def _np_b2p_tt_v10_m32x__w_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != _np_torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = _np_torch.full((n // r * total * rw, 64), -1, dtype=_np_torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = _np_b2p_tt_v10_m32x__w_c12_bytes(v)
        if not _np_torch.equal(_np_b2p_tt_v10_m32x__w_c12_unpack(data, b, entries), v):
            return None
        table = _np_b2p_tt_v10_m32x__w_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = _np_torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = _np_torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(_np_torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = _np_torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = _np_torch.arange(len(gs), device=w.device) - _np_torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = _np_b2p_tt_v10_m32x_m32x_to_i32(entry)
    else:
        data = _np_b2p_tt_v10_m32x_m32x_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def _np_b2p_tt_v10_m32x__w_launch_raw(ext, idx, pk, x, out):
    return ext.r3_pool32_pooler_d9_cap2_a_m32x_w_launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], _np_torch.cuda.current_stream().cuda_stream)

def _np_b2p_tt_v10_m32x__w_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = _np_torch.eye(k, device=w.device, dtype=_np_torch.bfloat16)
    out = _np_torch.empty((k, n), device=w.device, dtype=_np_torch.bfloat16)
    for start in range(0, k, m):
        if _np_b2p_tt_v10_m32x__w_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(_np_torch.equal(out.view(_np_torch.int16), w.T.contiguous().view(_np_torch.int16)))

def _np_b2p_tt_v10_m32x__w_build():
    name = 'r3_pool32_pooler_d9_cap2_a_m32x_unpooler_' + _np_hashlib.sha256(_np_b2p_tt_v10_m32x__W_CUDA.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_b2p_tt_v10_m32x__W_CPP, cuda_sources=_np_b2p_tt_v10_m32x__W_CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_w_launch', 'r3_pool32_pooler_d9_cap2_a_m32x_w_info'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def _np_b2p_tt_v10_m32x__w_make_plan(ext, weights, fmt):
    pack = _np_b2p_tt_v10_m32x__w_pack_weight(weights, 64, 4, 1, fmt == 1)
    if pack is None:
        return None
    return dict(pack, ext=ext, idx=0 if fmt == 1 else 1, fmt=fmt)

def _np_b2p_tt_v10_m32x__w_run(x, plan):
    out = _np_torch.empty((x.shape[0], 8192), dtype=_np_torch.bfloat16, device=x.device)
    rc = _np_b2p_tt_v10_m32x__w_launch_raw(plan['ext'], plan['idx'], plan, x, out)
    if rc:
        raise RuntimeError(f'm32x unpooler launch failed: {rc}')
    return out

def _np_b2p_tt_v10_m32x__w_verify_random(plan, weights):
    for m in (9, 16, 17, 24, 32):
        x = _np_torch.randn((m, 4096), device=weights.device, dtype=_np_torch.bfloat16)
        x[-1].zero_()
        x[-1, m * 131 % 4096] = 16
        ref = (x.double() @ weights.double().T).to(_np_torch.bfloat16).double()
        out = _np_b2p_tt_v10_m32x__w_run(x, plan).double()
        diff = out - ref
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > 0.0025:
            return False
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if float(rows.max()) > 0.01 or float(rows[-1]) > 0.00025:
            return False
    return True
_np_b2p_tt_v10_m32x__Q_CFG = (1, 4, 8, 4)
_np_b2p_tt_v10_m32x__Q_CUDA = _np__zsrc('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')
_np_b2p_tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def _np_b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.r3_pool32_pooler_d9_cap2_a_m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_b2p_tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_b2p_tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_b2p_tt_v10_m32x__q_build():
    name = 'r3_pool32_pooler_d9_cap2_a_m32x_qkv_' + _np_hashlib.sha256(_np_b2p_tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_b2p_tt_v10_m32x__Q_CPP, cuda_sources=_np_b2p_tt_v10_m32x__Q_CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def _np_b2p_tt_v10_m32x__q_make_plan(ext, weights, fmt):
    pack = _np_b2p_tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = _np_b2p_tt_v10_m32x_m32x_stream(pack, 1, *_np_b2p_tt_v10_m32x__Q_CFG)
    if stream is None:
        return None
    plan = dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=2304, cfg=_np_b2p_tt_v10_m32x__Q_CFG)
    x = weights[:32]
    out = _np_torch.empty((32, 2304), dtype=_np_torch.bfloat16, device=weights.device)
    capacity = _np_b2p_tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, _np_b2p_tt_v10_m32x__Q_CFG, 1)
    attr = _np_b2p_tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, _np_b2p_tt_v10_m32x__Q_CFG, 2)
    if capacity < 144 or attr < 0 or attr % 1000:
        return None
    return plan

def _np_b2p_tt_v10_m32x__q_run(x, plan):
    out = _np_torch.empty((x.shape[0], 2304), dtype=_np_torch.bfloat16, device=x.device)
    rc = _np_b2p_tt_v10_m32x__q_launch_raw(plan['ext'], plan['idx'], plan, x, out, 2304, _np_b2p_tt_v10_m32x__Q_CFG)
    if rc:
        raise RuntimeError(f'm32x qkv launch failed: {rc}')
    return out

def _np_b2p_tt_v10_m32x_prepare_m32x(weights, *, include_unpooler: bool=False) -> dict:
    state = {'plans': {}, 'device': None}
    configs = [('pooler', (4096, 8192), _np_b2p_tt_v10_m32x__build, _np_b2p_tt_v10_m32x__make_plan), ('qkv_gate', (2304, 8192), _np_b2p_tt_v10_m32x__q_build, _np_b2p_tt_v10_m32x__q_make_plan), ('unpooler', (8192, 4096), _np_b2p_tt_v10_m32x__w_build, _np_b2p_tt_v10_m32x__w_make_plan)]
    for layer, shape, build, make_plan in configs:
        if layer == 'unpooler' and (not include_unpooler):
            continue
        w = weights.get(layer)
        if w is None or w.dtype != _np_torch.bfloat16 or tuple(w.shape) != shape or (not w.is_cuda):
            continue
        if _np_torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        if state['device'] is not None and state['device'] != w.device:
            continue
        state['device'] = w.device
        ext = build()
        for fmt in (1, 0):
            plan = make_plan(ext, w, fmt)
            if plan is None:
                continue
            if layer == 'pooler':
                ok = _np_b2p_tt_v10_m32x_m32x_verify_onehot(ext, plan['idx'], plan, w, _np_b2p_tt_v10_m32x__CFG, 32) and _np_b2p_tt_v10_m32x_m32x_verify_random(ext, plan['idx'], plan, w, _np_b2p_tt_v10_m32x__CFG, [9, 16, 17, 24, 32])
            elif layer == 'qkv_gate':
                ok = _np_b2p_tt_v10_m32x__q_verify_onehot(ext, plan['idx'], plan, w, _np_b2p_tt_v10_m32x__Q_CFG, 32) and _np_b2p_tt_v10_m32x__q_verify_random(ext, plan['idx'], plan, w, _np_b2p_tt_v10_m32x__Q_CFG, [9, 16, 17, 24, 32])
            else:
                ok = _np_b2p_tt_v10_m32x__w_verify_onehot(ext, plan['idx'], plan, w, 32) and _np_b2p_tt_v10_m32x__w_verify_random(plan, w)
            if ok:
                state['plans'][layer] = plan
                break
    return state

def _np_b2p_tt_v10_m32x_run_m32x(x, layer, n, st) -> _np_torch.Tensor | None:
    expected = {'pooler': (4096, 8192), 'qkv_gate': (2304, 8192), 'unpooler': (8192, 4096)}.get(layer)
    if expected is None or x.ndim != 2 or x.shape[0] != 32:
        return None
    if (n, x.shape[1]) != expected or x.dtype != _np_torch.bfloat16 or (not x.is_cuda):
        return None
    if x.device != st.get('device') or not x.is_contiguous():
        return None
    plan = st.get('plans', {}).get(layer)
    if plan is None:
        return None
    if layer == 'pooler':
        return _np_b2p_tt_v10_m32x__run(x, plan)
    if layer == 'qkv_gate':
        return _np_b2p_tt_v10_m32x__q_run(x, plan)
    return _np_b2p_tt_v10_m32x__w_run(x, plan)
_np_b2q_tt_v10_m32x_M32X_ESC_CAP = 32

def _np_b2q_tt_v10_m32x_m32x_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_b2q_tt_v10_m32x_m32x_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_b2q_tt_v10_m32x_m32x_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_b2q_tt_v10_m32x_m32x_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_b2q_tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_b2q_tt_v10_m32x_m32x_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_b2q_tt_v10_m32x_m32x_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
    """Inverse of m32x_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_b2q_tt_v10_m32x_m32x_table(base):
    return base * 16843009

def _np_b2q_tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_b2q_tt_v10_m32x_m32x_units(hi), _np_b2q_tt_v10_m32x_m32x_units(lo)]
    else:
        mats = [_np_b2q_tt_v10_m32x_m32x_units(w, layout)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_b2q_tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_b2q_tt_v10_m32x_m32x_c12_bytes(v, decode)
        if not _np_torch.equal(_np_b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base * 16843009)
        else:
            tabs.append(_np_b2q_tt_v10_m32x_m32x_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_b2q_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_b2q_tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_b2q_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_b2q_tt_v10_m32x_M32X_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_b2q_tt_v10_m32x_M32X_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_b2q_tt_v10_m32x_m32x_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
_np_b2q_tt_v10_m32x__Q_CUDA = _np__zsrc('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')
_np_b2q_tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def _np_b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_b2q_tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_b2q_tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_b2q_tt_v10_m32x__q_build():
    name = 'm32x_qkv_' + _np_hashlib.sha256(_np_b2q_tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_b2q_tt_v10_m32x__Q_CPP, cuda_sources=_np_b2q_tt_v10_m32x__Q_CUDA, functions=['r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
_np_b2o_OLAYER_CUDA = _np__zsrc('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')
_np_b2o_OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_o32_o_direct_a_ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def _np_b2o_olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void r3_o32_o_direct_a_ol_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return r3_o32_o_direct_a_ol_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) r3_o32_o_direct_a_ol_mma{n}(a,aw,ax);' for n in ns))
    return _np_b2o_OLAYER_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def _np_b2o_build_olayer(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (0, 1, 2, 3)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = _np_b2o_olayer_source(configs)
    name = 'olayer_' + _np_hashlib.sha256(source.encode()).hexdigest()[:12]
    return _np_load_inline(name, _np_b2o_OLAYER_CPP, cuda_sources=source, functions=['r3_o32_o_direct_a_ol_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

def _np_b2o_pack_olayer(weight) -> _np_torch.Tensor:
    columns = _np_torch.arange(64, device=weight.device)[None, :] ^ (_np_torch.arange(64, device=weight.device)[:, None] & 7) * 8
    tiled = weight.reshape(128, 64, 16, 64).permute(0, 2, 1, 3)
    return tiled.gather(-1, columns.expand(128, 16, 64, 64)).contiguous()

def _np_b2o_prepare_olayer_candidates(weights, configs) -> dict:
    weight = weights['o'].contiguous()
    return {'enabled': True, 'ext': _np_b2o_build_olayer(configs), 'configs': configs, 'weight': weight, 'packed': _np_b2o_pack_olayer(weight) if any((c[4] == 1 for c in configs)) else None, 'tiled32': weight.reshape(256, 32, 16, 64).permute(0, 2, 1, 3).contiguous() if any((c[4] == 2 for c in configs)) else None}

def _np_b2o_run_olayer_candidate(x, st, index) -> _np_torch.Tensor:
    pack = st['configs'][index][4]
    weight = st['tiled32'] if pack == 2 else st['packed'] if pack == 1 else st['weight']
    out = _np_torch.empty((x.shape[0], 8192), device=x.device, dtype=_np_torch.bfloat16)
    rc = st['ext'].r3_o32_o_direct_a_ol_launch(index, x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.shape[0], _np_torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f"olayer launch failed: {rc}, config={st['configs'][index]}")
    return out
_np_BUNDLE2_ROUTES = {'qkv16': (16, 'qkv_gate'), 'pooler32': (32, 'pooler'), 'o32': (32, 'o'), 'unpooler32': (32, 'unpooler')}
_np_BUNDLE2_UNPOOL32 = 'decodediet'
_np_BUNDLE2_UNPOOL1 = 'v16'

def _np_b2_prepare_qkv(weights) -> dict:
    w = weights['qkv_gate']
    ext = _np_b2q_tt_v10_m32x__q_build()
    cfg = (1, 4, 8, 4)
    for fmt in (1, 0):
        pk = _np_b2q_tt_v10_m32x_m32x_pack_format(w, fmt, 0, 5)
        if pk is None:
            continue
        plan = _np_b2q_tt_v10_m32x_m32x_stream(pk, 1, *cfg)
        if plan is None:
            continue
        idx = 0 if fmt else 1
        plan.update(ext=ext, idx=idx, cfg=cfg, n=2304)
        x = w[:16]
        out = _np_torch.empty((16, 2304), device=w.device, dtype=w.dtype)
        capacity = _np_b2q_tt_v10_m32x__q_launch_raw(ext, idx, plan, x, out, 2304, cfg, 1)
        attr = _np_b2q_tt_v10_m32x__q_launch_raw(ext, idx, plan, x, out, 2304, cfg, 2)
        if capacity < 144 or attr < 0 or attr % 1000:
            continue
        if _np_b2q_tt_v10_m32x__q_verify_onehot(ext, idx, plan, w, cfg, 16) and _np_b2q_tt_v10_m32x__q_verify_random(ext, idx, plan, w, cfg, [9, 15, 16]):
            return plan
    return {}

def _np_prepare_bundle2(weights) -> dict:
    state = {'qkv': _np_b2_prepare_qkv(weights), 'pooler': _np_b2p_tt_v10_m32x_prepare_m32x({'pooler': weights['pooler']}), 'o': _np_b2o_prepare_olayer_candidates(weights, [(32, 128, 8, 1, 1, 1)])}
    state['diet'] = _np_dd_decodediet_tt_mw_prepare_m32w(weights, configs={(32, 'unpooler'): _np_dd_decodediet_tt_mw_CONFIGS[32, 'unpooler']})
    return state

def _np_run_bundle2(x, layer, n, st) -> _np_torch.Tensor | None:
    if x.ndim != 2 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()) or x.data_ptr() % 16:
        return None
    m, k = x.shape
    if layer == 'qkv_gate' and m == 16 and (n == 2304) and (k == 8192):
        p = st['qkv']
        if not p or x.device != p['wp'].device:
            return None
        out = _np_torch.empty((m, n), device=x.device, dtype=x.dtype)
        rc = _np_b2q_tt_v10_m32x__q_launch_raw(p['ext'], p['idx'], p, x, out, n, p['cfg'])
        if rc:
            raise RuntimeError(f'bundle2 qkv launch failed: {rc}')
        return out
    if layer == 'pooler' and m == 32:
        return _np_b2p_tt_v10_m32x_run_m32x(x, layer, n, st['pooler'])
    if layer == 'o' and m == 32 and (n == 8192) and (k == 1024) and (x.device == st['o']['weight'].device):
        return _np_b2o_run_olayer_candidate(x, st['o'], 0)
    if layer == 'unpooler' and m == 32:
        return _np_dd_decodediet_tt_mw_run_m32w(x, layer, n, st['diet'])
    return None

def _np_prepare(weights: _np_Weights) -> dict:
    state = _np__b2_base_prepare(weights)
    state['bundle2'] = _np_prepare_bundle2(weights)
    return state

def _np_kernel_fn(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    out = None
    if 'bundle2' in state:
        out = _np_run_bundle2(data['x'], data['config']['layer'], data['config']['n'], state['bundle2'])
    return _np__b2_base_kernel(data, state) if out is None else out
_np_V17_ROUTES = [{'name': 'qkv16', 'm': 16, 'layer': 'qkv_gate', 'module': 'v17_bundle2', 'owner_variant': 'push_d9_cap6'}, {'name': 'o32', 'm': 32, 'layer': 'o', 'module': 'v17_bundle2', 'owner_variant': 'direct'}, {'name': 'pooler32', 'm': 32, 'layer': 'pooler', 'module': 'v17_bundle2', 'owner_variant': 'd9_cap2'}, {'name': 'unpooler32', 'm': 32, 'layer': 'unpooler', 'module': 'v17_bundle2', 'owner_variant': 'decodediet'}]
_np_SELECTED_ROUTES = [r for r in _np_SELECTED_ROUTES if (r['m'], r['layer']) not in {(q['m'], q['layer']) for q in _np_V17_ROUTES}] + _np_V17_ROUTES
from concurrent.futures import ThreadPoolExecutor as _np__PreparePool

def _np_prepare(weights: _np_Weights) -> dict:
    state = {name: w.contiguous() for name, w in weights.items()}
    device = weights['qkv_gate'].device
    hi = weights['router'].to(_np_torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = (weights['router'] - hi.float()).to(_np_torch.bfloat16).t().contiguous()
    state['router_weight'] = weights['router']
    _np_torch.cuda.synchronize(device)
    jobs = [('sm', _np_prepare_small_m, weights, {}), ('m32', _np_prepare_m32, weights, {}), ('wide2', _np_prepare_wide2, weights, {}), ('router', _np_prepare_router, weights, {}), ('qkv', _np_prepare_qkv, weights, {}), ('router2', _np_prepare_router2, weights, {}), ('olayer', _np_prepare_olayer, weights, {}), ('cps', _np_prepare_cps, weights, {}), ('new_lt', _np_tt_lt_prepare_cublaslt, weights, {}), ('new_mw', _np_tt_mw_prepare_m32w, weights, {}), ('v10_sm2', _np_tt_v10_sm2_prepare_smallm2, weights, {}), ('v10_warps', _np_tt_v10_warps_prepare_warps, weights, {}), ('v10_tune', _np_prepare_v10_tune, weights, {}), ('v10_m32x', _np_tt_v10_m32x_prepare_m32x, weights, {'include_unpooler': False}), ('v11_m32x', _np_tt_v10_m32x_prepare_m32x, {'unpooler': weights['unpooler']}, {'include_unpooler': True}), ('v12_smtrace1', _np_tt_v12_smtrace1_prepare_smtrace, weights, {}), ('v12_o1', _np_prepare_v12_o1, weights, {}), ('v13_smtrace8', _np_tt_v13_smtrace8_prepare_smtrace, weights, {}), ('v14_qkv', _np_tt_v14_qkv_prepare_qkvbig, weights, {}), ('bundle', _np_prepare_bundle, weights, {}), ('bundle2', _np_prepare_bundle2, weights, {})]

    def run_job(fn, selected_weights, options):
        with _np_torch.cuda.device(device):
            return fn(selected_weights, **options)
    with _np__PreparePool(max_workers=4) as pool:
        pending = [(name, pool.submit(run_job, fn, selected_weights, options)) for name, fn, selected_weights, options in jobs]
        for name, future in pending:
            state[name] = future.result()
    if state['qkv'].get('large') is not None:
        state['qkv']['is_enabled'] = True
    for routes, fn in ((_np_V9_SELECTED_ROUTES, _np_candidate_v9), (_np_V10_ROUTES, _np_candidate_v10), (_np_V11_ROUTES, _np_candidate_v11), (_np_V12_ROUTES, _np_candidate_v12), (_np_V13_ROUTES, _np_candidate_v13), (_np_V14_ROUTES, _np_candidate_v14), (_np_V15_ROUTES, _np_candidate_v15)):
        for route in routes:
            m, layer = (route['m'], route['layer'])
            weight = weights[layer]
            n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
            data = {'x': _np_torch.zeros((m, weight.shape[1]), dtype=_np_torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
            fn(data, state)
    _np_torch.cuda.synchronize(device)
    return state
_np_v18d_SM_CUDA_TEMPLATE = _np__zsrc('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')
_np_v18d_SM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint v18d_native_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_v18d_SM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_np_v18d_SM_ESC_CAP = 32

def _np_v18d_v18d_native_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return v18d_native_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _np_v18d_SM_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _np_v18d_v18d_native_build(instances, name='dd_cap2_b_small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _np_load_inline(name, _np_v18d_SM_CPP + extra_cpp, cuda_sources=_np_v18d_v18d_native_cuda_source(instances) + extra_cuda, functions=['v18d_native_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_v18d_v18d_native_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_v18d__to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_v18d__to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_v18d_v18d_native_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_v18d__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_v18d_v18d_native_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_v18d__to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_v18d_v18d_native_c12_unpack(data, base, esc):
    """Inverse of v18d_native_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = _np_torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = _np_torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_v18d_v18d_native_table(base):
    return base * 16843009

def _np_v18d_v18d_native_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_v18d_v18d_native_units(hi), _np_v18d_v18d_native_units(lo)]
    else:
        mats = [_np_v18d_v18d_native_units(w)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_v18d_v18d_native_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_v18d_v18d_native_c12_bytes(v)
        if not _np_torch.equal(_np_v18d_v18d_native_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_v18d_v18d_native_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_v18d_v18d_native_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_v18d_v18d_native_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_v18d_v18d_native_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_v18d_SM_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_v18d_SM_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_v18d__to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_v18d_v18d_native_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.v18d_native_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)
_np_v18d_SM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_np_v18d_SM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_np_v18d_SM_MAX_M = 32
_np_v18d_SM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_np_v18d_SM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_np_v18d_SM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_np_v18d_SM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _np_v18d_v18d_native_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _np_v18d_v18d_native_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_v18d_v18d_native_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_v18d_v18d_native_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_v18d_v18d_native_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_v18d_v18d_native_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_v18d_v18d_native_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_v18d_v18d_native_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_v18d_v18d_native_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _np_v18d_v18d_native_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _np_v18d_v18d_native_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_v18d_v18d_native_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _np_v18d_v18d_native_verify_onehot(ext, inst_id[inst], sp, _np_v18d_v18d_native_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _np_v18d_v18d_native_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _np_v18d_prepare_small_m(weights, table_m=None, table_mt=None, name='dd_cap2_b_small_m_gemv'):
    table_m = dict(_np_v18d_SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(_np_v18d_SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = _np_v18d_v18d_native_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, _np_v18d_SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if _np_v18d_v18d_native_bf16_twin(inst):
            instances.append(_np_v18d_v18d_native_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = _np_v18d_v18d_native_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = _np_v18d_v18d_native_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and _np_v18d_v18d_native_bf16_twin(inst):
            plan = _np_v18d_v18d_native_plan(ext, inst_id, packs, streams, weights[layer], layer, key, _np_v18d_v18d_native_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def _np_v18d_run_small_m(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= _np_v18d_SM_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _np_v18d_v18d_native_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != _np_torch.bfloat16 or (x.shape[1] != _np_v18d_SM_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = _np_torch.empty((m, n), dtype=_np_torch.float32 if layer == 'router' else _np_torch.bfloat16, device=x.device)
    if _np_v18d_v18d_native_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
_np_V18S_CUDA_TEMPLATE = _np__zsrc('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')
_np_V18S_CPP = '\nint v18s_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_np_V18S_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_np_V18S_ESC_CAP = 32

def _np_v18s_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
        cases.append(f'    case {i}: return v18s_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _np_V18S_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _np_v18s_build(instances, name='v18s_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _np_load_inline(name, _np_V18S_CPP + extra_cpp, cuda_sources=_np_v18s_cuda_source(instances) + extra_cuda, functions=['v18s_launch', 'v18s_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _np_v18s_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(_np_torch.int16).to(_np_torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _np_v18s_to_i16(v):
    return _np_torch.where(v >= 32768, v - 65536, v).to(_np_torch.int16)

def _np_v18s_to_i32(v):
    return _np_torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(_np_torch.int32)

def _np_v18s_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _np_v18s_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(_np_torch.uint8).view(r, c, 2048)

def _np_v18s_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = _np_torch.bincount(eh.flatten(), minlength=128).cpu()
    window = _np_torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = _np_torch.where(inwin, d, _np_torch.zeros_like(d))
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_V18S_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _np_v18s_to_i32(code).contiguous().view(_np_torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(_np_torch.uint8)
    data = _np_torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _np_v18s_c12_unpack(data, base, esc):
    """Inverse of v18s_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = _np_torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(_np_torch.int32)
    code = p[:, :, 2].contiguous().view(_np_torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (_np_torch.arange(8, device=dev) * 4).long()
    pos = _np_torch.tensor(_np_V18S_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(_np_torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(_np_torch.int32)
    return val

def _np_v18s_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _np_v18s_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(_np_torch.bfloat16)
        lo = (w - hi.float()).to(_np_torch.bfloat16)
        mats = [_np_v18s_units(hi), _np_v18s_units(lo)]
    else:
        mats = [_np_v18s_units(w)]
    if fmt in (0, 2):
        data = _np_torch.cat([_np_v18s_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _np_v18s_c12_bytes(v)
        if not _np_torch.equal(_np_v18s_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_np_v18s_table(base))
        escs.append(esc + (_np_torch.full_like(esc[0], h),))
    esc = tuple((_np_torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': _np_torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _np_v18s_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _np_v18s_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _np_v18s_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = _np_torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = _np_torch.arange(G, device=dev).view(1, G, 1)
    j = _np_torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = _np_torch.full((G, _np_V18S_ESC_CAP), -1, dtype=_np_torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = _np_torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _np_V18S_ESC_CAP:
            return None
        order = _np_torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = _np_torch.arange(len(g_s), device=dev) - _np_torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _np_v18s_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _np_v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp['fmt'] == 4:
        return ext.v18s_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], _np_torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.v18s_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], _np_torch.cuda.current_stream().cuda_stream, query)

def _np_v18s_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _np_v18s_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _np_v18s_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = _np_torch.float32 if exact.dtype == _np_torch.float32 else _np_torch.bfloat16
    got = _np_torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = _np_torch.eye(k, dtype=_np_torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _np_v18s_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(_np_torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _np_v18s_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == _np_torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = _np_torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = _np_torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(_np_torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(_np_torch.float32 if is_router else _np_torch.bfloat16).double()
        out = _np_torch.empty((m, n), dtype=_np_torch.float32 if is_router else _np_torch.bfloat16, device=w_rows.device)
        if _np_v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / _np_torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(_np_torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _np_v18s_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(_np_torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(_np_torch.bfloat16).float()
    return w_rows

def _np_v18s_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _np_v18s_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _np_v18s_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _np_v18s_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _np_v18s_verify_onehot(ext, inst_id[inst], sp, _np_v18s_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _np_v18s_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
_np_V18_ROUTES = [{'m': 1, 'layer': 'qkv_gate', 'name': 'decodediet_qkv1', 'module': 'v18'}, {'m': 1, 'layer': 'pooler', 'name': 'p32_s1_o6', 'module': 'v18'}, {'m': 8, 'layer': 'o', 'name': 'warps_finalist_o8', 'module': 'v18'}]
_np_V18_SMALL = {('pooler', 1): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1)), ('o', 8): ((1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8, 2, 1, 6), (1, 8, 2, 1))}

def _np_prepare_v18_extra(weights):
    qkv = _np_v18d_prepare_small_m({'qkv_gate': weights['qkv_gate']}, table_m={}, table_mt={('qkv_gate', 1): _np_v18d_SM_TABLE_MT['qkv_gate', 1]}, name='v18_decoder_qkv1')
    instances = list(dict.fromkeys((inst for original, cfg in _np_V18_SMALL.values() for inst in (original, _np_v18s_bf16_twin(original)))))
    ext = _np_v18s_build(instances, 'v18_small_selected')
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams, plans = ({}, {}, {})
    for (layer, m), (inst, cfg) in _np_V18_SMALL.items():
        key = (layer, 'm', m)
        plan = _np_v18s_plan(ext, ids, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None:
            plan = _np_v18s_plan(ext, ids, packs, streams, weights[layer], layer, key, _np_v18s_bf16_twin(inst), cfg)
        if plan is not None:
            plans[layer, m] = plan
    return {'qkv': qkv, 'ext': ext, 'plans': plans}

def _np_run_v18(x, layer, n, st):
    if x.ndim != 2 or x.dtype != _np_torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()) or x.data_ptr() % 16:
        return None
    m, k = x.shape
    if (layer, m) == ('qkv_gate', 1):
        return _np_v18d_run_small_m(x, layer, n, st['qkv'])
    plan = st['plans'].get((layer, m))
    if plan is None or n != plan['rows'] or k != (8192 if layer == 'pooler' else 1024) or (x.device != plan['wp'].device):
        return None
    out = _np_torch.empty((m, n), device=x.device, dtype=_np_torch.bfloat16)
    rc = _np_v18s_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    if rc:
        raise RuntimeError(f'v18 selected launch failed: {rc}')
    return out
_np__v18_base_prepare = _np_prepare
_np__v18_base_kernel = _np_kernel_fn

def _np_prepare(weights: _np_Weights) -> dict:
    state = _np__v18_base_prepare(weights)
    state['v18'] = _np_prepare_v18_extra(weights)
    _np_torch.cuda.synchronize(weights['qkv_gate'].device)
    return state

def _np_kernel_fn(data: _np_Inputs, state: dict) -> _np_torch.Tensor:
    out = _np_run_v18(data['x'], data['config']['layer'], data['config']['n'], state['v18'])
    return _np__v18_base_kernel(data, state) if out is None else out
_np_SELECTED_ROUTES = [r for r in _np_SELECTED_ROUTES if (r['m'], r['layer']) not in {(x['m'], x['layer']) for x in _np_V18_ROUTES}] + _np_V18_ROUTES

# The public backend owns every dense projection, so do not build unused
# legacy BF16 tile buffers or its CUDA extension during decoder preparation.
def _dense_prepare(weights,device):
    return {}
_before_public_projection_prepare=prepare
def prepare(weights):
    state=_before_public_projection_prepare(weights)
    names=('qkv_gate','o','pooler','unpooler','router')
    state['public_projection']=_np_prepare({name:weights[name] for name in names})
    return state
_before_public_projection_dgemm=_dgemm
def _dgemm(x,layer,n,state):
    x=x.contiguous()
    data={'x':x,'config':{'m':x.shape[0],'n':n,'k':x.shape[1],'layer':layer}}
    return _np_kernel_fn(data,state['public_projection'])

_FP32_ROUTER_CPP='torch::Tensor fp32_router_run(torch::Tensor,torch::Tensor,int64_t);'
_FP32_ROUTER_CUDA='/* Adapted from Nathen Fernandes ROUTER_SCALAR: FP32 activation reads.\nMIT License\n\nCopyright (c) 2026 Nathen Fernandes\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n*/\n#include <cuda_runtime.h>\n#include <stdint.h>\n#include <math.h>\n\ntemplate<int M, int NT, int ST>\n__global__ void __launch_bounds__(NT,1) fp32_router(const float* __restrict__ w, const float* __restrict__ x,\n                                float* __restrict__ y, int m) {\n  constexpr int K=8192;\n  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;\n  float acc[M][4]={};\n  float4 ww[ST];\n  float4 xx[ST][M];\n  auto ld=[&](int slot,int i) {\n    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];\n    #pragma unroll\n    for(int t=0;t<M;t++) {\n      if(t<m) xx[slot][t]=reinterpret_cast<const float4*>(x+t*K)[i];\n    }\n  };\n  #pragma unroll\n  for(int s=0;s<ST;s++) ld(s,tid+s*NT);\n  #pragma unroll 1\n  for(int base=tid;base<K/4;base+=ST*NT) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      float4 v=ww[s];\n      float4 a[M];\n      #pragma unroll\n      for(int t=0;t<M;t++) a[t]=xx[s][t];\n      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);\n      #pragma unroll\n      for(int t=0;t<M;t++) if(t<m) {\n        acc[t][0]=fmaf(v.x,a[t].x,acc[t][0]);\n        acc[t][1]=fmaf(v.y,a[t].y,acc[t][1]);\n        acc[t][2]=fmaf(v.z,a[t].z,acc[t][2]);\n        acc[t][3]=fmaf(v.w,a[t].w,acc[t][3]);\n      }\n    }\n  }\n  __shared__ float sums[M][NT/32];\n  #pragma unroll\n  for(int t=0;t<M;t++) if(t<m) {\n    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);\n    #pragma unroll\n    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n    if(lane==0) sums[t][warp]=v;\n  }\n  __syncthreads();\n  if(warp==0) {\n    #pragma unroll\n    for(int t=0;t<M;t++) if(t<m) {\n      float v=lane<NT/32?sums[t][lane]:0.f;\n      #pragma unroll\n      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n      if(lane==0) y[t*256+row]=v;\n    }\n  }\n}\n\nextern "C" int fp32_router_c(const float* W,const float* X,float* Y,int M,int cfg,cudaStream_t stream){\n#define R(MM,NT,ST,C) if(M<=MM&&cfg==C){fp32_router<MM,NT,ST><<<256,NT,0,stream>>>(W,X,Y,M);return cudaGetLastError();}\n if(M==1){R(1,128,2,0) R(1,256,1,1) R(1,256,2,2) R(1,512,1,3) R(1,512,2,4)}\n else if(M<=8){R(8,128,2,0) R(8,256,1,1) R(8,256,2,2) R(8,512,1,3) R(8,512,2,4)}\n#undef R\n return cudaErrorInvalidValue;\n}\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\nextern "C" int fp32_router_c(const float*,const float*,float*,int,int,cudaStream_t);\ntorch::Tensor fp32_router_run(torch::Tensor W,torch::Tensor X,int64_t cfg){\n auto Y=torch::empty({X.size(0),256},X.options());\n int e=fp32_router_c(W.data_ptr<float>(),X.data_ptr<float>(),Y.data_ptr<float>(),X.size(0),cfg,at::cuda::getCurrentCUDAStream());\n TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));return Y;\n}\n'

_FP32_ROUTER_MOD=None
def _fp32_router_module():
    global _FP32_ROUTER_MOD
    if _FP32_ROUTER_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _FP32_ROUTER_MOD=load_inline('muse_decoder_fp32_router',_FP32_ROUTER_CPP,cuda_sources=_FP32_ROUTER_CUDA,functions=['fp32_router_run'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
    return _FP32_ROUTER_MOD
_before_fp32_router_prepare=prepare
def prepare(weights):
    state=_before_fp32_router_prepare(weights)
    state['fp32_router_mod']=_fp32_router_module()
    return state
