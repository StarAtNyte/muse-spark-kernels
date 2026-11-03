#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace muse {

constexpr int N_EXPERTS = 256;
constexpr int LOCAL_EXPERTS = 32;
constexpr int TOP_K = 8;
constexpr float ROUTER_MULT = 0.1767766952966369f; // 1 / sqrt(32)
constexpr int SK_T = 8;     // Skinny threshold
constexpr int FT_T = 64;    // Fat chunk size
constexpr int JMAX = 64;    // Maximum jobs per CTA
constexpr int EMAX = 2048;  // Maximum token-expert slots (256 * 8)

// -----------------------------------------------------------------------------
// Dispatch Descriptor: Holds all routing and job schedule state in Shared Memory
// Zero global memory roundtrips required for downstream MoE computation.
// -----------------------------------------------------------------------------
struct SmemDispatchDescriptor {
  // Local expert route distribution
  int cnt[LOCAL_EXPERTS];      // Tokens routed to each local expert
  int off[LOCAL_EXPERTS];      // Starting offset of each local expert in pair table
  int total_local;             // Total local route count

  // Token-expert mapping tables
  signed char eid[EMAX];       // Local expert ID (0..31) or -1 if non-local
  short ent[EMAX];             // Sorted entry: pair_idx -> (token * 8 + slot)
  short pos[EMAX];             // Direct map: (token * 8 + slot) -> pair_idx (-1 if non-local)
  float weights[EMAX];         // Normalized routing weights stored directly in smem

  // MoE execution job queue
  int njF;                     // Count of fat jobs (> SK_T tokens)
  int njS;                     // Count of skinny jobs (<= SK_T tokens)
  short jE[JMAX];              // Job -> local expert index
  short jS[JMAX];              // Job -> pair start offset
  short jC[JMAX];              // Job -> token count
};

// -----------------------------------------------------------------------------
// Monotonic float-to-uint32 transformation for bit-exact comparison
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t float_to_ordered_uint(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float ordered_uint_to_float(uint32_t u) {
  uint32_t f_u = (u & 0x80000000u) ? (u & ~0x80000000u) : ~u;
  return __uint_as_float(f_u);
}

// -----------------------------------------------------------------------------
// Fast numerical sigmoid
// -----------------------------------------------------------------------------
__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// -----------------------------------------------------------------------------
// Cooperative Warp Top-8 Extraction:
// 32 threads in a warp hold 256 experts (8 experts per thread).
// Extracts bit-exact top-8 expert IDs and unbiased sigmoid weights in registers.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void warp_cooperative_top8(
    const float* __restrict__ logits_token,
    const float* __restrict__ bias,
    int lane,
    int out_expert[8],
    float out_prob[8],
    float out_weight[8])
{
  float prob[8];
  float score[8];

  // 1. Compute probabilities and biased selection scores
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    int e = lane + 32 * j;
    float lg = logits_token[e];
    float x = lg * ROUTER_MULT;
    prob[j] = fast_sigmoid(x);
    score[j] = prob[j] + bias[e];
  }

  // 2. Iterative tournament: 8 rounds of cooperative max reduction
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    // Local thread maximum among its 8 elements
    float my_max_score = score[0];
    int my_max_exp = lane;
    #pragma unroll
    for (int j = 1; j < 8; ++j) {
      if (score[j] > my_max_score) {
        my_max_score = score[j];
        my_max_exp = lane + 32 * j;
      }
    }

    // Pack into uint64: high 32 bits = monotonic score key, low 32 bits = inverted expert (tie-break)
    uint32_t key = float_to_ordered_uint(my_max_score);
    uint32_t tie = (uint32_t)(255 - my_max_exp);
    uint64_t my_packed = ((uint64_t)key << 32) | (uint64_t)tie;

    // Warp-level reduction across 32 lanes
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      uint64_t other = __shfl_down_sync(0xffffffffu, my_packed, offset);
      if (other > my_packed) {
        my_packed = other;
      }
    }
    // Broadcast winner to all lanes
    my_packed = __shfl_sync(0xffffffffu, my_packed, 0);

    int win_exp = 255 - (int)(my_packed & 0xffffffffu);
    out_expert[k] = win_exp;

    // Retrieve winning thread's unbiased probability
    int win_lane = win_exp & 31;
    int win_slot = win_exp >> 5;
    float win_p = (lane == win_lane) ? prob[win_slot] : 0.0f;
    win_p = __shfl_sync(0xffffffffu, win_p, win_lane);
    out_prob[k] = win_p;

    // Mask winner from candidate scores
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (lane + 32 * j == win_exp) {
        score[j] = -1e30f;
      }
    }
  }

  // 3. Normalize top-8 weights in registers
  float p_sum = 0.0f;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    p_sum += out_prob[k];
  }
  float inv_sum = 1.0f / (p_sum + 1e-15f);
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    out_weight[k] = out_prob[k] * inv_sum;
  }
}

// -----------------------------------------------------------------------------
// Fused Gating, Top-K, and Shared Memory Dispatch Descriptor Kernel
// Operates on T tokens, T <= 256. Zero global memory writes required.
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256) fused_moe_router_dispatch_kernel(
    const float* __restrict__ logits,      // [T, 256] router logits
    const float* __restrict__ bias,        // [256] router bias
    int local_start,                       // Sharding start (e.g. 0, 32, ...)
    int T,                                 // Number of tokens
    SmemDispatchDescriptor* __restrict__ desc_out, // Optional diagnostic / external pointer
    int* __restrict__ diag_ids,            // Optional debug global output
    float* __restrict__ diag_weights)      // Optional debug global output
{
  __shared__ SmemDispatchDescriptor s_desc;
  __shared__ int warp_counts[8][LOCAL_EXPERTS]; // Up to 8 warps in block

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = blockDim.x >> 5;

  // Initialize shared memory tables
  if (tid < LOCAL_EXPERTS) {
    s_desc.cnt[tid] = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
      warp_counts[w][tid] = 0;
    }
  }
  __syncthreads();

  // Each warp processes tokens cooperatively
  for (int t = warp; t < T; t += num_warps) {
    int sel_exp[8];
    float sel_prob[8];
    float sel_weight[8];

    const float* logits_tok = logits + t * N_EXPERTS;
    warp_cooperative_top8(logits_tok, bias, lane, sel_exp, sel_prob, sel_weight);

    // Threads in the warp handle the 8 selected slots
    if (lane < 8) {
      int exp = sel_exp[lane];
      float w = sel_weight[lane];
      int slot_idx = t * TOP_K + lane;

      // Check if expert falls into local shard
      int loc = exp - local_start;
      bool is_local = (loc >= 0 && loc < LOCAL_EXPERTS);
      signed char le = is_local ? (signed char)loc : (signed char)-1;

      s_desc.eid[slot_idx] = le;
      s_desc.weights[slot_idx] = w;

      if (is_local) {
        atomicAdd(&warp_counts[warp][loc], 1);
      }

      // Write diagnostic outputs if requested
      if (diag_ids != nullptr) {
        diag_ids[slot_idx] = exp;
      }
      if (diag_weights != nullptr) {
        diag_weights[slot_idx] = w;
      }
    }
  }
  __syncthreads();

  // Consolidate expert counts and compute exclusive scan across local experts
  if (tid < LOCAL_EXPERTS) {
    int sum = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
      sum += warp_counts[w][tid];
    }
    s_desc.cnt[tid] = sum;

    // Warp scan for prefix sums across 32 local experts
    int inc = sum;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v = __shfl_up_sync(0xffffffffu, inc, d);
      if (tid >= d) inc += v;
    }
    s_desc.off[tid] = inc - sum;
    if (tid == 31) {
      s_desc.total_local = inc;
    }
  }
  __syncthreads();

  // Build job schedule descriptors directly in shared memory
  if (warp == 0) {
    int tot = (lane < LOCAL_EXPERTS) ? s_desc.cnt[lane] : 0;
    int off = (lane < LOCAL_EXPERTS) ? s_desc.off[lane] : 0;

    int nfj = (tot > SK_T) ? (tot + FT_T - 1) / FT_T : 0;
    int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;

    int fi = nfj, si = nsj;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d);
      int v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    int nfTot = __shfl_sync(0xffffffffu, fi, 31);
    int nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj;
    int sj = nfTot + si - nsj;

    for (int q = 0; q < nfj; ++q) {
      s_desc.jE[fj + q] = (short)lane;
      s_desc.jS[fj + q] = (short)(off + q * FT_T);
      s_desc.jC[fj + q] = (short)min(FT_T, tot - q * FT_T);
    }
    if (nsj) {
      s_desc.jE[sj] = (short)lane;
      s_desc.jS[sj] = (short)off;
      s_desc.jC[sj] = (short)tot;
    }
    if (lane == 0) {
      s_desc.njF = nfTot;
      s_desc.njS = nsTot;
    }
  }
  __syncthreads();

  // Populate pair-to-slot mapping (ent) and slot-to-pair mapping (pos)
  // Re-use warp_counts as private allocation pointers
  if (tid < LOCAL_EXPERTS) {
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
      warp_counts[w][tid] = 0;
    }
  }
  __syncthreads();

  const int total_slots = T * TOP_K;
  for (int i = tid; i < total_slots; i += blockDim.x) {
    signed char le = s_desc.eid[i];
    if (le >= 0) {
      int my_off = atomicAdd(&s_desc.off[le], 1);
      s_desc.ent[my_off] = (short)i;
      s_desc.pos[i] = (short)my_off;
    } else {
      s_desc.pos[i] = -1;
    }
  }
  __syncthreads();

  // Re-restore s_desc.off after atomic reservation
  if (tid < LOCAL_EXPERTS) {
    s_desc.off[tid] -= s_desc.cnt[tid];
  }
  __syncthreads();

  // If external descriptor pointer provided, export final shared memory state
  if (desc_out != nullptr && blockIdx.x == 0 && tid == 0) {
    *desc_out = s_desc;
  }
}

} // namespace muse
