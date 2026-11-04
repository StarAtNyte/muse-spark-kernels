

# =====================================================================================================
# Batch-1 latency kernel (attn_b1_v3): a separate, compact module used only when B == 1 (16 query heads,
# 2 KV heads, head dim 64, q_len 1 or 8). Every other shape takes the parent path above, unchanged.
#
# One CTA per contiguous key chunk (end-aligned, so the last CTA owns the block's new tokens):
#   1. speculative page-index loads (seq = page-table capacity - L; verified, re-issued on a miss)
#      overlapped with the q / gate / new-K/V loads,
#   2. every K/V row of the chunk gathered with cp.async the moment the indices land (one mbarrier
#      per 64-key tile), q norm + RoPE + FP8 and the new K/V (norm, RoPE, FP8, cache write) meanwhile,
#   3. QK (FP8 mma.sync, FP32 accumulate), online softmax (exp2), PV (FP16 P x exact FP16 V, FP32
#      accumulate); several warp streams per row block merged in shared memory,
#   4. one compact FP32 partial (acc[64], m, l) per (row, CTA) written with plain relaxed stores: no
#      atomics, no fence, no ticket. The workspace holds a NaN sentinel between calls; consumers poll the
#      data words themselves (each word carries its own validity), then reset them to the sentinel,
#   5. CTA c finalizes rows [c * rpc, (c + 1) * rpc): merge the n partials (exact max rescale),
#      normalize, rmsnorm * sigmoid(gate), store BF16.
# =====================================================================================================

B1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#ifndef B1_PROF
#define B1_PROF 0
#endif

namespace {

constexpr int ROWB = 128;                 // bytes per cache row: 2 KV heads x 64 FP8
constexpr int TILE = 64;                  // keys per tile
constexpr int TKV = TILE * ROWB;          // 8 KB: one tile of K (V follows)
constexpr int TSTRIDE = 2 * TKV;          // 16 KB per tile in shared memory
constexpr int REC = 68;                   // floats per partial record: acc[64], m, l, 0, 0
constexpr int QSTR = 80;                  // shared-memory row stride of the FP8 queries

struct B1P {
  const __nv_bfloat16* qkv;
  const int* seq_lens;
  const int64_t* cache_loc;
  uint8_t* kc;
  uint8_t* vc;
  const int* pt;
  const float* ksp;
  const float* vsp;
  const float* inv_freq;
  __nv_bfloat16* out;
  float* ws;
  long long* dbg;
  int ld, max_pages, lp, window, rope, spec, T, chunk, n, rpc;
  float scale_log2;
  int off_q, off_idx, off_bar, off_gate, off_w, off_ls, off_red;
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cpa_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(su32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void cpa_mbar_arrive(uint64_t* b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int par) {
  asm volatile("{\n.reg .pred q;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W_%=;\n}\n"
               ::"r"(su32(b)), "r"(par) : "memory");
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}
__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {_, t}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {t, _}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t lds32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }
// V B-fragments (m16n8k16) for 16 keys from kb, head hk; n-tile t of thread group g holds dim 8g + t
__device__ __forceinline__ void v_frags(const uint8_t* V, int kb, int hk, int g, int t4, uint32_t* b0, uint32_t* b1) {
  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;
  const int k0 = kb + 2 * t4;
  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);
  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);
  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);
  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);
  uint32_t x;
  x = prmt(v0.x, v1.x, 0x5140); b0[0] = cvt2(x); b0[1] = cvt2h(x);
  x = prmt(v0.x, v1.x, 0x7362); b0[2] = cvt2(x); b0[3] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x5140); b0[4] = cvt2(x); b0[5] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x7362); b0[6] = cvt2(x); b0[7] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x5140); b1[0] = cvt2(x); b1[1] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x7362); b1[2] = cvt2(x); b1[3] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x5140); b1[4] = cvt2(x); b1[5] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x7362); b1[6] = cvt2(x); b1[7] = cvt2h(x);
}
#ifndef B1_WEAK
#define B1_WEAK 0
#endif
#if B1_WEAK
__device__ __forceinline__ void st_rel4(float* a, float x, float y, float z, float w) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ uint4 ld_rel4(const float* a) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_sent4(float* a) {
  asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" ::"l"(a), "r"(0xFFFFFFFFu) : "memory");
}
#else
__device__ __forceinline__ void st_rel4(float* a, float x, float y, float z, float w) {
  asm volatile("st.relaxed.gpu.global.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ uint4 ld_rel4(const float* a) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_sent4(float* a) {
  asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1, %1, %1, %1};\n" ::"l"(a), "r"(0xFFFFFFFFu) : "memory");
}
#endif
__device__ __forceinline__ bool is_sent(const uint4& v) {
  return (v.x == 0xFFFFFFFFu) | (v.y == 0xFFFFFFFFu) | (v.z == 0xFFFFFFFFu) | (v.w == 0xFFFFFFFFu);
}
__device__ __noinline__ void b1_spin_fail() { __trap(); }
__device__ __forceinline__ uint4 ldg16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ void bf8(const uint4& u, float* x) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[2 * i] = __uint_as_float(w[i] << 16);
    x[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
  }
}
__device__ __forceinline__ uint2 fp8x8(const float* x) {
  uint32_t r[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(x[4 * h], x[4 * h + 1]), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(x[4 * h + 2], x[4 * h + 3]), __NV_SATFINITE, __NV_E4M3);
    r[h] = lo | (hi << 16);
  }
  return make_uint2(r[0], r[1]);
}
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#if B1_PROF
#define STAMP(k) do { if (threadIdx.x == 0) p.dbg[blockIdx.x * 16 + (k)] = gtime(); } while (0)
#define STAMPW(k) do { if (lane == 0) p.dbg[blockIdx.x * 16 + (k)] = gtime(); } while (0)
#else
#define STAMP(k) do {} while (0)
#define STAMPW(k) do {} while (0)
#endif

// interleaved RoPE of dims 8k..8k+7 (pairs 4k..4k+3): the FP32 angle pos * inv_freq as the reference builds it,
// reduced to [-pi, pi] in double precision, then the fast sin/cos (abs error ~1e-6 there): no sincosf slow path
__device__ __forceinline__ void rope8(float* x, int pos, const float4& ifr) {
  const float fr[4] = {ifr.x, ifr.y, ifr.z, ifr.w};
  const float pf = (float)pos;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const double a = (double)(pf * fr[i]);
    const double k = rint(a * 0.15915494309189535);
    const float r = (float)fma(-k, 6.283185307179586, a);
    float sn, cs;
    __sincosf(r, &sn, &cs);
    const float u = x[2 * i], v = x[2 * i + 1];
    x[2 * i] = u * cs - v * sn;
    x[2 * i + 1] = u * sn + v * cs;
  }
}

// page entries of [base, base + chunk) into IDX (one 4-byte cp.async each), one commit group
template <int NT>
__device__ __forceinline__ void issue_idx(const B1P& p, int* IDX, int base, int tid) {
  const int pg0 = base >> p.lp, pg1 = (base + p.chunk - 1) >> p.lp;
  for (int i = tid; i <= pg1 - pg0; i += NT) {
    const int pg = pg0 + i;
    if (pg >= 0 && pg < p.max_pages) cpa4(su32(IDX + i), p.pt + pg);
  }
  cpa_commit();
}

// gather every cached K/V row of the chunk (8 threads per 128-byte row, or 4 per head half when HS);
// zero the rows outside the attended range; leave the block's new tokens to the register path
template <int L, bool HS, int NT>
__device__ __forceinline__ void issue_kv(const B1P& p, uint8_t* R0, const int* IDX, uint64_t* BAR, int base, int seq,
                                         int hc, int tid) {
  constexpr int CPR = HS ? 4 : 8;
  constexpr int RPP = NT / CPR;
  const int q = tid % CPR;
  const int qc = HS ? hc * 4 + q : q;
  const int lo = p.window ? max(0, seq - p.window + 1) : 0;
  const int pg0 = base >> p.lp;
  const int pm = (1 << p.lp) - 1;
  for (int tl = 0; tl < p.T; ++tl) {
#pragma unroll 4
    for (int s = tl * TILE + tid / CPR; s < (tl + 1) * TILE; s += RPP) {
      const int kp = base + s;
      uint8_t* dk = R0 + tl * TSTRIDE + (s & 63) * ROWB + ((qc ^ (s & 7)) << 4);
      const bool in = s < p.chunk;
      if (in && kp >= lo && kp < seq) {
        const int64_t row = ((int64_t)IDX[(kp >> p.lp) - pg0] << p.lp) | (int64_t)(kp & pm);
        cpa16(su32(dk), p.kc + row * ROWB + qc * 16);
        cpa16(su32(dk + TKV), p.vc + row * ROWB + qc * 16);
      } else if (!(in && kp >= seq && kp < seq + L)) {
        *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dk + TKV) = make_uint4(0, 0, 0, 0);
      }
    }
    cpa_mbar_arrive(&BAR[tl]);
  }
}

// the speculated length was wrong (page-table capacity != seq + L): drain, reload the indices, gather again
template <int L, bool HS, int NT>
__device__ __noinline__ void b1_regather(const B1P& p, uint8_t* R0, int* IDX, uint64_t* BAR, int base, int seq, int hc,
                                         int tid) {
  cpa_wait_all();
  for (int tl = 0; tl < p.T; ++tl) mbar_wait(&BAR[tl], 0);
  __syncthreads();
  issue_idx<NT>(p, IDX, base, tid);
  cpa_wait<0>();
  __syncthreads();
  issue_kv<L, HS, NT>(p, R0, IDX, BAR, base, seq, hc, tid);
}

template <int L, int NS, bool HS>
__global__ void __launch_bounds__((HS ? 1 : 2) * (L == 1 ? 1 : 4) * NS * 32, 1) b1_attn(const __grid_constant__ B1P p) {
  constexpr int MT = (L == 1) ? 1 : 4;       // 16-row m-tiles per KV head
  constexpr int G = HS ? 1 : 2;              // KV heads per CTA
  constexpr int NT = G * MT * NS * 32;
  constexpr int NW = NT / 32;
  constexpr int RH = 8 * L;                  // rows per KV head (8 query heads x L tokens)
  constexpr int R = 16 * L;                  // rows in all
  constexpr int QCH = G * RH * 8;            // 16-byte query chunks of this CTA
  constexpr int QPT = (QCH + NT - 1) / NT;
  constexpr int KCH = L * G * 8;             // 16-byte new-K chunks (same for V)
  static_assert(2 * KCH <= NT, "new K/V chunks: one per thread");
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* R0 = smem;
  uint8_t* Q8 = smem + p.off_q;
  int* IDX = reinterpret_cast<int*>(smem + p.off_idx);
  uint64_t* BAR = reinterpret_cast<uint64_t*>(smem + p.off_bar);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int cb = blockIdx.x;
  const int hc = HS ? cb / p.n : 0;
  const int ci = HS ? cb - hc * p.n : cb;
  const int nrec = p.n;
  STAMP(0);

  // ---- 1. loads: speculative page indices first, then gate (consumer rows), then registers ----
  if (tid < p.T) mbar_init(&BAR[tid], NT);
  int base = p.spec + L - (p.n - ci) * p.chunk;
  issue_idx<NT>(p, IDX, base, tid);
  const int r0 = cb * p.rpc;
  const int nrows = r0 < R ? min(p.rpc, R - r0) : 0;
  uint8_t* GATE = smem + p.off_gate;
  for (int e = tid; e < nrows * 8; e += NT) {
    const int rr = r0 + (e >> 3), k = e & 7;
    const int h = rr / RH, i = rr % RH, j = i >> 3, qh = h * 8 + (i & 7);
    cpa16(su32(GATE + e * 16), p.qkv + (size_t)j * p.ld + 1280 + qh * 64 + k * 8);
  }
  cpa_commit();
  const int seq = __ldg(p.seq_lens);
  const float ks = __ldg(p.ksp), vs = __ldg(p.vsp);
  uint4 qraw[QPT];
#pragma unroll
  for (int m = 0; m < QPT; ++m) {
    const int e = tid + m * NT;
    if (e < QCH) {
      const int rl = e >> 3, k = e & 7;
      const int hh = HS ? hc : rl / RH, i = rl % RH, j = i >> 3, qh = hh * 8 + (i & 7);
      qraw[m] = ldg16(p.qkv + (size_t)j * p.ld + qh * 64 + k * 8);
    }
  }
  float4 ifr = make_float4(0.f, 0.f, 0.f, 0.f);
  if (p.rope) ifr = __ldg(reinterpret_cast<const float4*>(p.inv_freq) + (tid & 7));
  const bool last = ci == p.n - 1;
  const bool kvt = last && tid < 2 * KCH;
  uint4 kvraw = make_uint4(0, 0, 0, 0);
  int64_t cl = 0;
  {
    const bool isk = tid < KCH;
    const int e = isk ? tid : tid - KCH;
    const int j = e / (G * 8), hh = HS ? hc : (e >> 3) % G, k = e & 7;
    if (kvt) {
      kvraw = ldg16(p.qkv + (size_t)j * p.ld + (isk ? 1024 : 1152) + hh * 64 + k * 8);
      cl = __ldg(p.cache_loc + j);
    }
  }

  cpa_wait<1>();
  __syncthreads();
  STAMP(1);
  // ---- 2. gathers on the speculative range; verify the length ----
  issue_kv<L, HS, NT>(p, R0, IDX, BAR, base, p.spec, hc, tid);
  int par = 0;
  if (seq != p.spec) {
    base = seq + L - (p.n - ci) * p.chunk;
    b1_regather<L, HS, NT>(p, R0, IDX, BAR, base, seq, hc, tid);
    par = 1;
  }
  STAMP(2);
  // queries: rmsnorm, RoPE, E4M3 (qk_scale_factor folded into the score scale)
#pragma unroll
  for (int m = 0; m < QPT; ++m) {
    const int e = tid + m * NT;
    if (e < QCH) {
      const int rl = e >> 3, k = e & 7, j = (rl % RH) >> 3;
      float x[8];
      bf8(qraw[m], x);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) ss += x[i] * x[i];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] *= rs;
      if (p.rope) rope8(x, seq + j, ifr);
      *reinterpret_cast<uint2*>(Q8 + rl * QSTR + k * 8) = fp8x8(x);
    }
  }
  // the block's new K / V (last split only): norm + RoPE for K, FP8, into the cache and the tile
  if (kvt) {
    const bool isk = tid < KCH;
    const int e = isk ? tid : tid - KCH;
    const int j = e / (G * 8), hh = HS ? hc : (e >> 3) % G, k = e & 7;
    float x[8];
    bf8(kvraw, x);
    if (isk) {
      const unsigned gm = 0xFFu << (lane & 24);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) ss += x[i] * x[i];
      ss += __shfl_xor_sync(gm, ss, 1);
      ss += __shfl_xor_sync(gm, ss, 2);
      ss += __shfl_xor_sync(gm, ss, 4);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] *= rs;
      if (p.rope) rope8(x, seq + j, ifr);
    }
    const float rsc = __frcp_rn(isk ? ks : vs);   // (exact for the power-of-two scales)
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] *= rsc;
    const uint2 w = fp8x8(x);
    *reinterpret_cast<uint2*>((isk ? p.kc : p.vc) + cl * ROWB + hh * 64 + k * 8) = w;
    const int s = seq + j - base, tl = s >> 6, row = s & 63, qc = hh * 4 + (k >> 1);
    *reinterpret_cast<uint2*>(R0 + tl * TSTRIDE + (isk ? 0 : TKV) + row * ROWB + ((qc ^ (row & 7)) << 4) + (k & 1) * 8) = w;
  }
  __syncthreads();

  // ---- 3. attention over this chunk: warp (stream sid, head hl, m-tile mt) ----
  const int sid = warp / (G * MT);
  const int hl = (warp / MT) % G;
  const int h = HS ? hc : hl;
  const int mt = warp % MT;
  uint32_t qa[2][4];
  {
    const uint8_t* q0 = Q8 + (hl * RH + mt * 16 + g) * QSTR + 4 * t4;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      qa[s][0] = lds32(q0 + s * 32);
      qa[s][2] = lds32(q0 + s * 32 + 16);
      qa[s][1] = (L == 8) ? lds32(q0 + 8 * QSTR + s * 32) : 0u;
      qa[s][3] = (L == 8) ? lds32(q0 + 8 * QSTR + s * 32 + 16) : 0u;
    }
  }
  const int j0 = (L == 1) ? 0 : 2 * mt;
  const int cmin = max(0, base), cmax = base + p.chunk - 1;
  const int hi0 = min(seq + j0, cmax), hi1 = min(seq + j0 + 1, cmax);
  const int lo0 = max(cmin, p.window ? seq + j0 - p.window + 1 : 0);
  const int lo1 = max(cmin, p.window ? seq + j0 + 1 - p.window + 1 : 0);
  const float sl = p.scale_log2 * ks;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  float o[8][4];
#pragma unroll
  for (int d = 0; d < 8; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  bool first = true;
  for (int u = sid; u < 2 * p.T; u += NS) {
    const int tl = u >> 1, kh = (u & 1) << 5;
    mbar_wait(&BAR[tl], par);
#if B1_PROF
    if (first && warp == 0) STAMPW(3);
#endif
    first = false;
    const uint8_t* Kt = R0 + tl * TSTRIDE;
    const uint8_t* Vt = Kt + TKV;
    float sc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
      const uint8_t* kr = Kt + (kh + nt * 8 + g) * ROWB + 4 * t4;
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const uint32_t b0 = lds32(kr + (((h * 4 + s * 2) ^ g) << 4));
        const uint32_t b1 = lds32(kr + (((h * 4 + s * 2 + 1) ^ g) << 4));
        mma_e4m3(sc[nt], qa[s], b0, b1);
      }
    }
    const int kb = base + tl * TILE + kh + 2 * t4;
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int kp = kb + nt * 8 + c;
        const float x0 = (kp >= lo0 && kp <= hi0) ? sc[nt][c] * sl : -INFINITY;
        sc[nt][c] = x0;
        mx0 = fmaxf(mx0, x0);
        if (L == 8) {
          const float x1 = (kp >= lo1 && kp <= hi1) ? sc[nt][2 + c] * sl : -INFINITY;
          sc[nt][2 + c] = x1;
          mx1 = fmaxf(mx1, x1);
        }
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    const float mn0 = fmaxf(m0, mx0), mu0 = (mn0 == -INFINITY) ? 0.f : mn0;
    const float f0 = ex2(m0 - mu0);
    m0 = mn0;
    float f1 = 1.f, mu1 = 0.f;
    if (L == 8) {
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      const float mn1 = fmaxf(m1, mx1);
      mu1 = (mn1 == -INFINITY) ? 0.f : mn1;
      f1 = ex2(m1 - mu1);
      m1 = mn1;
    }
    uint32_t pa[2][4];
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const float p0 = ex2(sc[nt][0] - mu0), p1 = ex2(sc[nt][1] - mu0);
      ps0 += p0 + p1;
      uint32_t hi = 0u;
      if (L == 8) {
        const float p2 = ex2(sc[nt][2] - mu1), p3 = ex2(sc[nt][3] - mu1);
        ps1 += p2 + p3;
        hi = pack_h2(p2, p3);
      }
      pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);
      pa[nt >> 1][(nt & 1) * 2 + 1] = hi;
    }
    l0 = l0 * f0 + ps0;
    l1 = l1 * f1 + ps1;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      o[d][0] *= f0; o[d][1] *= f0;
      if (L == 8) { o[d][2] *= f1; o[d][3] *= f1; }
    }
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t b0[8], b1[8];
      v_frags(Vt, kh + kk * 16, h, g, t4, b0, b1);
#pragma unroll
      for (int d = 0; d < 8; ++d) mma_f16(o[d], pa[kk], b0[d], b1[d]);
    }
  }
  STAMP(4);
  // ---- merge the NS streams of each row block in shared memory ----
  if (NS > 1) {
    constexpr int NSLOT = G * MT * 32;
    __syncthreads();
    float* XS = reinterpret_cast<float*>(R0);
    const int slot = (hl * MT + mt) * 32 + lane;
    if (sid > 0) {
      float* x = XS + (sid - 1) * 36 * NSLOT + slot;
      x[0] = m0; x[NSLOT] = m1; x[2 * NSLOT] = l0; x[3 * NSLOT] = l1;
#pragma unroll
      for (int e = 0; e < 32; ++e) x[(4 + e) * NSLOT] = o[e >> 2][e & 3];
    }
    __syncthreads();
    if (sid == 0) {
#pragma unroll 1
      for (int s2 = 1; s2 < NS; ++s2) {
        const float* x = XS + (s2 - 1) * 36 * NSLOT + slot;
        const float xm0 = x[0], xm1 = x[NSLOT], xl0 = x[2 * NSLOT], xl1 = x[3 * NSLOT];
        const float a0 = fmaxf(m0, xm0), u0 = (a0 == -INFINITY) ? 0.f : a0;
        const float fa0 = ex2(m0 - u0), fb0 = ex2(xm0 - u0);
        const float a1 = fmaxf(m1, xm1), u1 = (a1 == -INFINITY) ? 0.f : a1;
        const float fa1 = ex2(m1 - u1), fb1 = ex2(xm1 - u1);
        m0 = a0; m1 = a1;
        l0 = l0 * fa0 + xl0 * fb0;
        l1 = l1 * fa1 + xl1 * fb1;
#pragma unroll
        for (int e = 0; e < 32; ++e) {
          const float fa = (e & 2) ? fa1 : fa0, fb = (e & 2) ? fb1 : fb0;
          o[e >> 2][e & 3] = o[e >> 2][e & 3] * fa + x[(4 + e) * NSLOT] * fb;
        }
      }
    }
  }
  // ---- 4. the CTA's partial: plain relaxed stores (the data words are their own flags) ----
  if (sid == 0) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const int rr0 = h * RH + mt * 16 + g;
    float* rec = p.ws + ((size_t)rr0 * nrec + ci) * REC;
    st_rel4(rec + 16 * t4, o[0][0], o[1][0], o[2][0], o[3][0]);
    st_rel4(rec + 16 * t4 + 4, o[4][0], o[5][0], o[6][0], o[7][0]);
    st_rel4(rec + 16 * t4 + 8, o[0][1], o[1][1], o[2][1], o[3][1]);
    st_rel4(rec + 16 * t4 + 12, o[4][1], o[5][1], o[6][1], o[7][1]);
    if (t4 == 0) st_rel4(rec + 64, m0, l0, 0.f, 0.f);
    if (L == 8) {
      float* rec1 = rec + (size_t)8 * nrec * REC;
      st_rel4(rec1 + 16 * t4, o[0][2], o[1][2], o[2][2], o[3][2]);
      st_rel4(rec1 + 16 * t4 + 4, o[4][2], o[5][2], o[6][2], o[7][2]);
      st_rel4(rec1 + 16 * t4 + 8, o[0][3], o[1][3], o[2][3], o[3][3]);
      st_rel4(rec1 + 16 * t4 + 12, o[4][3], o[5][3], o[6][3], o[7][3]);
      if (t4 == 0) st_rel4(rec1 + 64, m1, l1, 0.f, 0.f);
    }
  }
  STAMP(5);
  // ---- 5. finalize this CTA's rows from all n partials ----
  if (nrows == 0) return;
  cpa_wait_all();   // the gate rows (and every gather) landed
  __syncthreads();
  float* CB = reinterpret_cast<float*>(R0);
  const int tot = nrows * nrec * (REC / 4);
  float* src = p.ws + (size_t)r0 * nrec * REC;
  for (int e0 = tid; e0 < tot; e0 += 4 * NT) {
    uint4 v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int e = e0 + k * NT;
      if (e < tot) v[k] = ld_rel4(src + 4 * e);
    }
    unsigned spins = 0;
    bool pend;
    do {
      pend = false;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int e = e0 + k * NT;
        if (e < tot && is_sent(v[k])) { v[k] = ld_rel4(src + 4 * e); pend = true; }
      }
      if (pend && ++spins > (1u << 24)) b1_spin_fail();
    } while (pend);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int e = e0 + k * NT;
      if (e < tot) *reinterpret_cast<uint4*>(CB + 4 * e) = v[k];
    }
  }
  __syncthreads();
  STAMP(6);
  float* W = reinterpret_cast<float*>(smem + p.off_w);
  float* LS = reinterpret_cast<float*>(smem + p.off_ls);
  float* RED = reinterpret_cast<float*>(smem + p.off_red);
  for (int row = warp; row < nrows; row += NW) {
    float M = -INFINITY;
    for (int c2 = lane; c2 < nrec; c2 += 32) M = fmaxf(M, CB[(row * nrec + c2) * REC + 64]);
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o2));
    float ls = 0.f;
    for (int c2 = lane; c2 < nrec; c2 += 32) {
      const float m = CB[(row * nrec + c2) * REC + 64];
      const float w = (m == -INFINITY) ? 0.f : ex2(m - M);
      W[row * nrec + c2] = w;
      ls += w * CB[(row * nrec + c2) * REC + 65];
    }
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o2);
    if (lane == 0) LS[row] = ls;
  }
  __syncthreads();
  STAMP(8);
  const int parts = nrows >= NW ? 1 : NW / nrows;
  for (int it = warp; it < nrows * parts; it += NW) {
    const int row = it % nrows, part = it / nrows;
    float ax = 0.f, ay = 0.f;
    for (int c2 = part; c2 < nrec; c2 += parts) {
      const float w = W[row * nrec + c2];
      const float2 x = *reinterpret_cast<const float2*>(CB + (row * nrec + c2) * REC + 2 * lane);
      ax += w * x.x;
      ay += w * x.y;
    }
    *reinterpret_cast<float2*>(RED + it * 64 + 2 * lane) = make_float2(ax, ay);
  }
  __syncthreads();
  STAMP(9);
  for (int row = warp; row < nrows; row += NW) {
    float ax = 0.f, ay = 0.f;
    for (int part = 0; part < parts; ++part) {
      const float2 x = *reinterpret_cast<const float2*>(RED + (part * nrows + row) * 64 + 2 * lane);
      ax += x.x;
      ay += x.y;
    }
    const float inv = __fdividef(vs, LS[row]);
    ax *= inv;
    ay *= inv;
    float ss = ax * ax + ay * ay;
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const uint32_t gw = *reinterpret_cast<const uint32_t*>(GATE + row * 128 + 4 * lane);
    const float gx = __uint_as_float(gw << 16), gy = __uint_as_float(gw & 0xFFFF0000u);
    const float sx = 1.f / (1.f + __expf(-gx)), sy = 1.f / (1.f + __expf(-gy));
    const int rr = r0 + row, hh = rr / RH, i = rr % RH, j = i >> 3, qh = hh * 8 + (i & 7);
    __nv_bfloat162 ov = __floats2bfloat162_rn(ax * rs * sx, ay * rs * sy);
    *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)j * 1024 + qh * 64 + 2 * lane) = ov;
  }
  STAMP(10);
  // reset the consumed records to the sentinel for the next call
  for (int e = tid; e < tot; e += NT) st_sent4(src + 4 * e);
  STAMP(7);
}

template <int L, int NS, bool HS>
void launch_b1(const B1P& p, int grid, int smem, cudaStream_t st) {
  constexpr int NT = (HS ? 1 : 2) * (L == 1 ? 1 : 4) * NS * 32;
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(b1_attn<L, NS, HS>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    init = true;
  }
  b1_attn<L, NS, HS><<<grid, NT, smem, st>>>(p);
}

}  // namespace

void run_b1(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
            torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
            torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws, std::optional<torch::Tensor> dbg,
            int64_t L, int64_t NS, int64_t HS, int64_t lp, int64_t window, int64_t rope, int64_t spec, int64_t T,
            int64_t chunk, int64_t n, int64_t rpc, double scale_log2, int64_t off_q, int64_t off_idx,
            int64_t off_bar, int64_t off_gate, int64_t off_w, int64_t off_ls, int64_t off_red, int64_t smem) {
  B1P p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.kc = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.vc = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.pt = page_table.data_ptr<int>();
  p.ksp = k_scale.data_ptr<float>();
  p.vsp = v_scale.data_ptr<float>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws = ws.data_ptr<float>();
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.ld = (int)qkv.size(-1);
  p.max_pages = (int)page_table.size(1);
  p.lp = (int)lp; p.window = (int)window; p.rope = (int)rope; p.spec = (int)spec; p.T = (int)T;
  p.chunk = (int)chunk; p.n = (int)n; p.rpc = (int)rpc; p.scale_log2 = (float)scale_log2;
  p.off_q = (int)off_q; p.off_idx = (int)off_idx; p.off_bar = (int)off_bar; p.off_gate = (int)off_gate;
  p.off_w = (int)off_w; p.off_ls = (int)off_ls; p.off_red = (int)off_red;
  const int grid = (int)(HS ? 2 * n : n);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int sm = (int)smem;
  if (L == 1) {
    if (HS) {
      if (NS == 1) launch_b1<1, 1, true>(p, grid, sm, st);
      else if (NS == 2) launch_b1<1, 2, true>(p, grid, sm, st);
      else launch_b1<1, 4, true>(p, grid, sm, st);
    } else {
      if (NS == 2) launch_b1<1, 2, false>(p, grid, sm, st);
      else launch_b1<1, 4, false>(p, grid, sm, st);
    }
  } else {
    if (HS) {
      if (NS == 1) launch_b1<8, 1, true>(p, grid, sm, st);
      else launch_b1<8, 2, true>(p, grid, sm, st);
    } else {
      if (NS == 1) launch_b1<8, 1, false>(p, grid, sm, st);
      else launch_b1<8, 2, false>(p, grid, sm, st);
    }
  }
}
"""

B1_CPP_SRC = r"""
void run_b1(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
            torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
            torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws, std::optional<torch::Tensor> dbg,
            int64_t L, int64_t NS, int64_t HS, int64_t lp, int64_t window, int64_t rope, int64_t spec, int64_t T,
            int64_t chunk, int64_t n, int64_t rpc, double scale_log2, int64_t off_q, int64_t off_idx,
            int64_t off_bar, int64_t off_gate, int64_t off_w, int64_t off_ls, int64_t off_red, int64_t smem);
"""

_B1_MOD = {}
_B1_WS = {}
_B1_PLANS = {}
B1_SMEM_MAX = 227 * 1024
B1_MAX_CTAS = 132
B1_REC = 68
# (q_len, sliding) -> plan knobs: NS warp streams per row block, HS head split (one KV head per CTA),
# n CTAs per head (None: one 64-key tile per CTA on sliding layers, a CTA per SM on full layers)
B1_KNOBS = {
    (1, True): dict(NS=2, HS=0, n=None),
    (8, True): dict(NS=2, HS=0, n=None),
    (1, False): dict(NS=4, HS=0, n=None),
    (8, False): dict(NS=2, HS=0, n=None),
}
B1_FULL = True   # route batch-1 full-attention layers to the batch-1 kernel


def _b1_module(flags=()):
    key = tuple(flags)
    mod = _B1_MOD.get(key)
    if mod is None:
        from torch.utils.cpp_extension import load_inline
        tag = "".join(f.replace("-D", "_").replace("=", "") for f in flags)
        mod = load_inline(name=f"okc_attn_b1v3{tag}_90a", cpp_sources=B1_CPP_SRC, cuda_sources=B1_CUDA_SRC,
                          functions=["run_b1"],
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", *flags], verbose=False)
        _B1_MOD[key] = mod
    return mod


def _b1_ws(n_floats, device):
    """FP32 partial records, NaN sentinel (all ones) between calls: every consumer restores what it read."""
    key = str(device)
    ws = _B1_WS.get(key)
    if ws is None or ws.numel() < n_floats:
        n = max(n_floats, 128 * B1_MAX_CTAS * B1_REC)
        ws = torch.full((n,), -1, dtype=torch.int32, device=device).view(torch.float32)
        _B1_WS[key] = ws
    return ws


def _al(x, a):
    return (x + a - 1) // a * a


def _b1_plan(L, window, max_pages, P, NS=None, HS=None, n=None):
    key = (L, window, max_pages, P, NS, HS, n)
    plan = _B1_PLANS.get(key)
    if plan is not None or key in _B1_PLANS:
        return plan
    knobs = B1_KNOBS[(L, bool(window))]
    NS = knobs["NS"] if NS is None else NS
    HS = knobs["HS"] if HS is None else HS
    n = knobs["n"] if n is None else n
    cap = max_pages * P
    keys_max = min(window + L - 1, cap) if window else cap
    per_head_max = B1_MAX_CTAS // (2 if HS else 1)
    if n is None:
        if window:
            n = -(-keys_max // 64)
        else:
            n = per_head_max if L == 1 else (per_head_max if HS else 128)
    n = max(1, min(n, per_head_max, -(-keys_max // 16)))
    chunk = -(-keys_max // n)
    T = -(-chunk // 64)
    MT = 1 if L == 1 else 4
    G = 1 if HS else 2
    NT = G * MT * NS * 32
    NW = NT // 32
    R = 16 * L
    rpc = -(-R // (n * (2 if HS else 1)))
    RH = 8 * L
    r0 = max(T * 16384, (NS - 1) * 36 * G * MT * 32 * 4, rpc * n * B1_REC * 4)
    off_q = _al(r0, 128)
    off_idx = _al(off_q + G * RH * 80, 16)
    off_bar = _al(off_idx + (chunk // P + 2) * 4, 8)
    off_gate = _al(off_bar + T * 8, 16)
    off_w = _al(off_gate + rpc * 128, 16)
    off_ls = _al(off_w + rpc * n * 4, 16)
    off_red = _al(off_ls + rpc * 4, 16)
    smem = _al(off_red + max(NW, rpc) * 64 * 4, 16)
    ok = (smem <= B1_SMEM_MAX and NT <= 1024 and keys_max + L < (1 << 30) and
          (NS in (2, 4) if (L == 1 and not HS) else NS in ((1, 2, 4) if L == 1 else (1, 2))))
    plan = dict(NS=NS, HS=HS, n=n, chunk=chunk, T=T, rpc=rpc, off_q=off_q, off_idx=off_idx, off_bar=off_bar,
                off_gate=off_gate, off_w=off_w, off_ls=off_ls, off_red=off_red, smem=smem,
                ws=R * n * B1_REC) if ok else None
    _B1_PLANS[key] = plan
    return plan


def _b1_supported(data):
    cfg = data["config"]
    return (cfg["batch"] == 1 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64
            and cfg["q_len"] in (1, 8) and (B1_FULL or cfg["window"]) and data["qkv"].is_contiguous()
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].is_contiguous())


def _b1_launch(data, dbg=None, flags=(), **knobs):
    cfg = data["config"]
    L, P, window = cfg["q_len"], cfg["page_size"], cfg["window"]
    page_table = data["page_table"]
    plan = _b1_plan(L, window, page_table.shape[1], P, **knobs)
    if plan is None:
        return None
    qkv = data["qkv"]
    device = qkv.device
    out = torch.empty((1, L, 16 * 64), dtype=torch.bfloat16, device=device)
    ws = _b1_ws(plan["ws"], device)
    lp = P.bit_length() - 1
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    spec = page_table.shape[1] * P - L
    _b1_module(flags).run_b1(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                             data["k_scale"], data["v_scale"], _inv_freq(cfg["rope_theta"], device), out, ws, dbg,
                             L, plan["NS"], plan["HS"], lp, window, 1 if cfg["rope_theta"] else 0, spec, plan["T"],
                             plan["chunk"], plan["n"], plan["rpc"], scale_log2, plan["off_q"], plan["off_idx"],
                             plan["off_bar"], plan["off_gate"], plan["off_w"], plan["off_ls"], plan["off_red"],
                             plan["smem"])
    return out


def kernel_fn(data: Inputs) -> torch.Tensor:
    if not _supported(data["config"]) or not _fits(data):
        return _fallback(data)
    if _b1_supported(data):
        out = _b1_launch(data)
        if out is not None:
            return out
    return _launch(data)
