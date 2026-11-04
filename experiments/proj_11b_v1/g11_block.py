

# ======================================================================================================================
# proj_11b_v1 (WILD): lossless 11-bit BF16 weights for the byte-bound m <= 32 cells of qkv_gate / pooler / unpooler.
# Codec: each BF16 weight keeps its low byte (exponent LSB + 7 mantissa bits) raw, plus a 3-bit code: sign + 2-bit offset
# of the upper 7 exponent bits from a per-matrix base (a window of 8 consecutive exponents, picked to cover the most
# weights). The judge's weights are N(0, 1/K): the exponent entropy is 2.54 bits (10.46 bits per weight in all), and the
# best 8-exponent window misses 1.25-1.36% of the weights (all tiny magnitudes). Those escape: the stream holds them with
# a clamped offset, and a per-row list (k, original BF16 bits; 4 B each, ~0.44 bit per weight) lets dedicated escape
# warps add x[t, k] * (w - w_decoded) in FP32 (both products exact). 11.44 bits per weight against the parent's 12.
# Kernel g11: one 16-row tile per compute warp over a 1024-wide K range, A fragments decoded in registers (LOP3/SHF/PRMT,
# no tables), x^T in a swizzled smem slab (ldmatrix.trans B fragments, also the escape warps' lookups), K split over the
# S CTAs of a thread-block cluster, partials summed in fixed order through DSMEM and written once as BF16.
# prepare() proves each pack lossless (decode emulation), validates each path against FP64 (random rows, outliers,
# one-hot probes) and keeps a path only where it beats the dispatcher's own pick in paired cold-L2 graph timings.
_G11_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <type_traits>
#include <stdint.h>
namespace cg = cooperative_groups;

struct P11 { const uint32_t* W; const uint32_t* rp; const uint32_t* ent; const __nv_bfloat16* x; __nv_bfloat16* out;
             int m, N, K; uint32_t Tb, base; int ecap, flags; };

__device__ __forceinline__ uint4 g11_ld4(const uint32_t* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint32_t g11_ld1(const uint32_t* p) {
  uint32_t v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint32_t g11_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d;
}
__device__ __forceinline__ void g11_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void g11_ldsm4t(uint32_t* b, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
}
__device__ __forceinline__ uint4 g11_lds4(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
// x^T slab: row k holds TK tokens (16 B chunks of 8), chunk index XOR-swizzled so ldmatrix rows k..k+7 hit 8 distinct
// 16 B bank groups.
template <int TK> __device__ __forceinline__ uint32_t g11_soff(int k, int t) {
  const int c = t >> 3;
  const int f = TK == 32 ? ((k >> 1) & 3) : (TK == 16 ? ((k >> 2) & 1) : 0);
  return (uint32_t)(k * TK * 2 + ((c ^ f) << 4) + (t & 7) * 2);
}
// 32 values per lane per unit: low words L[0..7] (low bytes of values 4i..4i+3), code words C0..C2. Group i's high bytes
// are (X_i & 0x83838383) + base: bits 0-1 of each byte the exponent offset, bit 7 the sign (base + 3 <= 127, no carry).
__device__ __forceinline__ void g11_dec(const uint32_t* q, uint32_t Tb, uint32_t* a) {
  const uint32_t M = 0x83838383u, C0 = q[8], C1 = q[9], C2 = q[10];
  uint32_t H[8];
  H[0] = (C0 & M) + Tb;
  H[1] = (__funnelshift_r(C0, C0, 3) & M) + Tb;
  H[2] = (C1 & M) + Tb;
  H[3] = (__funnelshift_r(C1, C1, 3) & M) + Tb;
  H[4] = (C2 & M) + Tb;
  H[5] = (__funnelshift_r(C2, C2, 3) & M) + Tb;
  H[6] = ((__funnelshift_r(C0, C0, 5) & 0x03030303u) | (__funnelshift_l(C2, C2, 2) & 0x80808080u)) + Tb;
  H[7] = ((__funnelshift_r(C1, C1, 5) & 0x03030303u) | (__funnelshift_l(C2, C2, 1) & 0x80808080u)) + Tb;
#pragma unroll
  for (int i = 0; i < 8; i++) { a[2 * i] = g11_prmt(q[i], H[i], 0x5140); a[2 * i + 1] = g11_prmt(q[i], H[i], 0x7362); }
}

// XA: x B-fragments prefetched with the W unit (1) or loaded at use (0). Within a 64-wide chunk, logical k (step s, kl)
// sits at physical 16 t + 4 s + {0,1,2,3} for kl = {2t, 2t+1, 2t+8, 2t+9}: a lane's B elements are 32 contiguous bytes.
template <int MT, int RB, int EW, int NU, int S, int D, int XA>
__global__ void __launch_bounds__((RB + EW) * 32, 1) g11k(P11 p) {
  constexpr int TK = MT * 8, KL = NU * 64, R = RB * 16, NT = (RB + EW) * 32, RR = R / S, XD = XA ? XA : 1;
  static_assert(XA <= D, "x distance <= W distance");
  static_assert(R % S == 0, "rows per rank");
  extern __shared__ __align__(128) uint8_t sm[];
  float* PE = (float*)sm;  // [TK][R] fp32: escape sums, then + the MMA partials
  cg::cluster_group cl = cg::this_cluster();
  const int ks = (int)cl.block_rank(), rb = blockIdx.x / S, cta = blockIdx.x;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, tid = threadIdx.x, k0 = ks * KL, g = lane >> 2, t4 = lane & 3;
  float acc[MT][4];
  if (w < RB) {
    uint32_t wr[D][11];
    uint4 xb[XD][MT][2];
    const uint32_t* wp = p.W + ((size_t)(rb * RB + w) * S + ks) * NU * 352;
    const __nv_bfloat16* xp[MT];
#pragma unroll
    for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + k0 + t4 * 16;
    auto ldu = [&](int s, int u) {
      const uint32_t* b = wp + u * 352;
      const uint4 v0 = g11_ld4(b + lane * 4), v1 = g11_ld4(b + 128 + lane * 4);
      uint32_t* r = wr[s];
      r[0] = v0.x; r[1] = v0.y; r[2] = v0.z; r[3] = v0.w; r[4] = v1.x; r[5] = v1.y; r[6] = v1.z; r[7] = v1.w;
      r[8] = g11_ld1(b + 256 + lane); r[9] = g11_ld1(b + 288 + lane); r[10] = g11_ld1(b + 320 + lane);
    };
    auto ldx = [&](int s, int u) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s][n][0] = __ldg((const uint4*)(xp[n] + u * 64)); xb[s][n][1] = __ldg((const uint4*)(xp[n] + u * 64 + 8)); }
    };
#pragma unroll
    for (int d = 0; d < D; d++) ldu(d, d);
    if (XA) {
#pragma unroll
      for (int d = 0; d < XA; d++) ldx(d, d);
    }
#pragma unroll
    for (int n = 0; n < MT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
    for (int u = 0; u < NU; u++) {
      uint32_t a[16], B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XA) { b0 = xb[u % XD][n][0]; b1 = xb[u % XD][n][1]; }
        else { b0 = __ldg((const uint4*)(xp[n] + u * 64)); b1 = __ldg((const uint4*)(xp[n] + u * 64 + 8)); }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (XA && u + XA < NU) ldx(u % XD, u + XA);
      g11_dec(wr[u % D], p.Tb, a);
      if (u + D < NU) ldu(u % D, u + D);
#pragma unroll
      for (int s = 0; s < 4; s++)
#pragma unroll
        for (int n = 0; n < MT; n++) g11_mma(acc[n], a + 4 * s, B[n][2 * s], B[n][2 * s + 1]);
    }
  } else if (!(p.flags & 1)) {  // escape warps: lane-owned rows, sum x[t, k] * (w - w_decoded) over the row's escapes
    // this CTA's x slice [m][KL] and CSR slice (row offsets, 16 B aligned entry run) land in smem through cp.async,
    // off the compute warps' path (named barrier 1 among the escape warps only)
    uint32_t* srp = (uint32_t*)(sm + TK * R * 4);
    uint32_t* sen = srp + ((R + 4) & ~3);
    unsigned short* sx = (unsigned short*)(sen + ((p.ecap + 7) & ~3));
    const int et = (w - RB) * 32 + lane;
    constexpr int EN = EW * 32;
    const uint32_t* grp = p.rp + (size_t)cta * (R + 1);
    for (int i = et; i < p.m * (KL / 8); i += EN) {
      const int t = i / (KL / 8), c = i % (KL / 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((uint32_t)__cvta_generic_to_shared(sx + t * KL + c * 8)),
                   "l"(p.x + (size_t)t * p.K + k0 + c * 8) : "memory");
    }
    for (int i = et; i <= R; i += EN)
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((uint32_t)__cvta_generic_to_shared(srp + i)), "l"(grp + i) : "memory");
    const uint32_t eb = __ldg(grp), ne = __ldg(grp + R) - eb;
    for (uint32_t i = 4 * et; i < ne; i += 4 * EN)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((uint32_t)__cvta_generic_to_shared(sen + i)), "l"(p.ent + eb + i) : "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    asm volatile("bar.sync 1, %0;\n" :: "r"(EN) : "memory");
    // unconditional smem loads (rows >= m read row m - 1; their sums are never reduced): no branch per token
    auto rows = [&](auto ttc) {
      constexpr int TT = decltype(ttc)::value;
      int xo[TT];
#pragma unroll
      for (int t = 0; t < TT; t++) xo[t] = min(t, p.m - 1) * KL;
      for (int r = et; r < R; r += EN) {
        float e[TT];
#pragma unroll
        for (int t = 0; t < TT; t++) e[t] = 0.f;
        const uint32_t i0 = srp[r] - eb, i1 = srp[r + 1] - eb;
        for (uint32_t i = i0; i < i1; i += 2) {
          float wf[2], df[2];
          int kk[2];
#pragma unroll
          for (int q = 0; q < 2; q++) {
            const bool ok = i + q < i1;
            const uint32_t en = sen[ok ? i + q : i], wb = en & 0xffffu;
            const uint32_t h = (wb >> 8) & 0x7fu, hc = min(max(h, p.base), p.base + 3u);
            const uint32_t db = (wb & 0x80ffu) | (hc << 8);
            kk[q] = (int)(en >> 16);
            wf[q] = ok ? __uint_as_float(wb << 16) : 0.f;
            df[q] = ok ? __uint_as_float(db << 16) : 0.f;
          }
          float xv[2][TT];
#pragma unroll
          for (int q = 0; q < 2; q++)
#pragma unroll
            for (int t = 0; t < TT; t++) xv[q][t] = __uint_as_float(((uint32_t)sx[xo[t] + kk[q]]) << 16);
#pragma unroll
          for (int q = 0; q < 2; q++)
#pragma unroll
            for (int t = 0; t < TT; t++) { e[t] = fmaf(xv[q][t], wf[q], e[t]); e[t] = fmaf(-xv[q][t], df[q], e[t]); }
        }
#pragma unroll
        for (int t = 0; t < TT; t++) PE[t * R + r] = e[t];
      }
    };
    if (p.m == 1) rows(std::integral_constant<int, 1>{});
    else rows(std::integral_constant<int, TK>{});
  } else {
    for (int i = (w - RB) * 32 + lane; i < TK * R; i += EW * 32) PE[i] = 0.f;  // probe: no escapes (wrong, timing only)
  }
  __syncthreads();
  if (w < RB) {
    const int r0 = w * 16 + g;
#pragma unroll
    for (int n = 0; n < MT; n++) {
      const int tok = n * 8 + 2 * t4;
      PE[tok * R + r0] += acc[n][0]; PE[(tok + 1) * R + r0] += acc[n][1];
      PE[tok * R + r0 + 8] += acc[n][2]; PE[(tok + 1) * R + r0 + 8] += acc[n][3];
    }
  }
  cl.sync();
  for (int i = tid; i < p.m * RR; i += NT) {
    const int t = i / RR, r = ks * RR + i % RR;
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < S; q++) s += cl.map_shared_rank(PE, q)[t * R + r];
    p.out[(size_t)t * p.N + rb * R + r] = __float2bfloat16(s);
  }
  cl.sync();
}

template <int MT, int RB, int EW, int NU, int S, int D, int XA>
static void g11_launch(const P11& p, cudaStream_t st) {
  constexpr int TK = MT * 8, R = RB * 16;
  const int smem = TK * R * 4 + ((R + 4) & ~3) * 4 + ((p.ecap + 7) & ~3) * 4 + p.m * NU * 64 * 2;
  TORCH_CHECK(smem <= 227 * 1024, "g11: escape slice too large for smem");
  static bool init = false;
  if (!init) {
    TORCH_CHECK(cudaFuncSetAttribute(g11k<MT, RB, EW, NU, S, D, XA>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024) == cudaSuccess, "g11 smem attr");
    init = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((p.N / R) * S, 1, 1);
  cfg.blockDim = dim3((RB + EW) * 32, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = S; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, g11k<MT, RB, EW, NU, S, D, XA>, p) == cudaSuccess, "g11 launch");
}

void g11run(int64_t id, torch::Tensor W, torch::Tensor rp, torch::Tensor ent, torch::Tensor x, torch::Tensor out,
            int64_t m, int64_t N, int64_t K, int64_t base, int64_t ecap, int64_t flags) {
  P11 p;
  p.W = (const uint32_t*)W.data_ptr(); p.rp = (const uint32_t*)rp.data_ptr(); p.ent = (const uint32_t*)ent.data_ptr();
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = (__nv_bfloat16*)out.data_ptr();
  p.m = (int)m; p.N = (int)N; p.K = (int)K; p.base = (uint32_t)base; p.Tb = (uint32_t)base * 0x01010101u; p.ecap = (int)ecap; p.flags = (int)flags;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (id) {
@@CASES@@
    default: TORCH_CHECK(false, "g11: bad config id");
  }
}
"""

# (MT, RB, EW, NU, S, D, XA) per instance; id = index. RB * 16 rows per CTA, NU * 64 K per CTA, S CTAs per cluster.
_G11_CFGS = [
    (1, 9, 1, 16, 8, 4, 4), (1, 9, 2, 16, 8, 8, 4), (1, 6, 2, 32, 4, 8, 4), (1, 3, 1, 32, 4, 8, 4),   # qkv m <= 8: 0-3
    (1, 8, 2, 32, 4, 8, 4), (1, 8, 2, 32, 4, 10, 4), (1, 4, 1, 32, 4, 8, 4),                          # pooler m <= 8: 4-6
    (2, 9, 2, 16, 8, 4, 2), (2, 6, 2, 32, 4, 6, 2), (2, 8, 2, 32, 4, 6, 2), (2, 4, 2, 32, 4, 6, 2),    # m = 16: qkv 7-8, pooler 9-10
]
# cells -> candidate config ids (each candidate geometry R, KL, S has its own escape CSR pack)
_G11_CELLS = {
    ("qkv_gate", 1): [0, 1, 2, 3], ("qkv_gate", 8): [0, 1, 2, 3], ("qkv_gate", 16): [7, 8],
    ("pooler", 1): [4, 5, 6], ("pooler", 8): [4, 5, 6], ("pooler", 16): [9, 10],
}
_G11_SRCBIT = None


def _g11_bits():
    """(group, byte, q) -> (code word, bit) for q in (0, 1, 7): offset bit 0, offset bit 1, sign. Mirrors g11_dec."""
    tab = {}
    for gi in range(8):
        for j in range(4):
            for q in (0, 1, 7):
                ob = 8 * j + q
                if gi < 6:
                    wd, r = gi // 2, (0 if gi % 2 == 0 else 3)
                    src = (wd, (ob + r) % 32)
                elif q in (0, 1):
                    src = (gi - 6, (ob + 5) % 32)
                else:
                    src = (2, (ob - (2 if gi == 6 else 1)) % 32)
                tab[(gi, j, q)] = src
    assert len(set(tab.values())) == 96, "g11 code bit map is not a bijection"
    return tab


def _g11_pack(W, cfg):
    """W bf16 [N, K] -> dict(W words [T, C, 352] int32, rp, ent, base) for one geometry (escape CSR depends on R, KL, S)."""
    global _G11_SRCBIT
    if _G11_SRCBIT is None:
        _G11_SRCBIT = _g11_bits()
    MT, RB, EW, NU, S, D = cfg[:6]
    N, K = W.shape
    T, C, R, KL = N // 16, K // 64, RB * 16, NU * 64
    assert N % R == 0 and K == KL * S
    wi = W.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    h = (wi >> 8) & 0x7F
    hist = torch.bincount(h.flatten(), minlength=128)
    base = int(hist.unfold(0, 4, 1).sum(1).argmax())
    assert base + 3 <= 127
    # physical k within a chunk = 16 t + 4 s + 2 kh + p (lane-contiguous B fragments); value index = 2 (4 s + 2 kh + rh) + p
    v = wi.view(T, 2, 8, C, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(T, C, 32, 32).to(torch.int64)
    hv = (v >> 8) & 0x7F
    off = (hv - base).clamp(0, 3)
    sg = (v >> 15) & 1
    low = (v & 0xFF).view(T, C, 32, 8, 4)
    Lw = low[..., 0] | (low[..., 1] << 8) | (low[..., 2] << 16) | (low[..., 3] << 24)  # [T, C, 32, 8]
    Cw = torch.zeros((T, C, 32, 3), dtype=torch.int64, device=W.device)
    for (gi, j, q), (wd, pos) in _G11_SRCBIT.items():
        vi = 4 * gi + j
        bit = (off[..., vi] & 1) if q == 0 else ((off[..., vi] >> 1) & 1 if q == 1 else sg[..., vi])
        Cw[..., wd] |= bit << pos
    unit = torch.cat([Lw[..., 0:4].reshape(T, C, 128), Lw[..., 4:8].reshape(T, C, 128),
                      Cw[..., 0], Cw[..., 1], Cw[..., 2]], dim=2)  # [T, C, 352]
    words = torch.where(unit >= (1 << 31), unit - (1 << 32), unit).to(torch.int32).contiguous()
    # lossless proof: emulate g11_dec on the packed words, compare with the original bits (escapes: the clamp rule)
    u = unit
    Lr = torch.stack([u[..., 0:128].reshape(T, C, 32, 4), u[..., 128:256].reshape(T, C, 32, 4)], 3).reshape(T, C, 32, 8)
    c0, c1, c2 = u[..., 256:288], u[..., 288:320], u[..., 320:352]
    msk = 0xFFFFFFFF

    def rr(x, s):
        return ((x >> s) | (x << (32 - s))) & msk

    def rl(x, s):
        return ((x << s) | (x >> (32 - s))) & msk
    M = 0x83838383
    X = [c0 & M, rr(c0, 3) & M, c1 & M, rr(c1, 3) & M, c2 & M, rr(c2, 3) & M,
         (rr(c0, 5) & 0x03030303) | (rl(c2, 2) & 0x80808080), (rr(c1, 5) & 0x03030303) | (rl(c2, 1) & 0x80808080)]
    Tb = base * 0x01010101
    H = torch.stack([(x_ + Tb) & msk for x_ in X], 3)  # [T, C, 32, 8]
    dec = []
    for j in range(4):
        dec.append((((H >> (8 * j)) & 0xFF) << 8) | ((Lr >> (8 * j)) & 0xFF))
    dec = torch.stack(dec, 4).reshape(T, C, 32, 32)
    clamp = (v & 0x80FF) | ((hv.clamp(base, base + 3)) << 8)
    assert torch.equal(dec, clamp), "g11 pack: decode emulation mismatch"
    del dec, clamp, H, X, Lr, u, unit, v, hv, off, sg, low, Lw, Cw
    # escapes: CSR over (cta, local row), entries (k_local << 16) | bf16 bits, in k order
    hW = (wi >> 8) & 0x7F
    em = (hW < base) | (hW > base + 3)
    nz = em.nonzero()
    n_, k_ = nz[:, 0], nz[:, 1]
    cta = (n_ // R) * S + k_ // KL
    rowkey = cta * R + n_ % R
    key = rowkey * KL + k_ % KL
    order = torch.argsort(key)
    n_, k_, rowkey, cta = n_[order], k_[order], rowkey[order], cta[order]
    ncta = (N // R) * S
    cnt = torch.bincount(rowkey, minlength=ncta * R).view(ncta, R)
    tot = cnt.sum(1)
    pad = (tot + 3) // 4 * 4
    start = torch.zeros(ncta, dtype=torch.int64, device=W.device)
    start[1:] = torch.cumsum(pad, 0)[:-1]
    rp = torch.zeros((ncta, R + 1), dtype=torch.int64, device=W.device)
    rp[:, 1:] = torch.cumsum(cnt, 1)
    rp += start[:, None]
    loc = torch.arange(n_.numel(), device=W.device) - torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64, device=W.device), tot]), 0)[cta]
    ent = torch.zeros(int(pad.sum()) + 4, dtype=torch.int64, device=W.device)
    ent[start[cta] + loc] = ((k_ % KL) << 16) | wi[n_, k_].to(torch.int64)
    ent = torch.where(ent >= (1 << 31), ent - (1 << 32), ent).to(torch.int32)
    ecap = int(tot.max())
    rp = rp.view(-1)
    nesc = int(nz.shape[0])
    bits = (words.numel() * 32 + ent.numel() * 32 + rp.numel() * 32) / (N * K)
    return {"W": words, "rp": rp.to(torch.int32).contiguous(), "ent": ent.contiguous(), "base": base, "ecap": ecap,
            "esc": nesc / (N * K), "bits": bits, "N": N, "K": K}


_G11_EXT = None


def _g11_build():
    global _G11_EXT
    if _G11_EXT is None:
        cases = "\n".join("    case %d: g11_launch<%d, %d, %d, %d, %d, %d, %d>(p, st); break;" % ((i,) + c) for i, c in enumerate(_G11_CFGS))
        src = _G11_SRC.replace("@@CASES@@", cases)
        from torch.utils.cpp_extension import load_inline as _li
        _G11_EXT = _li(name="g11_" + __import__("hashlib").sha1(src.encode()).hexdigest()[:10], cpp_sources="void g11run(int64_t id, torch::Tensor W, torch::Tensor rp, torch::Tensor ent, torch::Tensor x, torch::Tensor out, int64_t m, int64_t N, int64_t K, int64_t base, int64_t ecap, int64_t flags);",
                       cuda_sources=src, functions=["g11run"],
                       extra_cuda_cflags=["-O3", "-Xptxas=-v"], verbose=bool(__import__("os").environ.get("G11_VERBOSE")))
    return _G11_EXT


def _g11_run(x, pk, cid, m, N, flags=0):
    out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
    _G11_EXT.g11run(cid, pk["W"], pk["rp"], pk["ent"], x, out, m, N, pk["K"], pk["base"], pk["ecap"], flags)
    return out


def _g11_ref(x, W):
    return (x.double() @ W.double().T)


def _g11_valid(pk, cid, W, m, N, gen):
    """random normal rows, outlier channels, one-hot power-of-two probe rows (must be exact); repeat-deterministic"""
    K = W.shape[1]
    for dist in ("normal", "outliers", "probe"):
        x = torch.randn((m, K), generator=gen, device=W.device)
        if dist == "outliers":
            idx = torch.randperm(K, generator=gen, device=W.device)[:K // 100]
            x[:, idx] *= 30
        if dist == "probe":
            x.zero_()
            cols = torch.randint(K, (m,), generator=gen, device=W.device)
            x[torch.arange(m, device=W.device), cols] = 2.0 ** torch.randint(3, 8, (m,), generator=gen, device=W.device).float()
        x = x.to(torch.bfloat16).contiguous()
        out = _g11_run(x, pk, cid, m, N)
        out2 = _g11_run(x, pk, cid, m, N)
        torch.cuda.synchronize()
        if not torch.equal(out, out2):
            return False, "nondeterministic"
        ref = _g11_ref(x, W[:N])
        if dist == "probe":
            if not torch.equal(out.double(), ref.to(torch.bfloat16).double()):
                return False, "probe rows not exact (max %.3e)" % float((out.double() - ref).abs().max())
        else:
            err = float((out.double() - ref).norm() / ref.norm())
            row = float(((out.double() - ref).norm(dim=1) / ref.norm(dim=1)).max())
            if not (err < 4e-3 and row < 8e-3):
                return False, "%s rel %.3e row %.3e" % (dist, err, row)
    return True, "ok"


def _g11_tpair(fa, fb, reps=40):
    """paired, alternated cold-L2 CUDA-graph replays: (median us of fa, median us of fb)"""
    gs = []
    for f in (fa, fb):
        f(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        gs.append(g)
    flush = torch.empty(64 << 20, dtype=torch.float32, device="cuda")
    ts = ([], [])
    for i in range(reps):
        for j in ((0, 1) if i % 2 == 0 else (1, 0)):
            flush.zero_()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); gs[j].replay(); b.record(); b.synchronize()
            ts[j].append(a.elapsed_time(b) * 1e3)
    del gs, flush
    return sorted(ts[0])[reps // 2], sorted(ts[1])[reps // 2]


_G11_LOG = []


def _g11_getpack(packs, layer, W, cid):
    c = _G11_CFGS[cid]
    key = (layer, c[1], c[3], c[4])
    if key not in packs:
        packs[key] = _g11_pack(W, c)
        pk = packs[key]
        _G11_LOG.append("G11 pack %s R %d KL %d S %d base %d esc %.5f bits/weight %.3f ecap %d" % (layer, 16 * c[1], 64 * c[3], c[4], pk["base"], pk["esc"], pk["bits"], pk["ecap"]))
    return packs[key]


def _g11_prepare(weights, st, margin=0.99):
    _g11_build()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    routes, packs = {}, {}
    for (layer, m), cids in _G11_CELLS.items():
        W = weights[layer]
        N = W.shape[0] if not (layer == "unpooler" and m <= 8) else 1024
        x = torch.randn((m, W.shape[1]), generator=gen, device=W.device).to(torch.bfloat16)
        data = {"x": x, "config": {"m": m, "n": N, "k": W.shape[1], "layer": layer}}
        best = None
        for cid in cids:
            pk = _g11_getpack(packs, layer, W, cid)
            ok, msg = _g11_valid(pk, cid, W, m, N, gen)
            if not ok:
                _G11_LOG.append("G11 %s m=%d cfg %d INVALID %s" % (layer, m, cid, msg))
                continue
            tg, tb = _g11_tpair(lambda: _g11_run(x, pk, cid, m, N), lambda: _g11_parent_kernel(data, st))
            _G11_LOG.append("G11 %s m=%d cfg %d %s g11 %.2f us parent %.2f us ratio %.3f" % (layer, m, cid, _G11_CFGS[cid], tg, tb, tg / tb))
            if tg < margin * tb and (best is None or tg < best[0]):
                best = (tg, cid)
        if best is not None:
            routes[(layer, m)] = {"pk": _g11_getpack(packs, layer, W, best[1]), "cid": best[1]}
            _G11_LOG.append("G11 PICK %s m=%d cfg %d" % (layer, m, best[1]))
    used = {id(r["pk"]) for r in routes.values()}
    for k in list(packs):
        if id(packs[k]) not in used:
            del packs[k]
    torch.cuda.empty_cache()
    return {"routes": routes}


_g11_parent_prepare = prepare
_g11_parent_kernel = kernel_fn


def prepare(weights):
    import threading
    th = threading.Thread(target=lambda: _g11_safe_build(), daemon=True)
    th.start()
    st = _g11_parent_prepare(weights)
    th.join()
    try:
        st["g11"] = _g11_prepare(weights, st)
    except Exception as exc:  # noqa: BLE001  (any failure leaves the parent's dispatch)
        _G11_LOG.append("G11 unavailable: %s" % repr(exc)[:300])
        st["g11"] = {"routes": {}}
    print("\n".join(_G11_LOG), flush=True)
    return st


def _g11_safe_build():
    try:
        _g11_build()
    except Exception as exc:  # noqa: BLE001
        _G11_LOG.append("G11 build failed: %s" % repr(exc)[:2000])


def kernel_fn(data, state):
    cfg = data["config"]
    r = state.get("g11", {}).get("routes", {}).get((cfg["layer"], cfg["m"])) if isinstance(state, dict) else None
    if r is not None:
        _ctx_bind(data["x"])
        return _g11_run(data["x"].contiguous(), r["pk"], r["cid"], cfg["m"], cfg["n"])
    return _g11_parent_kernel(data, state)
