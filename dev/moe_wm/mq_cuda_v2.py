"""Marlin-layout NVFP4 MoE GEMM (reads SGLang's repacked Marlin weights/scales in place; no second copy of the experts)."""
import wm_cuda

MQ_KERNELS = r"""
// ---------------------------------------------------------------------------------------------------------------------
// mq: weights/scales exactly as SGLang's gptq_marlin_repack / marlin scale processing leave them.
//   weights: int32 [E, K/16, NTILES*128]; word index ((kt * NTILES + nt) * 128 + lane * 4 + wq): the 8 nibbles of
//            (16 rows n = nt*64 + wq*16 + {g, g+8}) x (k16 step kt) = one mma A fragment, nibble pairs (j, j+4) per register.
//   scales : uint8 [E, K/16, N]; position within a 64-row block is permuted so that lane g finds its 8 scales (rows g/g+8 of
//            the 4 sub-tiles) in the 8 contiguous bytes at offset 8 g.  byte B -> bf16 bits ((B << 4) + 0x6F80) = S * 2^118.
//   global scale: stored as gs * 2^119 in bf16; the accumulator is rescaled by stored * 2^-111 (undoes the 2^-8 of the above).
// ---------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t mq_dq(uint32_t q, int j) {
  const uint32_t t = j == 3 ? q : (q << (4 * (3 - j)));
  const uint32_t m = (t & 0x70007000u) >> 6;
  uint32_t o;
  asm("lop3.b32 %0, %1, %2, %3, 0xea;\n" : "=r"(o) : "r"(t), "r"(0x80008000u), "r"(m));
  return o;
}
__device__ __forceinline__ void mq_ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ uint32_t mq_scale(uint32_t w32, int b) {
  const uint32_t x = __byte_perm(w32, 0u, 0x4040u | (b << 8) | b);   // byte b in both 16-bit lanes
  return x * 16u + 0x6F806F80u;
}

template <int MODE, int NCH, int S, int DBG>
__device__ __forceinline__ void mq_body(const uint4* __restrict__ Wm, const uint8_t* __restrict__ Sm, const float* __restrict__ GS,
                                        const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst, int e,
                                        int ntok, uint8_t* smem_raw, const int bx) {
  constexpr int NTILES = MODE == 0 ? 128 : 64;   // 64-row tiles of the layer
  constexpr int NROWS = NTILES * 64;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int XSLOTS = NCH * 64;      // 16-byte granules of the x tile: NCH*8 token rows x 8
  constexpr int XPITCH = 144;           // bytes per token row (128 + pad) -> conflict-free ldmatrix
  constexpr int XSTAGE = NCH * 8 * XPITCH;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int nt = MODE == 0 ? ((warp < 4) ? (bx * 4 + warp) : (64 + bx * 4 + (warp - 4))) : (bx * 8 + warp);
  uint4* wring = reinterpret_cast<uint4*>(smem_raw);            // [S][8 warps][4 ks][32 lanes]
  uint8_t* xring = reinterpret_cast<uint8_t*>(wring + S * 8 * 4 * 32);   // [S][NCH*8][XPITCH]

  const uint4* wsrc = Wm + (size_t)e * (256 * NTILES * 32) + (size_t)nt * 32 + lane;
  const uint8_t* ssrc = Sm + (size_t)e * ((size_t)256 * NROWS) + (size_t)nt * 64 + 8 * g;

  const bf16* xb = X;
  bool xok = false;
  if (tid < XSLOTS) {
    const int col = tid >> 3, gr = tid & 7;
    if (col < ntok) {
      const int p = __ldcg(lst + col);
      xb = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + gr * 8;
      xok = true;
    }
  }
  float acc[4][NCH][4];
#pragma unroll
  for (int n = 0; n < 4; n++)
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[n][c][r] = 0.f;

  auto issue = [&](int c) {
    if (c < KC) {
#pragma unroll
      for (int ks = 0; ks < 4; ks++) cp16(&wring[(((c % S) * 8 + warp) * 4 + ks) * 32 + lane], wsrc + (size_t)(4 * c + ks) * (NTILES * 32));
      if (tid < XSLOTS) cp16z(xring + (c % S) * XSTAGE + (tid >> 3) * XPITCH + (tid & 7) * 16, xb + c * 64, xok ? 16u : 0u);
    }
    cp_commit();
  };
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
  uint2 scur[4], snxt[4];
#pragma unroll
  for (int ks = 0; ks < 4; ks++) {
    scur[ks] = __ldg(reinterpret_cast<const uint2*>(ssrc + (size_t)ks * NROWS));
    snxt[ks] = __ldg(reinterpret_cast<const uint2*>(ssrc + (size_t)(4 + ks) * NROWS));
  }
#pragma unroll 1
  for (int i = 0; i < KC; i++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(i + S - 1);
    uint32_t xr[NCH][4][2];   // [token group][ks][b0, b1]
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        const uint8_t* ap = xring + (i % S) * XSTAGE + (h * 8 + (lane & 7)) * XPITCH + (hf * 32 + (lane >> 3) * 8) * 2;
        mq_ldsm4(xr[h][2 * hf][0], xr[h][2 * hf][1], xr[h][2 * hf + 1][0], xr[h][2 * hf + 1][1], ap);
      }
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const uint4 q4 = wring[(((i % S) * 8 + warp) * 4 + ks) * 32 + lane];
      const uint32_t qw[4] = {q4.x, q4.y, q4.z, q4.w};
      const uint2 sc = scur[ks];
#pragma unroll
      for (int n = 0; n < 4; n++) {
        // scale bytes of (sub-tile n, row g / g+8): offsets 0,2 | 1,3 | 4,6 | 5,7
        const int bl = (n < 2) ? (n == 0 ? 0 : 1) : (n == 2 ? 4 : 5);
        const uint32_t wl = bl < 4 ? sc.x : sc.y;
        const int bsh = bl & 3;
        const uint32_t s_lo = mq_scale(wl, bsh);
        const uint32_t s_hi = mq_scale(wl, bsh + 2);
        const uint32_t a0 = hmul(mq_dq(qw[n], 0), s_lo);
        const uint32_t a1 = hmul(mq_dq(qw[n], 2), s_hi);
        const uint32_t a2 = hmul(mq_dq(qw[n], 1), s_lo);
        const uint32_t a3 = hmul(mq_dq(qw[n], 3), s_hi);
#pragma unroll
        for (int h = 0; h < NCH; h++) {
          mma16816(acc[n][h], a0, a1, a2, a3, xr[h][ks][0], xr[h][ks][1]);
        }
      }
    }
    // next chunk's scales (two chunks ahead of use)
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      scur[ks] = snxt[ks];
      if (i + 2 < KC) snxt[ks] = __ldg(reinterpret_cast<const uint2*>(ssrc + (size_t)(4 * (i + 2) + ks) * NROWS));
    }
  }
  cp_wait<0>();
  __syncthreads();

  const float gs = GS[e];
  if (MODE == 0) {
    float* ex = reinterpret_cast<float*>(smem_raw);   // [NCH*8][260]
    if (warp >= 4) {
#pragma unroll
      for (int n = 0; n < 4; n++)
#pragma unroll
        for (int c = 0; c < NCH; c++)
#pragma unroll
          for (int r = 0; r < 4; r++) ex[(c * 8 + 2 * t + (r & 1)) * 260 + (warp - 4) * 64 + n * 16 + g + 8 * (r >> 1)] = acc[n][c][r];
    }
    __syncthreads();
    if (warp < 4) {
#pragma unroll
      for (int c = 0; c < NCH; c++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int col = c * 8 + 2 * t + (r & 1);
          if (col >= ntok) continue;
          const int p = __ldcg(lst + col);
#pragma unroll
          for (int n = 0; n < 4; n++) {
            const int fl = warp * 64 + n * 16 + g + 8 * (r >> 1);
            const float gate = bfr(acc[n][c][r] * gs), up = bfr(ex[col * 260 + fl] * gs);
            const float sil = gate / (1.0f + __expf(-gate));
            OUT[(size_t)p * ID + bx * 256 + fl] = __float2bfloat16(sil * up);
          }
        }
    }
  } else {
#pragma unroll
    for (int c = 0; c < NCH; c++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int col = c * 8 + 2 * t + (r & 1);
        if (col >= ntok) continue;
        const int p = __ldcg(lst + col);
#pragma unroll
        for (int n = 0; n < 4; n++) OUT[(size_t)p * HD + nt * 64 + n * 16 + g + 8 * (r >> 1)] = __float2bfloat16(acc[n][c][r] * gs);
      }
  }
}


// K-split variant for small token counts (<= 16 per item): one CTA per 64-row tile (MODE 0: a gate tile plus its up tile), the
// eight warps split K; their fp32 partial tiles are added through shared memory.
template <int MODE, int NCH, int S, int DBG>
__device__ __forceinline__ void mq_kbody(const uint4* __restrict__ Wm, const uint8_t* __restrict__ Sm, const float* __restrict__ GS,
                                         const bf16* __restrict__ X, bf16* __restrict__ OUT, const int* __restrict__ lst, int e,
                                         int ntok, uint8_t* smem_raw, const int bx) {
  constexpr int NTILES = MODE == 0 ? 128 : 64;
  constexpr int NROWS = NTILES * 64;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int NG = MODE == 0 ? 2 : 1;           // n-tiles per CTA (gate, up)
  constexpr int XPITCH = 144;
  constexpr int XSTAGE = NCH * 8 * XPITCH;
  constexpr int CPW = KC / 8;                     // k64 chunks per warp
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  uint4* wring = reinterpret_cast<uint4*>(smem_raw) + (size_t)warp * (S * NG * 4 * 32);           // per warp [S][NG][4 ks][32]
  uint8_t* xring = smem_raw + (size_t)8 * S * NG * 4 * 32 * 16 + (size_t)warp * (S * XSTAGE);      // per warp [S][NCH*8][XPITCH]
  const int c0 = warp * CPW;

  const uint4* wsrc[NG];
  const uint8_t* ssrc[NG];
#pragma unroll
  for (int q = 0; q < NG; q++) {
    const int nt = bx + 64 * q;
    wsrc[q] = Wm + (size_t)e * (256 * NTILES * 32) + (size_t)nt * 32 + lane;
    ssrc[q] = Sm + (size_t)e * ((size_t)256 * NROWS) + (size_t)nt * 64 + 8 * g;
  }
  // x staging: this lane copies granules lane, lane + 32, ... of the warp's NCH*64 granules
  const bf16* xb[2 * NCH];
  bool xok[2 * NCH];
#pragma unroll
  for (int u = 0; u < 2 * NCH; u++) {
    const int gi = lane + 32 * u;               // granule id: token = gi >> 3, gr = gi & 7
    const int col = gi >> 3;
    xok[u] = col < ntok;
    xb[u] = X;
    if (xok[u]) {
      const int p = __ldcg(lst + col);
      xb[u] = X + (size_t)(MODE == 0 ? (p >> 3) : p) * XW + (gi & 7) * 8;
    }
  }
  float acc[NG][4][NCH][4];
#pragma unroll
  for (int q = 0; q < NG; q++)
#pragma unroll
    for (int n = 0; n < 4; n++)
#pragma unroll
      for (int c = 0; c < NCH; c++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc[q][n][c][r] = 0.f;

  auto issue = [&](int c) {
    if (c < CPW) {
#pragma unroll
      for (int q = 0; q < NG; q++)
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
          cp16(&wring[(((c % S) * NG + q) * 4 + ks) * 32 + lane], wsrc[q] + (size_t)(4 * (c0 + c) + ks) * (NTILES * 32));
#pragma unroll
      for (int u = 0; u < 2 * NCH; u++) {
        const int gi = lane + 32 * u;
        cp16z(xring + (c % S) * XSTAGE + (gi >> 3) * XPITCH + (gi & 7) * 16, xb[u] + (c0 + c) * 64, xok[u] ? 16u : 0u);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int c = 0; c < S - 1; c++) issue(c);
  uint2 scur[NG][4], snxt[NG][4];
#pragma unroll
  for (int q = 0; q < NG; q++)
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      scur[q][ks] = __ldg(reinterpret_cast<const uint2*>(ssrc[q] + (size_t)(4 * c0 + ks) * NROWS));
      snxt[q][ks] = __ldg(reinterpret_cast<const uint2*>(ssrc[q] + (size_t)(4 * (c0 + 1) + ks) * NROWS));
    }
#pragma unroll 1
  for (int i = 0; i < CPW; i++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(i + S - 1);
    uint32_t xr[NCH][4][2];
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        const uint8_t* ap = xring + (i % S) * XSTAGE + (h * 8 + (lane & 7)) * XPITCH + (hf * 32 + (lane >> 3) * 8) * 2;
        mq_ldsm4(xr[h][2 * hf][0], xr[h][2 * hf][1], xr[h][2 * hf + 1][0], xr[h][2 * hf + 1][1], ap);
      }
#pragma unroll
    for (int q = 0; q < NG; q++)
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        const uint4 q4 = wring[(((i % S) * NG + q) * 4 + ks) * 32 + lane];
        const uint32_t qw[4] = {q4.x, q4.y, q4.z, q4.w};
        const uint2 sc = scur[q][ks];
#pragma unroll
        for (int n = 0; n < 4; n++) {
          const int bl = (n < 2) ? (n == 0 ? 0 : 1) : (n == 2 ? 4 : 5);
          const uint32_t wl = bl < 4 ? sc.x : sc.y;
          const int bsh = bl & 3;
          const uint32_t s_lo = mq_scale(wl, bsh);
          const uint32_t s_hi = mq_scale(wl, bsh + 2);
          const uint32_t a0 = hmul(mq_dq(qw[n], 0), s_lo);
          const uint32_t a1 = hmul(mq_dq(qw[n], 2), s_hi);
          const uint32_t a2 = hmul(mq_dq(qw[n], 1), s_lo);
          const uint32_t a3 = hmul(mq_dq(qw[n], 3), s_hi);
#pragma unroll
          for (int h = 0; h < NCH; h++) mma16816(acc[q][n][h], a0, a1, a2, a3, xr[h][ks][0], xr[h][ks][1]);
        }
      }
#pragma unroll
    for (int q = 0; q < NG; q++)
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        scur[q][ks] = snxt[q][ks];
        if (i + 2 < CPW) snxt[q][ks] = __ldg(reinterpret_cast<const uint2*>(ssrc[q] + (size_t)(4 * (c0 + i + 2) + ks) * NROWS));
      }
  }
  cp_wait<0>();
  __syncthreads();

  // cross-warp K reduction: red[w][q][n][h][r][lane]
  float* red = reinterpret_cast<float*>(smem_raw);
#pragma unroll
  for (int q = 0; q < NG; q++)
#pragma unroll
    for (int n = 0; n < 4; n++)
#pragma unroll
      for (int h = 0; h < NCH; h++)
#pragma unroll
        for (int r = 0; r < 4; r++) red[((((warp * NG + q) * 4 + n) * NCH + h) * 4 + r) * 32 + lane] = acc[q][n][h][r];
  __syncthreads();
  const float gs = GS[e];
  for (int item = tid; item < 4 * NCH * 4 * 32; item += 256) {
    const int li = item & 31, r = (item >> 5) & 3, h = (item >> 7) % NCH, n = (item >> 7) / NCH;
    const int row = (li >> 2) + 8 * (r >> 1);
    const int col = h * 8 + 2 * (li & 3) + (r & 1);
    if (col >= ntok) continue;
    const int p = __ldcg(lst + col);
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int w = 0; w < 8; w++) {
      s0 += red[((((w * NG + 0) * 4 + n) * NCH + h) * 4 + r) * 32 + li];
      if (MODE == 0) s1 += red[((((w * NG + 1) * 4 + n) * NCH + h) * 4 + r) * 32 + li];
    }
    if (MODE == 0) {
      const float gate = bfr(s0 * gs), up = bfr(s1 * gs);
      const float sil = gate / (1.0f + __expf(-gate));
      OUT[(size_t)p * ID + bx * 64 + n * 16 + row] = __float2bfloat16(sil * up);
    } else {
      OUT[(size_t)p * HD + bx * 64 + n * 16 + row] = __float2bfloat16(s0 * gs);
    }
  }
}

template <int MODE, int NMAX, int S, int DBG>
__global__ void __launch_bounds__(256, 2)
mq_gemm(const uint4* __restrict__ Wm, const uint8_t* __restrict__ Sm, const float* __restrict__ GS, const bf16* __restrict__ X,
        bf16* __restrict__ OUT, const int* __restrict__ lst, const int* __restrict__ act, int lmax) {
  extern __shared__ __align__(16) uint8_t smem_mq[];
  pdl_wait();
  pdl_trigger();
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);
  if (item.z == 0) return;
  const int e = item.x, ntok = item.z;
  const int* l = lst + e * lmax + item.y;
  if (ntok <= 8) mq_kbody<MODE, 1, S, DBG>(Wm, Sm, GS, X, OUT, l, e, ntok, smem_mq, blockIdx.x);
  else if (NMAX >= 2 && ntok <= 16) mq_kbody<MODE, (NMAX >= 2 ? 2 : 1), S, DBG>(Wm, Sm, GS, X, OUT, l, e, ntok, smem_mq, blockIdx.x);
  else if (NMAX >= 4) {
    if (blockIdx.x >= (MODE == 0 ? 16 : 8)) return;   // the row-split path uses fewer, wider CTAs
    mq_body<MODE, (NMAX >= 4 ? 4 : 1), S, DBG>(Wm, Sm, GS, X, OUT, l, e, ntok, smem_mq, blockIdx.x);
  }
}

template <int MODE, int NMAX, int S, int DBG>
static void launch_mq(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst, const int* act,
                      int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid(64, (unsigned)maxslots);
  constexpr int NG_ = MODE == 0 ? 2 : 1;
  const int NK = NMAX >= 2 ? 2 : 1;
  int dyn = S * (8 * 4 * 32 * 16 + NMAX * 8 * 144);                                  // row-split path
  const int dynk = 8 * S * (NG_ * 4 * 32 * 16 + NK * 8 * 144);                       // K-split path
  if (dynk > dyn) dyn = dynk;
  const int ex = NMAX * 8 * 260 * 4;
  if (ex > dyn) dyn = ex;
  const int redk = 8 * NG_ * 4 * NK * 4 * 32 * 4;
  if (redk > dyn) dyn = redk;
  cudaFuncSetAttribute(mq_gemm<MODE, NMAX, S, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(mq_gemm<MODE, NMAX, S, DBG>, grid, dim3(256), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const uint8_t*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);
}

extern "C" int mq_gemm_c(int mode, int nmax, int s, const void* W, const void* Sc, const float* GS, const void* X, void* OUT,
                         const int* lst, const int* act, int maxslots, int lmax, cudaStream_t st) {
  const int pdl = g_pdl;
#define MQ_D(M, N, S_) if (mode == M && nmax == N && s == S_) { launch_mq<M, N, S_, 0>(W, Sc, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  MQ_D(0, 4, 4) MQ_D(1, 4, 4) MQ_D(0, 2, 4) MQ_D(1, 2, 4) MQ_D(0, 1, 4) MQ_D(1, 1, 4)
  MQ_D(0, 4, 2) MQ_D(1, 4, 2) MQ_D(0, 2, 2) MQ_D(1, 2, 2) MQ_D(0, 1, 2) MQ_D(1, 1, 2)
  MQ_D(0, 4, 3) MQ_D(1, 4, 3) MQ_D(0, 2, 3) MQ_D(1, 2, 3) MQ_D(0, 1, 3) MQ_D(1, 1, 3)
  MQ_D(0, 4, 3) MQ_D(1, 4, 3) MQ_D(0, 4, 6) MQ_D(1, 4, 6) MQ_D(0, 2, 6) MQ_D(1, 2, 6) MQ_D(0, 1, 6) MQ_D(1, 1, 6)
#undef MQ_D
  return 1;
}
"""

MQ_GLUE = r"""
extern "C" int mq_gemm_c(int, int, int, const void*, const void*, const float*, const void*, void*, const int*, const int*, int, int, cudaStream_t);
void mq_gemm_run(int64_t mode, int64_t nmax, int64_t s, torch::Tensor W, torch::Tensor Sc, torch::Tensor GS, torch::Tensor X, torch::Tensor OUT,
                 torch::Tensor lst, torch::Tensor act, int64_t maxslots, int64_t lmax) {
  int r = mq_gemm_c((int)mode, (int)nmax, (int)s, W.data_ptr(), Sc.data_ptr(), GS.data_ptr<float>(), X.data_ptr(), OUT.data_ptr(), lst.data_ptr<int>(),
                    act.data_ptr<int>(), (int)maxslots, (int)lmax, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported mq config");
}
"""

MQ_CPP = r"""
void mq_gemm_run(int64_t mode, int64_t nmax, int64_t s, torch::Tensor W, torch::Tensor Sc, torch::Tensor GS, torch::Tensor X, torch::Tensor OUT,
                 torch::Tensor lst, torch::Tensor act, int64_t maxslots, int64_t lmax);
"""

# CUDA source: wm kernels + mq kernels (the mq kernels need launch_k, KC, cp16, hmul, ... from wm_cuda)
CUDA_SRC = wm_cuda.CUDA_SRC + MQ_KERNELS
TORCH_GLUE = wm_cuda.TORCH_GLUE + MQ_GLUE
CPP_SRC = wm_cuda.CPP_SRC + MQ_CPP


def marlin_pack(w, sf):
    """Pure-torch replica of SGLang's NVFP4 Marlin MoE repack (gptq_marlin_repack + scale processing).
    w: uint8 [E, N, K/2] (low nibble = even k), sf: float8_e4m3fn [E, N, K/16].
    Returns (int32 [E, K/16, N/64*128], uint8 [E, K/16, N])."""
    import torch

    E, N, K2 = w.shape
    K = K2 * 2
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    ws, ss = [], []
    for e in range(E):
        nib = torch.stack((w[e] & 15, w[e] >> 4), dim=-1).reshape(N, K)
        v = nib.view(N // 64, 4, 2, 8, K // 16, 2, 4, 2)      # [nt, wq, hn, g, kt, kh, tq, kl]
        word = torch.zeros((N // 64, 4, 8, K // 16, 4), dtype=torch.int64, device=w.device)   # [nt, wq, g, kt, tq]
        for kl in range(2):
            for hn in range(2):
                for kh in range(2):
                    word |= v[:, :, hn, :, :, kh, :, kl].to(torch.int64) << (4 * (kl * 4 + hn * 2 + kh))
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        ws.append(word.permute(3, 0, 2, 4, 1).contiguous().view(K // 16, N // 64 * 128))       # [kt, nt, g, tq, wq]
        s = sf[e].to(torch.bfloat16).T.contiguous()                  # [K/16, N]
        s = s.reshape((-1, 64))[:, scale_perm].reshape((-1, N)).contiguous()
        s = s.to(torch.half)
        s = s.view(-1, 4)[:, [0, 2, 1, 3]].view(s.size(0), -1)
        s = (s * (2 ** 7)).view(torch.int16) << 1
        s = s.view(torch.float8_e4m3fn)[:, 1::2].contiguous()
        ss.append(s.view(torch.uint8))
    return torch.stack(ws), torch.stack(ss)
