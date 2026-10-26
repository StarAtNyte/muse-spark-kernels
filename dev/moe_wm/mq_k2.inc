// K-split small-token kernel with register-resident weight prefetch (no shared-memory weight ring): 4 warps split K,
// each warp double-buffers the uint4 fragments of its next k64 chunk in registers; x tiles come through a small cp.async ring.
template <int MODE, int NCH, int SX, int DBG>
__global__ void __launch_bounds__(128, 3)
mq_gemm_k2(const uint4* __restrict__ Wm, const uint8_t* __restrict__ Sm, const float* __restrict__ GS, const bf16* __restrict__ X,
           bf16* __restrict__ OUT, const int* __restrict__ lst, const int* __restrict__ act, int lmax) {
  extern __shared__ __align__(16) uint8_t smem_k2[];
  constexpr int NTILES = MODE == 0 ? 128 : 64;
  constexpr int NROWS = NTILES * 64;
  constexpr int XW = MODE == 0 ? HD : ID;
  constexpr int NG = MODE == 0 ? 2 : 1;
  constexpr int XPITCH = 144;
  constexpr int XSTAGE = NCH * 8 * XPITCH;
  constexpr int CPW = KC / 4;                     // 16 k64 chunks per warp
  pdl_wait();
  pdl_trigger();
  const int4 item = __ldcg(reinterpret_cast<const int4*>(act) + blockIdx.y);
  if (item.z == 0) return;
  const int e = item.x, ntok = item.z;
  const int* lstp = lst + e * lmax + item.y;
  const int bx = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2;
  uint8_t* xring = smem_k2 + (size_t)warp * (SX * XSTAGE);
  const int c0 = warp * CPW;

  const uint4* wsrc[NG];
  const uint8_t* ssrc[NG];
#pragma unroll
  for (int q = 0; q < NG; q++) {
    const int nt = bx + 64 * q;
    wsrc[q] = Wm + (size_t)e * (256 * NTILES * 32) + (size_t)nt * 32 + lane;
    ssrc[q] = Sm + (size_t)e * ((size_t)256 * NROWS) + (size_t)nt * 64 + 8 * g;
  }
  const bf16* xb[2 * NCH];
  bool xok[2 * NCH];
#pragma unroll
  for (int u = 0; u < 2 * NCH; u++) {
    const int gi = lane + 32 * u;
    const int col = gi >> 3;
    xok[u] = col < ntok;
    xb[u] = X;
    if (xok[u]) {
      const int p = __ldcg(lstp + col);
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

  auto xissue = [&](int c) {
    if (c < CPW) {
#pragma unroll
      for (int u = 0; u < 2 * NCH; u++) {
        const int gi = lane + 32 * u;
        cp16z(xring + (c % SX) * XSTAGE + (gi >> 3) * XPITCH + (gi & 7) * 16, xb[u] + (c0 + c) * 64, xok[u] ? 16u : 0u);
      }
    }
    cp_commit();
  };
  uint4 wv[2][NG][4];
  uint2 sv[2][NG][4];
  auto wload = [&](int c, int b) {
#pragma unroll
    for (int q = 0; q < NG; q++)
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        wv[b][q][ks] = __ldg(wsrc[q] + (size_t)(4 * (c0 + c) + ks) * (NTILES * 32));
        sv[b][q][ks] = __ldg(reinterpret_cast<const uint2*>(ssrc[q] + (size_t)(4 * (c0 + c) + ks) * NROWS));
      }
  };
  wload(0, 0);
#pragma unroll
  for (int c = 0; c < SX - 1; c++) xissue(c);
  auto compute = [&](int i, int b) {
    uint32_t xr[NCH][4][2];
#pragma unroll
    for (int h = 0; h < NCH; h++)
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        const uint8_t* ap = xring + (i % SX) * XSTAGE + (h * 8 + (lane & 7)) * XPITCH + (hf * 32 + (lane >> 3) * 8) * 2;
        mq_ldsm4(xr[h][2 * hf][0], xr[h][2 * hf][1], xr[h][2 * hf + 1][0], xr[h][2 * hf + 1][1], ap);
      }
#pragma unroll
    for (int q = 0; q < NG; q++)
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        const uint4 q4 = wv[b][q][ks];
        const uint32_t qw[4] = {q4.x, q4.y, q4.z, q4.w};
        const uint2 sc = sv[b][q][ks];
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
  };
#pragma unroll 1
  for (int i = 0; i < CPW; i += 2) {
    cp_wait<SX - 2>();
    __syncwarp();
    xissue(i + SX - 1);
    wload(i + 1, 1);
    compute(i, 0);
    cp_wait<SX - 2>();
    __syncwarp();
    xissue(i + SX);
    if (i + 2 < CPW) wload(i + 2, 0);
    compute(i + 1, 1);
  }
  cp_wait<0>();
  __syncthreads();

  float* red = reinterpret_cast<float*>(smem_k2);
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
  for (int it = tid; it < 4 * NCH * 4 * 32; it += 128) {
    const int li = it & 31, r = (it >> 5) & 3, h = (it >> 7) % NCH, n = (it >> 7) / NCH;
    const int row = (li >> 2) + 8 * (r >> 1);
    const int col = h * 8 + 2 * (li & 3) + (r & 1);
    if (col >= ntok) continue;
    const int p = __ldcg(lstp + col);
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int w = 0; w < 4; w++) {
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

template <int MODE, int NCH, int SX, int DBG>
static void launch_k2(const void* W, const void* Sc, const float* GS, const void* X, void* OUT, const int* lst, const int* act,
                      int maxslots, int lmax, int pdl, cudaStream_t st) {
  dim3 grid(64, (unsigned)maxslots);
  constexpr int NG_ = MODE == 0 ? 2 : 1;
  int dyn = 4 * SX * (NCH * 8 * 144);
  const int redk = 4 * NG_ * 4 * NCH * 4 * 32 * 4;
  if (redk > dyn) dyn = redk;
  cudaFuncSetAttribute(mq_gemm_k2<MODE, NCH, SX, DBG>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  launch_k(mq_gemm_k2<MODE, NCH, SX, DBG>, grid, dim3(128), (size_t)dyn, st, pdl != 0, reinterpret_cast<const uint4*>(W),
           reinterpret_cast<const uint8_t*>(Sc), GS, reinterpret_cast<const bf16*>(X), reinterpret_cast<bf16*>(OUT), lst, act, lmax);
}

extern "C" int mq_gemm_k2_c(int mode, int nch, int sx, const void* W, const void* Sc, const float* GS, const void* X, void* OUT,
                            const int* lst, const int* act, int maxslots, int lmax, cudaStream_t st) {
  const int pdl = g_pdl;
#define MK_D(M, N, S_) if (mode == M && nch == N && sx == S_) { launch_k2<M, N, S_, 0>(W, Sc, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  MK_D(0, 1, 2) MK_D(1, 1, 2) MK_D(0, 2, 2) MK_D(1, 2, 2) MK_D(0, 1, 3) MK_D(1, 1, 3) MK_D(0, 2, 3) MK_D(1, 2, 3) MK_D(0, 1, 4) MK_D(1, 1, 4) MK_D(0, 2, 4) MK_D(1, 2, 4)
#undef MK_D
  return 1;
}

