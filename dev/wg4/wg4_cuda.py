CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define HD 64
#define WT 64              // keys per tile

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)), "l"(gsrc));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ uint32_t cvt_fp8x2(uint32_t x) {   // low two e4m3 bytes -> f16x2
  uint32_t r;
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %1;\ncvt.rn.f16x2.e4m3x2 %0, a;\n}" : "=r"(r) : "r"(x));
  return r;
}

__device__ __forceinline__ void wg_qk(float* d, uint64_t da, uint64_t db, int accum) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "l"(da), "l"(db), "r"(accum));
#endif
}

__device__ __forceinline__ void wg_pv(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
}

__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void wg_pv72(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
}
__device__ __forceinline__ void wg_pvt(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
}


__device__ __forceinline__ void wg_commit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_wait0() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_qk128_nw(float* d, uint64_t da, uint64_t db, int accum);
__device__ __forceinline__ void wg_qk128(float* d, uint64_t da, uint64_t db, int accum) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
               : "l"(da), "l"(db), "r"(accum));
#endif
}
__device__ __forceinline__ uint64_t mkdesc(const void* p, uint32_t lbo, uint32_t sbo) {
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
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// core-matrix layout of a [rows][64 byte] fp8 tile: 8 rows x 16 B cores, K cores 128 B apart, row groups 512 B apart
__device__ __forceinline__ int core_off(int r, int chunk) { return (r >> 3) * 512 + chunk * 128 + (r & 7) * 16; }
// vT: [d][key] f16, cores of 8 d-rows x 8 keys (16 B per row): key cores 128 B apart, d groups 1024 B apart
__device__ __forceinline__ int vt_off(int d, int key) { return (d >> 3) * 1024 + (key >> 3) * 128 + (d & 7) * 16 + (key & 7) * 2; }

__device__ __forceinline__ int vmn_off(int key, int d) { return (key >> 3) * 1152 + (d >> 3) * 128 + (key & 7) * 16 + (d & 7) * 2; }
#define WS 3   // raw stages
#define TILEB 8192   // bytes of a raw K or V tile
__device__ __forceinline__ void wg_qk128_nw(float* d, uint64_t da, uint64_t db, int accum) { wg_qk128(d, da, db, accum); }

__global__ void __launch_bounds__(256, 2)
wg4_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x, wgi = tid >> 7, tw = tid & 127, lane = tid & 31, warp = (tid >> 5) & 3;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = wgi, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * 2 + kh) * splits + split) * 64;
  const int n = end > begin ? end - begin : 0;
  const int nst = (n + WT - 1) / WT;

  uint8_t* qs = smem + wgi * 4096;              // 2 x 4 KB, core layout
  uint8_t* vt = smem + 8192 + wgi * 9216;      // 2 x 18 KB f16 V (MN-major), a ninth column group of ones gives the row sums
  uint8_t* stages = smem + 8192 + 18432;                // WS x (K core 2 x 8 KB + V raw 128 keys x 128 B)
  int* rows = reinterpret_cast<int*>(stages + WS * 16384);

  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 256) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  for (int c = tw; c < 64 * 4; c += 128) {
    const int r = c >> 2, ch = c & 3;
    const int token = r >> 3, qh = kh * 8 + (r & 7);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (token < L) v = *reinterpret_cast<const uint4*>(Q + ((long)(b * L + token) * HQ + qh) * HD + ch * 16);
    *reinterpret_cast<uint4*>(qs + core_off(r, ch)) = v;
  }
  {
    const uint32_t ad = (uint32_t)__cvta_generic_to_shared(vt + (tw >> 3) * 1152 + 1024 + (tw & 7) * 16);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ad), "r"(0x3C003C00u), "r"(0x3C003C00u), "r"(0x3C003C00u), "r"(0x3C003C00u));
  }
  proxy_fence();
  __syncthreads();

  auto issue = [&](int s) {
    if (s < nst && !(dbg & 8)) {
      uint8_t* kst = stages + (s % WS) * 16384;
      uint8_t* vst = kst + 8192;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int c = tid + 256 * i;            // 1024 16-byte chunks per tile: 64 keys x (8 K + 8 V)
        const int key = c >> 4, chunk16 = c & 15, cc = chunk16 & 7;
        const int li = s * WT + key;
        const long rr = li < n ? rows[li] : 0;
        const long goff = rr * 128 + cc * 16;
        if (chunk16 < 8) cp16(kst + (cc >> 2) * 4096 + core_off(key, cc & 3), KC + goff);
        else cp16(vst + key * 128 + cc * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < WS - 1; s++) issue(s);

  float o[36];
#pragma unroll
  for (int i = 0; i < 36; i++) o[i] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY};

#pragma unroll 1
  for (int s = 0; s < nst; s++) {
    cp_wait<WS - 2>();
    proxy_fence();
    __syncthreads();
    issue(s + WS - 1);
    const uint8_t* kst = stages + (s % WS) * 16384 + wgi * 4096;
    const uint8_t* vst = stages + (s % WS) * 16384 + 8192;

    float sacc[32];
#pragma unroll
    for (int i = 0; i < 32; i++) sacc[i] = 0.f;
    wg_fence();
    if (!(dbg & 4)) {
      wg_qk(sacc, mkdesc(qs, 128, 512), mkdesc(kst, 128, 512), 0);
      wg_qk(sacc, mkdesc(qs + 256, 128, 512), mkdesc(kst + 256, 128, 512), 1);
    }
    wg_commit();
    // V raw [key][d] fp8 -> MN-major f16 core layout (thread = key x half of d); overlaps with the QK wgmma
    {
      const int key = tw >> 1, dh = tw & 1;
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        const uint4 u = *reinterpret_cast<const uint4*>(vst + key * 128 + wgi * 64 + dh * 32 + h2 * 16);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int q4 = 0; q4 < 2; q4++) {
          uint4 o4;
          o4.x = cvt_fp8x2(w[2 * q4]); o4.y = cvt_fp8x2(w[2 * q4] >> 16);
          o4.z = cvt_fp8x2(w[2 * q4 + 1]); o4.w = cvt_fp8x2(w[2 * q4 + 1] >> 16);
          *reinterpret_cast<uint4*>(vt + vmn_off(key, dh * 32 + h2 * 16 + 8 * q4)) = o4;
        }
      }
    }
    proxy_fence();
    wg_wait0();
#pragma unroll
    for (int i = 0; i < 32; i++) asm volatile("" : "+f"(sacc[i])::"memory");
    asm volatile("bar.sync %0, 128;\n" ::"r"(1 + wgi) : "memory");

    const int tb = begin + s * WT;
    const bool full_vis = (tb + WT <= end) && (L == 1 || tb + WT - 1 <= prefix) && (!window || L == 1 || tb > prefix + L - 1 - window);
    if (!full_vis) {
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int row = warp * 16 + g + 8 * (r >> 1);
          const int token = row >> 3;
          const int col = tb + j * 8 + 2 * t + (r & 1);
          const int pos = prefix + token;
          const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
          if (!vis) sacc[4 * j + r] = -INFINITY;
        }
    }
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sacc[4 * j + r]);
    float corr[2], safe[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      mx[h] *= sc2;
      const float newm = fmaxf(m_run[h], mx[h]);
      safe[h] = newm == -INFINITY ? 0.f : newm;
      corr[h] = ex2f(m_run[h] - safe[h]);
      m_run[h] = newm;
    }
    uint32_t pa[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float p[4];
#pragma unroll
      for (int r = 0; r < 4; r++) p[r] = ex2f(fmaf(sacc[4 * j + r], sc2, -safe[r >> 1]));
      pa[j][0] = pack_h2(p[0], p[1]);
      pa[j][1] = pack_h2(p[2], p[3]);
    }
    if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
      for (int j = 0; j < 9; j++) {
        o[4 * j] *= corr[0]; o[4 * j + 1] *= corr[0];
        o[4 * j + 2] *= corr[1]; o[4 * j + 3] *= corr[1];
      }
    }
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
      if (!(dbg & 2)) wg_pv72(o, pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1], mkdesc(vt + kk * 2304, 1152, 128));
    wg_commit_wait();
#pragma unroll
    for (int i = 0; i < 36; i++) asm volatile("" : "+f"(o[i])::"memory");
  }
  cp_wait<0>();
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = warp * 16 + g + 8 * h;
    const float ls = o[32 + 2 * h];
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      PART[(base_row + row) * HD + j * 8 + 2 * t] = o[4 * j + 2 * h] * inv * value_scale;
      PART[(base_row + row) * HD + j * 8 + 2 * t + 1] = o[4 * j + 2 * h + 1] * inv * value_scale;
    }
    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
#endif
}

extern "C" int wg4_run(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                       const float* VS, float* PART, float* LSE, int B, int L, int HQ, int table_n, int page,
                       int window, int splits, int chunk, float sm_scale, int scaled, int dbg, cudaStream_t st) {
  const size_t rmax = ((chunk + 255) / 256) * 256;
  const size_t dyn = 1024 + 8192 + 18432 + (size_t)WS * 16384 + rmax * 4;
  cudaFuncSetAttribute(wg4_partials, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  wg4_partials<<<grid, 256, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                       reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ,
                                       table_n, page, window, splits, chunk, sm_scale, scaled, dbg);
  return 0;
}
"""
