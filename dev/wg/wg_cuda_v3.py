CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define HD 64
#define WT 64               // keys per tile
#define WRAW (64 * 64)      // bytes of a raw K or V tile

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

__device__ __forceinline__ void wg_pvt(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
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

__device__ __forceinline__ int vmn_off(int key, int d) { return (key >> 3) * 1024 + (d >> 3) * 128 + (key & 7) * 16 + (d & 7) * 2; }
#define WS 3   // raw stages

__global__ void __launch_bounds__(128)
wg_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
            const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
            const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
            int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * HKV + kh) * splits + split) * 64;
  const int n = end > begin ? end - begin : 0;
  const int nst = (n + WT - 1) / WT;

  uint8_t* qs = smem;                           // 4 KB, core layout
  uint8_t* vt = qs + 4096;                      // 8 KB
  uint8_t* stages = vt + 8192;                  // WS x (K core-layout 4 KB + V raw 4 KB)
  int* rows = reinterpret_cast<int*>(stages + WS * 8192);

  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  // Q tile: row r = token (r >> 3), head (r & 7)
  for (int c = tid; c < 64 * 4; c += 128) {
    const int r = c >> 2, ch = c & 3;
    const int token = r >> 3, qh = kh * 8 + (r & 7);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (token < L) v = *reinterpret_cast<const uint4*>(Q + ((long)(b * L + token) * HQ + qh) * HD + ch * 16);
    *reinterpret_cast<uint4*>(qs + core_off(r, ch)) = v;
  }
  __syncthreads();

  auto issue = [&](int s) {
    if (s < nst && !(dbg & 8)) {
      uint8_t* kst = stages + (s % WS) * 8192;
      uint8_t* vst = kst + 4096;
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int c = tid + 128 * i;            // 256 chunks per tile
        const int key = c >> 2, ch = c & 3;
        const int li = s * WT + key;
        const long rr = li < n ? rows[li] : 0;
        const long goff = (rr * HKV + kh) * HD + ch * 16;
        cp16(kst + core_off(key, ch), KC + goff);
        cp16(vst + key * 64 + ch * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < WS - 1; s++) issue(s);

  float o[32];
#pragma unroll
  for (int i = 0; i < 32; i++) o[i] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

#pragma unroll 1
  for (int s = 0; s < nst; s++) {
    cp_wait<WS - 2>();
    proxy_fence();
    __syncthreads();
    issue(s + WS - 1);
    const uint8_t* kst = stages + (s % WS) * 8192;
    const uint8_t* vst = kst + 4096;
    // V raw [key][d] fp8 -> vT [d][key] f16 (thread: key = tid >> 1, d half = tid & 1)
    if (!(dbg & 1) && (dbg & 16)) {
      const int key = tid >> 1, dh = (tid & 1) * 32;
      const uint4 u0 = *reinterpret_cast<const uint4*>(vst + key * 64 + dh);
      const uint4 u1 = *reinterpret_cast<const uint4*>(vst + key * 64 + dh + 16);
      const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
      for (int q4 = 0; q4 < 4; q4++) {
        uint4 o4;
        o4.x = cvt_fp8x2(w[2 * q4]); o4.y = cvt_fp8x2(w[2 * q4] >> 16);
        o4.z = cvt_fp8x2(w[2 * q4 + 1]); o4.w = cvt_fp8x2(w[2 * q4 + 1] >> 16);
        *reinterpret_cast<uint4*>(vt + vmn_off(key, dh + 8 * q4)) = o4;
      }
    } else if (!(dbg & 1)) {
      const int key = tid >> 1, dh = (tid & 1) * 32;
      const uint4 u0 = *reinterpret_cast<const uint4*>(vst + key * 64 + dh);
      const uint4 u1 = *reinterpret_cast<const uint4*>(vst + key * 64 + dh + 16);
      const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const uint32_t lo = cvt_fp8x2(w[i]), hi = cvt_fp8x2(w[i] >> 16);
        const int d = dh + 4 * i;
        *reinterpret_cast<uint16_t*>(vt + vt_off(d, key)) = (uint16_t)(lo & 0xFFFF);
        *reinterpret_cast<uint16_t*>(vt + vt_off(d + 1, key)) = (uint16_t)(lo >> 16);
        *reinterpret_cast<uint16_t*>(vt + vt_off(d + 2, key)) = (uint16_t)(hi & 0xFFFF);
        *reinterpret_cast<uint16_t*>(vt + vt_off(d + 3, key)) = (uint16_t)(hi >> 16);
      }
    }
    proxy_fence();
    __syncthreads();

    // ---- S = Q K^T : 64 x 64, two k32 steps ----
    float sacc[32];
#pragma unroll
    for (int i = 0; i < 32; i++) sacc[i] = 0.f;
    wg_fence();
    if (!(dbg & 4)) {
    wg_qk(sacc, mkdesc(qs, 128, 512), mkdesc(kst, 128, 512), 0);
    wg_qk(sacc, mkdesc(qs + 256, 128, 512), mkdesc(kst + 256, 128, 512), 1);
    }
    wg_commit_wait();

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
    float corr[2], newm[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      mx[h] *= sc2;
      newm[h] = fmaxf(m_run[h], mx[h]);
      const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
      corr[h] = exp2f(m_run[h] - safe);
    }
    uint32_t pa[8][2];
    float rs[2] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float p[4];
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int h = r >> 1;
        const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
        p[r] = exp2f(fmaf(sacc[4 * j + r], sc2, -safe));
        rs[h] += p[r];
      }
      pa[j][0] = pack_h2(p[0], p[1]);
      pa[j][1] = pack_h2(p[2], p[3]);
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
      for (int j = 0; j < 8; j++) {
        o[4 * j] *= corr[0]; o[4 * j + 1] *= corr[0];
        o[4 * j + 2] *= corr[1]; o[4 * j + 3] *= corr[1];
      }
    }
    // ---- O += P V : four k16 steps over the 64 keys ----
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
      if (!(dbg & 2)) {
        if (dbg & 16) {
          if (dbg & 32) wg_pvt(o, pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1], mkdesc(vt + kk * 2048, 1024, 128));
          else wg_pvt(o, pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1], mkdesc(vt + kk * 2048, 128, 1024));
        } else wg_pv(o, pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1], mkdesc(vt + kk * 256, 128, 1024));
      }
    wg_commit_wait();
  }
  cp_wait<0>();
  // each thread owns rows warp*16 + g and +8
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = warp * 16 + g + 8 * h;
    const float ls = l_run[h];
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

extern "C" int wg_run(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                      const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                      int window, int splits, int chunk, float sm_scale, int scaled, int dbg, cudaStream_t st) {
  const size_t rmax = ((chunk + 63) / 64) * 64;
  const size_t dyn = 1024 + 4096 + 8192 + (size_t)WS * 8192 + rmax * 4;
  cudaFuncSetAttribute(wg_partials, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  wg_partials<<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                      reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                      table_n, page, window, splits, chunk, sm_scale, scaled, dbg);
  return 0;
}
"""
