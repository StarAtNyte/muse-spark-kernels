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


__device__ __forceinline__ void wg_qk_first(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 0, 1, 1;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
               : "l"(da), "l"(db));
#endif
}
__device__ __forceinline__ void wg_qk_acc(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "l"(da), "l"(db));
#endif
}

#define LAZY_TH 8.0f
__global__ void __launch_bounds__(256, 2)
wg5_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
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

  uint8_t* qs = smem + wgi * 4096;
  uint8_t* vt = smem + 8192 + wgi * 9216;
  uint8_t* stages = smem + 8192 + 18432;
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
    if (tw < 64) asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ad), "r"(0x3C003C00u), "r"(0x3C003C00u), "r"(0x3C003C00u), "r"(0x3C003C00u));
  }
  proxy_fence();
  __syncthreads();

  // ---- loop-invariant addressing ----
  const uint32_t stages_s = (uint32_t)__cvta_generic_to_shared(stages);
  const int ci = tid & 15;                   // 16-byte chunk within the key row (K 0-7, V 8-15)
  const int cc = ci & 7;
  const bool isv = ci >= 8;
  const uint8_t* srcb = (isv ? VC : KC) + cc * 16;
  uint32_t dst_off[4];
  int key4[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int key = (tid >> 4) + 16 * i;
    key4[i] = key;
    dst_off[i] = isv ? (8192 + key * 128 + cc * 16) : ((cc >> 2) * 4096 + core_off(key, cc & 3));
  }
  auto issue = [&](int s, int stage) {
    if (s < nst && !(dbg & 8)) {
      const uint32_t base = stages_s + stage * 16384;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int li = s * WT + key4[i];
        const long rr = li < n ? rows[li] : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(base + dst_off[i]), "l"(srcb + rr * 128));
      }
    }
    cp_commit();
  };
  int wr_stage = 0;
#pragma unroll
  for (int s = 0; s < WS - 1; s++) { issue(s, wr_stage); wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1; }

  const uint64_t qd0 = mkdesc(qs, 128, 512), qd1 = mkdesc(qs + 256, 128, 512);
  const uint64_t kd0 = mkdesc(stages + wgi * 4096, 128, 512);
  const uint64_t vd0 = mkdesc(vt, 1152, 128);
  // per-thread V conversion offsets (loop invariant)
  const int vkey = tw >> 1, vdh = tw & 1;
  const uint32_t vsrc_off = vkey * 128 + wgi * 64 + vdh * 32;
  const uint32_t vdst_off[4] = {(uint32_t)vmn_off(vkey, vdh * 32), (uint32_t)vmn_off(vkey, vdh * 32 + 8),
                                (uint32_t)vmn_off(vkey, vdh * 32 + 16), (uint32_t)vmn_off(vkey, vdh * 32 + 24)};

  float o[36];
#pragma unroll
  for (int i = 0; i < 36; i++) o[i] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY};

  int rd_stage = 0;
#pragma unroll 1
  for (int s = 0; s < nst; s++) {
    cp_wait<WS - 2>();
    if (!(dbg & 128)) { proxy_fence(); __syncthreads(); }
    issue(s + WS - 1, wr_stage);
    wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1;
    const uint8_t* vst = stages + rd_stage * 16384 + 8192;
    const uint64_t kdesc = kd0 + (uint64_t)(rd_stage * 1024);
    rd_stage = rd_stage + 1 == WS ? 0 : rd_stage + 1;

    float sacc[32];
    wg_fence();
    wg_qk_first(sacc, qd0, kdesc);
    wg_qk_acc(sacc, qd1, kdesc + 16);
    wg_commit();
    // V raw [key][d] fp8 -> MN-major f16 core layout; overlaps with the QK wgmma
#pragma unroll
    for (int h2 = 0; h2 < ((dbg & 16) ? 0 : 2); h2++) {
      const uint4 u = *reinterpret_cast<const uint4*>(vst + vsrc_off + h2 * 16);
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int q4 = 0; q4 < 2; q4++) {
        uint4 o4;
        o4.x = cvt_fp8x2(w[2 * q4]); o4.y = cvt_fp8x2(w[2 * q4] >> 16);
        o4.z = cvt_fp8x2(w[2 * q4 + 1]); o4.w = cvt_fp8x2(w[2 * q4 + 1] >> 16);
        *reinterpret_cast<uint4*>(vt + vdst_off[2 * h2 + q4]) = o4;
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
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < ((dbg & 64) ? 0 : 8); j++) {
      mx0 = fmaxf(mx0, fmaxf(sacc[4 * j], sacc[4 * j + 1]));
      mx1 = fmaxf(mx1, fmaxf(sacc[4 * j + 2], sacc[4 * j + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    mx0 *= sc2; mx1 *= sc2;
    // lazy rescale: the reference max only moves when a tile exceeds it by more than LAZY_TH (log2 domain), so p <= 2^LAZY_TH
    if (__any_sync(0xffffffffu, mx0 > m_run[0] + LAZY_TH || mx1 > m_run[1] + LAZY_TH)) {
      const float n0 = fmaxf(m_run[0], mx0), n1 = fmaxf(m_run[1], mx1);
      const float s0 = n0 == -INFINITY ? 0.f : n0, s1 = n1 == -INFINITY ? 0.f : n1;
      const float c0 = ex2f(m_run[0] - s0), c1 = ex2f(m_run[1] - s1);
      m_run[0] = n0; m_run[1] = n1;
#pragma unroll
      for (int j = 0; j < 9; j++) {
        o[4 * j] *= c0; o[4 * j + 1] *= c0;
        o[4 * j + 2] *= c1; o[4 * j + 3] *= c1;
      }
    }
    const float sf0 = m_run[0] == -INFINITY ? 0.f : m_run[0], sf1 = m_run[1] == -INFINITY ? 0.f : m_run[1];
    uint32_t pa[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float p0 = (dbg & 32) ? sacc[4 * j] : ex2f(fmaf(sacc[4 * j], sc2, -sf0)), p1 = (dbg & 32) ? sacc[4 * j + 1] : ex2f(fmaf(sacc[4 * j + 1], sc2, -sf0));
      const float p2 = (dbg & 32) ? sacc[4 * j + 2] : ex2f(fmaf(sacc[4 * j + 2], sc2, -sf1)), p3 = (dbg & 32) ? sacc[4 * j + 3] : ex2f(fmaf(sacc[4 * j + 3], sc2, -sf1));
      pa[j][0] = pack_h2(p0, p1);
      pa[j][1] = pack_h2(p2, p3);
    }
    wg_fence();
    const uint64_t vd = vd0;
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
      if (!(dbg & 2)) wg_pv72(o, pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1], vd + (uint64_t)(kk * 144));
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

extern "C" int wg5_run(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                       const float* VS, float* PART, float* LSE, int B, int L, int HQ, int table_n, int page,
                       int window, int splits, int chunk, float sm_scale, int scaled, int dbg, cudaStream_t st) {
  const size_t rmax = ((chunk + 255) / 256) * 256;
  const size_t dyn = 1024 + 8192 + 18432 + (size_t)WS * 16384 + rmax * 4;
  cudaFuncSetAttribute(wg5_partials, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  wg5_partials<<<grid, 256, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                       reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ,
                                       table_n, page, window, splits, chunk, sm_scale, scaled, dbg);
  return 0;
}
"""
