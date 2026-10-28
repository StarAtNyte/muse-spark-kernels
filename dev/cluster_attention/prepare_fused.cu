#include <cooperative_groups.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
__device__ __forceinline__ uint32_t norm_rope4(const __nv_bfloat16* src,int col,float pos,float theta,float scale,unsigned mask){
 float v[4];
#pragma unroll
 for(int j=0;j<4;j++)v[j]=__bfloat162float(src[col+j]);
 float sq=v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3];
#pragma unroll
 for(int k=1;k<=8;k*=2)sq+=__shfl_xor_sync(mask,sq,k);
 float inv=rsqrtf(sq/64.f+1.e-5f);uint32_t packed=0;
#pragma unroll
 for(int j=0;j<4;j+=2){
   float angle=pos*expf(-(float)(col+j)*(1.f/64.f)*logf(theta));
   float cs=cosf(angle),sn=sinf(angle),x=v[j]*inv,y=v[j+1]*inv;
   uint32_t a=__nv_cvt_float_to_fp8((x*cs-y*sn)*scale,__NV_SATFINITE,__NV_E4M3);
   uint32_t b=__nv_cvt_float_to_fp8((y*cs+x*sn)*scale,__NV_SATFINITE,__NV_E4M3);
   packed|=a<<(j*8);packed|=b<<((j+1)*8);
 }
 return packed;
}
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
__device__ __forceinline__ void pv8_ldt(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}
__device__ __forceinline__ void wg_pv8(float* c,uint32_t* a,uint64_t b){
 asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},{%32,%33,%34,%35},%36,1,1,1;"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}

__device__ __forceinline__ uint32_t packp8(float a,float b){uint16_t y;
 asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(y):"f"(a),"f"(b));return y;}

#define WS 2   // raw stages
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

#define LAZY_TH 0.0f
__global__ void __cluster_dims__(8,1,1) __launch_bounds__(128, 4)
wg5_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg, const __nv_bfloat16* GATE, __nv_bfloat16* OUT, int stride, const long* POS, const long* LOC, float theta) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x, wgi = 0, tw = tid & 127, lane = tid & 31, warp = (tid >> 5) & 3;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * 2 + kh) * splits + split) * 64;
  const int n = end > begin ? end - begin : 0;
  const int nst = (n + WT - 1) / WT;

  uint8_t* qs = smem;
  uint8_t* vt = smem + 4096;
  uint8_t* stages = smem + 4096 + 4096;
  int* rows = reinterpret_cast<int*>(stages + WS * 8192);

  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }

  auto prep_cluster=cooperative_groups::this_cluster();
  int prep_token=b*8+split,col=(tid&15)*4,head=tid>>4;
  float pos=(float)POS[prep_token];
  const __nv_bfloat16* input=GATE+(long)prep_token*stride;
  uint32_t qpack=norm_rope4(input+(kh*8+head)*64,col,pos,theta,1.f,0xffffffff);
  *reinterpret_cast<uint32_t*>(qs+core_off(split*8+head,col>>4)+(col&15))=qpack;
  if(tid<16){
    uint32_t kpack=norm_rope4(input+(HQ+kh)*64,col,pos,theta,1.f/key_scale,0xffff);
    uint32_t vpack=0;
#pragma unroll
    for(int j=0;j<4;j++)vpack|=(uint32_t)__nv_cvt_float_to_fp8(__bfloat162float(input[(HQ+2+kh)*64+col+j])/value_scale,__NV_SATFINITE,__NV_E4M3)<<(8*j);
    long slot=LOC[prep_token]*128+kh*64+col;
    *reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(KC)+slot)=kpack;
    *reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(VC)+slot)=vpack;
  }
  prep_cluster.sync();
  for(int i=tid;i<256;i+=128){
    int source=i/32;
    if(source!=split){
      uint8_t* remote=prep_cluster.map_shared_rank(smem,source);
      *reinterpret_cast<uint4*>(qs+i*16)=*reinterpret_cast<uint4*>(remote+i*16);
    }
  }
  // No rank may reuse its original query fragment until all readers finish.
  prep_cluster.sync();
  proxy_fence();
  __syncthreads();

  // ---- loop-invariant addressing ----
  const uint32_t stages_s = (uint32_t)__cvta_generic_to_shared(stages);
  const int ci = tid & 7;                   // 16-byte chunk within the key row (K 0-7, V 8-15)
  const int cc = ci & 3;
  const bool isv = ci >= 4;
  const uint8_t* srcb = (isv ? VC : KC) + kh * 64 + cc * 16;
  uint32_t dst_off[4];
  int key4[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int key = (tid >> 3) + 16 * i;
    key4[i] = key;
    dst_off[i] = isv ? (4096 + key * 64 + (cc ^ (key & 3)) * 16) : core_off(key, cc);
  }
  auto issue = [&](int s, int stage) {
    if (s < nst && !(dbg & 8)) {
      const uint32_t base = stages_s + stage * 8192;
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
  const uint64_t kd0 = mkdesc(stages, 128, 512);
  const uint64_t vd0 = mkdesc(vt,128,512);
  // per-thread V conversion offsets (loop invariant)
  // rotated chunk assignment: lanes take consecutive keys but different 16 B chunks, so both the smem loads and the stores are (nearly) conflict free
  const int vkey = (tw & 31) + 32 * (tw >> 6);
  const int vcb = ((tw >> 5) & 1) * 2;
  uint32_t vsrc_off[2], vdst_off[4];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int cs = vcb + j;
    vsrc_off[j] = vkey * 64 + (cs ^ (vkey & 3)) * 16;
    vdst_off[2 * j] = (uint32_t)vmn_off(vkey, 16 * cs);
    vdst_off[2 * j + 1] = (uint32_t)vmn_off(vkey, 16 * cs + 8);
  }

  float o[32];
  float ls_run[2]={0.f,0.f};
#pragma unroll
  for (int i = 0; i < 32; i++) o[i] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY};

  int rd_stage = 0;
#pragma unroll 1
  for (int s = 0; s < nst; s++) {
    cp_wait<WS - 2>();
    if (!(dbg & 128)) { proxy_fence(); __syncthreads(); }
    issue(s + WS - 1, wr_stage);
    wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1;
    const uint8_t* vst = stages + rd_stage * 8192 + 4096;
    const uint64_t kdesc = kd0 + (uint64_t)(rd_stage * 512);
    rd_stage = rd_stage + 1 == WS ? 0 : rd_stage + 1;

    float sacc[32];
    // Key-permuted FP8 transpose: P uses the identical within-block order.
#pragma unroll
    for(int pr=0;pr<2;pr++){
      int key0=warp*16+(lane&7)+8*((lane>>3)&1),byte=pr*32+16*(lane>>4);
      uint32_t r[4];pv8_ldt(r,vst+key0*64+(((byte>>4)^(key0&3))<<4));
      uint32_t z[4]={__byte_perm(r[0],r[1],0x6420),__byte_perm(r[0],r[1],0x7531),
                     __byte_perm(r[2],r[3],0x6420),__byte_perm(r[2],r[3],0x7531)};
      int matrix=lane>>3,d=pr*32+(matrix>>1)*16+2*(lane&7)+(matrix&1);
      uint32_t ad=__cvta_generic_to_shared(vt+core_off(d,warp));
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(ad),"r"(z[0]),"r"(z[1]),"r"(z[2]),"r"(z[3]):"memory");
    }
    proxy_fence();
    wg_fence();
    wg_qk_first(sacc, qd0, kdesc);
    wg_qk_acc(sacc, qd1, kdesc + 16);
    wg_commit();
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
      m_run[0] = n0; m_run[1] = n1;ls_run[0]*=c0;ls_run[1]*=c1;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        o[4 * j] *= c0; o[4 * j + 1] *= c0;
        o[4 * j + 2] *= c1; o[4 * j + 3] *= c1;
      }
    }
    const float sf0 = m_run[0] == -INFINITY ? 0.f : m_run[0], sf1 = m_run[1] == -INFINITY ? 0.f : m_run[1];
    uint32_t pa[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float p0 = (dbg & 32) ? sacc[4 * j] : ex2f(fmaf(sacc[4 * j], sc2, 8.f-sf0)), p1 = (dbg & 32) ? sacc[4 * j + 1] : ex2f(fmaf(sacc[4 * j + 1], sc2, 8.f-sf0));
      const float p2 = (dbg & 32) ? sacc[4 * j + 2] : ex2f(fmaf(sacc[4 * j + 2], sc2, 8.f-sf1)), p3 = (dbg & 32) ? sacc[4 * j + 3] : ex2f(fmaf(sacc[4 * j + 3], sc2, 8.f-sf1));
      ls_run[0]+=p0+p1;ls_run[1]+=p2+p3;
      pa[j][0] = packp8(p0,p1);
      pa[j][1] = packp8(p2,p3);
    }
    uint32_t pp[2][4];
#pragma unroll
    for(int kk=0;kk<2;kk++){
#pragma unroll
      for(int h=0;h<2;h++){
        pp[kk][h]=pa[kk*4][h]|(pa[kk*4+1][h]<<16);
        pp[kk][h+2]=pa[kk*4+2][h]|(pa[kk*4+3][h]<<16);
      }
    }
    wg_fence();wg_pv8(o,pp[0],vd0);wg_pv8(o,pp[1],vd0+16);
    wg_commit_wait();
#pragma unroll
    for (int i = 0; i < 32; i++) asm volatile("" : "+f"(o[i])::"memory");
  }

  cp_wait<0>();
  __syncthreads();
  float* shared=reinterpret_cast<float*>(smem);
#pragma unroll
  for(int h=0;h<2;h++){
    int row=warp*16+g+8*h;
    float ls=ls_run[h];ls+=__shfl_xor_sync(0xffffffff,ls,1);ls+=__shfl_xor_sync(0xffffffff,ls,2);
    float inv=ls>0.f?1.f/ls:1.f;
#pragma unroll
    for(int j=0;j<8;j++){
      shared[row*64+j*8+2*t]=o[4*j+2*h]*inv*value_scale;
      shared[row*64+j*8+2*t+1]=o[4*j+2*h+1]*inv*value_scale;
    }
    if(t==0)shared[4096+row]=ls>0.f?m_run[h]+log2f(ls)-8.f:-INFINITY;
  }
  auto cluster=cooperative_groups::this_cluster();
  cluster.sync();
  int row=split*8+warp*2+(lane>>4),d=(lane&15)*4;
  float high=-INFINITY;
#pragma unroll
  for(int r=0;r<8;r++)high=fmaxf(high,cluster.map_shared_rank(shared,r)[4096+row]);
  float den=0.f,x[4]={0.f,0.f,0.f,0.f};
#pragma unroll
  for(int r=0;r<8;r++){
    float* remote=cluster.map_shared_rank(shared,r);float mx=remote[4096+row];
    float w=mx==-INFINITY?0.f:exp2f(mx-high);den+=w;
    float4 v=*reinterpret_cast<float4*>(remote+row*64+d);
    x[0]=fmaf(v.x,w,x[0]);x[1]=fmaf(v.y,w,x[1]);x[2]=fmaf(v.z,w,x[2]);x[3]=fmaf(v.w,w,x[3]);
  }
  float square=0.f;
#pragma unroll
  for(int j=0;j<4;j++){x[j]=__bfloat162float(__float2bfloat16_rn(x[j]/(den>0.f?den:1.f)));square+=x[j]*x[j];}
#pragma unroll
  for(int k=1;k<=8;k*=2)square+=__shfl_xor_sync(0xffffffff,square,k);
  float inv=rsqrtf(square/64.f+1.e-5f);
  int token=row>>3,qh=kh*8+(row&7);
  long go=(long)(b*L+token)*stride+(HQ+4+qh)*64+d;
  long oo=((long)(b*L+token)*HQ+qh)*64+d;
#pragma unroll
  for(int j=0;j<4;j++)OUT[oo+j]=__float2bfloat16_rn(x[j]*inv/(1.f+expf(-__bfloat162float(GATE[go+j]))));
  cluster.sync();

#endif
}



extern "C" int fused_launch(const void* qkv,const long* pos,const long* loc,void* kc,void* vc,const int* table,const int* seq,const float* ks,const float* vs,void* out,int B,int tn,int page,int chunk,float scale,int scaled,int stride,float theta,cudaStream_t stream){
 int dyn=1024+4096+4096+WS*8192+((chunk+255)/256)*256*4;
 cudaFuncSetAttribute(wg5_partials,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);
 wg5_partials<<<dim3(8,2,B),128,dyn,stream>>>(nullptr,(const uint8_t*)kc,(const uint8_t*)vc,table,seq,ks,vs,nullptr,nullptr,8,16,tn,page,2048,8,chunk,scale,scaled,0,(const __nv_bfloat16*)qkv,(__nv_bfloat16*)out,stride,pos,loc,theta);
 return cudaGetLastError();
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int fused_launch(const void*,const long*,const long*,void*,void*,const int*,const int*,const float*,const float*,void*,int,int,int,int,float,int,int,float,cudaStream_t);
torch::Tensor fused_attention(torch::Tensor qkv,torch::Tensor pos,torch::Tensor loc,torch::Tensor kc,torch::Tensor vc,torch::Tensor table,torch::Tensor seq,torch::Tensor ks,torch::Tensor vs,int64_t page,int64_t chunk,double scale,int64_t scaled,double theta){
 auto out=torch::empty({seq.numel(),8,1024},qkv.options());
 int e=fused_launch(qkv.data_ptr(),pos.data_ptr<long>(),loc.data_ptr<long>(),kc.data_ptr(),vc.data_ptr(),table.data_ptr<int>(),seq.data_ptr<int>(),ks.data_ptr<float>(),vs.data_ptr<float>(),out.data_ptr(),seq.numel(),table.size(1),page,chunk,scale,scaled,qkv.stride(0),theta,at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));return out;
}
