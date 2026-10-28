
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


template<int MODE,int WK>
__global__ void __launch_bounds__(WK*64,1)
wide(const uint4* W,const uint2* SF,const float* GS,const bf16* X,bf16* OUT,
     const int* lst,const int* act,const int* nact,int lmax){
 __shared__ float red[WK*2*128];
 pdl_wait();pdl_trigger();
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,wr=warp&1,wk=warp>>1;
 constexpr int CPW=64/WK,GPW=CPW/4,TILES=MODE==0?256:128;
 int count=__ldcg(nact);
 for(int job=blockIdx.x;job<count*TILES;job+=gridDim.x){
  int slot=job/TILES,bx=job%TILES;
  int4 item=__ldcg(reinterpret_cast<const int4*>(act)+slot);
  int e=item.x,ntok=item.z;const int* list=lst+e*lmax+item.y;
  int tile=MODE==0?(wr?256+bx:bx):(bx*2+wr);
  int rt=MODE==0?512:256;
  const uint4* wp=W+((size_t)e*rt+tile)*64*32+wk*CPW*32+lane;
  const uint2* sp=SF+((size_t)e*rt+tile)*16*32+wk*GPW*32+lane;
  int col=lane>>2,t=lane&3;
  bool valid=col<ntok;int p=valid?__ldcg(list+col):0;
  const bf16* xp=X+(size_t)(MODE==0?(p>>3):p)*4096+wk*CPW*64+16*t;
  float acc[4]={};
#pragma unroll
  for(int q=0;q<GPW;q++){
   uint2 sc=__ldg(sp+q*32);uint4 weights[4];
#pragma unroll
   for(int c=0;c<4;c++)weights[c]=__ldg(wp+(q*4+c)*32);
#pragma unroll
   for(int c=0;c<4;c++){
    uint4 xa=make_uint4(0,0,0,0),xb=xa;
    if(valid){const uint4* v=reinterpret_cast<const uint4*>(xp+(q*4+c)*64);xa=__ldcg(v);xb=__ldcg(v+1);}
    uint32_t s0=scale2(sc.x,c),s1=scale2(sc.y,c);
    uint32_t words[4]={weights[c].x,weights[c].y,weights[c].z,weights[c].w};
#pragma unroll
    for(int ks=0;ks<4;ks++){
     uint32_t w=words[ks];const uint4 xx=ks<2?xa:xb;
     mma16816(acc,hmul(dec<0>(w),s0),hmul(dec<1>(w),s1),hmul(dec<2>(w),s0),hmul(dec<3>(w),s1),(ks&1)?xx.z:xx.x,(ks&1)?xx.w:xx.y);
    }
   }
  }
#pragma unroll
  for(int r=0;r<4;r++)red[(warp*4+r)*32+lane]=acc[r];
  __syncthreads();
  for(int it=tid;it<128;it+=WK*64){
   int li=it&31,r=it>>5,row=(li>>2)+8*(r>>1),token=2*(li&3)+(r&1);
   if(token<ntok){
    int outp=__ldcg(list+token);float a=0.f,b=0.f;
#pragma unroll
    for(int k=0;k<WK;k++){a+=red[(k*2*4+r)*32+li];b+=red[((k*2+1)*4+r)*32+li];}
    a=bfr(bfr(a*256.f)*GS[e]);b=bfr(bfr(b*256.f)*GS[e]);
    if constexpr(MODE==0)OUT[(size_t)outp*4096+bx*16+row]=__float2bfloat16_rn((a/(1.f+__expf(-a)))*b);
    else{OUT[(size_t)outp*4096+(bx*2)*16+row]=__float2bfloat16_rn(a);OUT[(size_t)outp*4096+(bx*2+1)*16+row]=__float2bfloat16_rn(b);}
   }
  }
  __syncthreads();
 }
}
extern "C" int wide_c(int mode,int wk,int cap,const void* W,const void* SF,const float* GS,const void* X,void* Y,const int* lst,const int* act,const int* nact,int lmax,cudaStream_t stream){
#define L(M,K) if(mode==M&&wk==K){launch_k(wide<M,K>,dim3(cap),dim3(K*64),0,stream,g_pdl!=0,(const uint4*)W,(const uint2*)SF,GS,(const bf16*)X,(bf16*)Y,lst,act,nact,lmax);return cudaGetLastError();}
 L(0,4) L(1,4) L(0,8) L(1,8) L(0,16) L(1,16)
#undef L
 return cudaErrorInvalidValue;
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" void wm_set_pdl(int);
extern "C" int wide_c(int,int,int,const void*,const void*,const float*,const void*,void*,const int*,const int*,const int*,int,cudaStream_t);
void wide_set_pdl(int64_t value){wm_set_pdl(value);}
void wm_wide_run(int64_t mode,int64_t wk,int64_t cap,torch::Tensor W,torch::Tensor SF,torch::Tensor GS,torch::Tensor X,torch::Tensor Y,torch::Tensor lst,torch::Tensor act,torch::Tensor nact,int64_t lmax){
 int e=wide_c(mode,wk,cap,W.data_ptr(),SF.data_ptr(),GS.data_ptr<float>(),X.data_ptr(),Y.data_ptr(),lst.data_ptr<int>(),act.data_ptr<int>(),nact.data_ptr<int>(),lmax,at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));
}
