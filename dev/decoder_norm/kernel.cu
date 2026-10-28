
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

__global__ void __launch_bounds__(1024,1) norm_route_kernel(const int* __restrict__ ids, int npair, const int* __restrict__ ls_ptr, int ls_val, int lmax,
                             int* __restrict__ cnt, int* __restrict__ lst, int* __restrict__ act,
                             int* __restrict__ nact, int* __restrict__ local_ids, int* __restrict__ woff, int grp, int nslots, const bf16* X, const bf16* W, bf16* Y, int M) {
  __shared__ int s_cnt[32];
  pdl_wait();
  const int ls = ls_ptr ? *ls_ptr : ls_val;
  pdl_trigger();

  __shared__ float partial[8][4];
  const int nr=threadIdx.x>>7,ni=threadIdx.x&127,wl=threadIdx.x&31,ww=(threadIdx.x&127)>>5;
  float sq=0.f;
  if(nr<M){
    for(int col=ni;col<4096;col+=128){float v=__bfloat162float(X[nr*4096+col]);sq=fmaf(v,v,sq);}
#pragma unroll
    for(int d=16;d>0;d>>=1)sq+=__shfl_down_sync(0xffffffff,sq,d);
    if(wl==0)partial[nr][ww]=sq;
  }

  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;
  for (int i = threadIdx.x; i < nslots; i += blockDim.x) reinterpret_cast<int4*>(act)[i] = make_int4(0, 0, 0, 0);  // work items {expert, offset, tokens, 0}
  __syncthreads();

  if(nr<M){
    float sum=(partial[nr][0]+partial[nr][1])+(partial[nr][2]+partial[nr][3]);
    float inv=rsqrtf(sum*(1.f/4096.f)+1.e-5f);
    for(int col=ni;col<4096;col+=128){
      float v=__bfloat162float(X[nr*4096+col]);
      float w=__bfloat162float(W[col]);
      Y[nr*4096+col]=__float2bfloat16_rn((v*inv)*w);
    }
  }

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

extern "C" int wm_norm_route_c(const int* ids,int npair,const int* ls,int lsv,int lmax,int* cnt,int* lst,int* act,int* nact,int* lids,int* woff,int grp,int nslots,const void* X,const void* W,void* Y,int M,cudaStream_t stream){
  if(M<1||M>8)return cudaErrorInvalidValue;
  launch_k(norm_route_kernel,dim3(1),dim3(1024),0,stream,g_pdl!=0,ids,npair,ls,lsv,lmax,cnt,lst,act,nact,lids,woff,grp,nslots,(const bf16*)X,(const bf16*)W,(bf16*)Y,M);
  return cudaGetLastError();
}

extern "C" void wm_route_probe_c(const int* ids,int npair,int lmax,int* cnt,int* lst,int* act,int* nact,int* lids,int* woff,int grp,int nslots,cudaStream_t stream){
 launch_k(route_kernel,dim3(1),dim3(1024),0,stream,g_pdl!=0,ids,npair,(const int*)nullptr,0,lmax,cnt,lst,act,nact,lids,woff,grp,nslots);
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" void wm_set_pdl(int);
extern "C" void wm_route_probe_c(const int*,int,int,int*,int*,int*,int*,int*,int*,int,int,cudaStream_t);
void wm_set_pdl_t(int64_t v){wm_set_pdl(v);}
void wm_route(torch::Tensor ids,int64_t lmax,torch::Tensor cnt,torch::Tensor lst,torch::Tensor act,torch::Tensor nact,torch::Tensor lids,torch::Tensor woff,int64_t grp){
 wm_route_probe_c(ids.data_ptr<int>(),ids.numel(),lmax,cnt.data_ptr<int>(),lst.data_ptr<int>(),act.data_ptr<int>(),nact.data_ptr<int>(),lids.data_ptr<int>(),woff.data_ptr<int>(),grp,act.numel()/4,at::cuda::getCurrentCUDAStream());
}

extern "C" int wm_norm_route_c(const int*,int,const int*,int,int,int*,int*,int*,int*,int*,int*,int,int,const void*,const void*,void*,int,cudaStream_t);
void wm_norm_route(torch::Tensor ids,torch::Tensor ls,int64_t lsv,int64_t lmax,torch::Tensor cnt,torch::Tensor lst,torch::Tensor act,torch::Tensor nact,torch::Tensor lids,torch::Tensor woff,int64_t grp,torch::Tensor X,torch::Tensor W,torch::Tensor Y){
 int e=wm_norm_route_c(ids.data_ptr<int>(),ids.numel(),ls.numel()?ls.data_ptr<int>():nullptr,lsv,lmax,cnt.data_ptr<int>(),lst.data_ptr<int>(),act.data_ptr<int>(),nact.data_ptr<int>(),lids.data_ptr<int>(),woff.data_ptr<int>(),grp,act.numel()/4,X.data_ptr(),W.data_ptr(),Y.data_ptr(),X.size(0),at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));
}
