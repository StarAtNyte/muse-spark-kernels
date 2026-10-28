"""M=1 fused routed MoE with occupancy-bounded cooperative grid barriers."""
from pathlib import Path
import ast
s=Path('kernels/muse_moe_w4a16/submission.py').read_text()
cu=ast.literal_eval(next(n.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WM_CUDA' for t in n.targets)))
original_cu=cu
cut=cu.index('template <int MODE, int NT, int NMAX, int NW, int RING, int DBG>\n__global__')
cu='#include <cooperative_groups.h>\n'+cu[:cut]
cu+=r'''
__global__ void __launch_bounds__(128,4) moe_coop(
 const uint4* W13,const uint2* S13,const float* GS13,const uint4* W2,const uint2* S2,const float* GS2,
 const bf16* X,const int* IDS,const float* TW,const bf16* NWT,const int* PIDS,
 bf16* INTER,bf16* Y,float* NSQ,bf16* OUT,float eps,int prefetch){
 extern __shared__ uint4 scratch[];
 __shared__ int experts[8],pairs[8],nlocal;
 __shared__ float scales[8];
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 if(tid==0){int n=0;
#pragma unroll
  for(int p=0;p<8;p++){int e=IDS[p];if((unsigned)e<32){experts[n]=e;pairs[n]=p;n++;}}
  nlocal=n;
 }
 __syncthreads();
 auto grid=cooperative_groups::this_grid();
 // Fixed resident workers distribute only live experts' work. Nothing exits
 // before the grid barriers, including workers with no local work.
 for(int job=blockIdx.x;job<nlocal*256;job+=gridDim.x){
  int a=job>>8,bx=job&255,e=experts[a],p=pairs[a];
  if(prefetch && tid==0){
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(policy));
    const void* wp=W2+((size_t)e*256+bx)*64*32;
    const void* sp=S2+((size_t)e*256+bx)*16*32;
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"::"l"(wp),"r"(32768),"l"(policy):"memory");
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"::"l"(sp),"r"(4096),"l"(policy):"memory");
  }
  wm_body<0,2,2,1,4,0,6,0>(W13,S13,GS13,X,INTER,PIDS+p,e,1,0,scratch,bx);
  __syncthreads();
 }
 grid.sync();
 for(int job=blockIdx.x;job<nlocal*256;job+=gridDim.x){
  int a=job>>8,bx=job&255,e=experts[a],p=pairs[a];
  wm_body<1,1,1,1,4,0,6,0>(W2,S2,GS2,INTER,Y,PIDS+p,e,1,0,scratch,bx);
  __syncthreads();
  if(tid<16){
   int d=bx*16+tid;float v=__bfloat162float(Y[p*4096+d])*__bfloat162float(NWT[d]);
   float square=v*v;
#pragma unroll
   for(int k=8;k;k>>=1)square+=__shfl_xor_sync(0xffff,square,k);
   if(tid==0)NSQ[p*256+bx]=square;
  }
  __syncthreads();
 }
 grid.sync();
 // Two blocks finish disjoint halves of the output. Norm sufficient
 // statistics come from the down-projection producers, after BF16 rounding.
 if(blockIdx.x<2){
#pragma unroll
  for(int pass=0;pass<2;pass++){
   int p=warp+pass*4;float ss=0.f;bool live=(unsigned)IDS[p]<32;
   if(live){
#pragma unroll
    for(int j=0;j<8;j++)ss+=NSQ[p*256+lane+j*32];
#pragma unroll
    for(int k=16;k;k>>=1)ss+=__shfl_xor_sync(0xffffffff,ss,k);
   }
   if(lane==0)scales[p]=live?TW[p]*rsqrtf(ss/4096.f+eps):0.f;
  }
  __syncthreads();
  int column=(blockIdx.x*128+tid)*16;
  float acc[16]={},nw[16];
#pragma unroll
  for(int j=0;j<16;j++)nw[j]=__bfloat162float(NWT[column+j]);
#pragma unroll
  for(int p=0;p<8;p++){
   float scale=scales[p];
   if(scale!=0.f){
#pragma unroll
    for(int j=0;j<16;j++)acc[j]+=scale*(__bfloat162float(Y[p*4096+column+j])*nw[j]);
   }
  }
#pragma unroll
  for(int j=0;j<16;j++)OUT[column+j]=__float2bfloat16_rn(acc[j]);
 }
}
extern "C" int coop_launch(const void* w13,const void* s13,const float* gs13,const void* w2,const void* s2,const float* gs2,
 const void* x,const int* ids,const float* tw,const void* nw,const int* pids,void* inter,void* y,float* nsq,void* out,float eps,int cap,int prefetch,cudaStream_t stream){
 int dev=0,sms=0,supported=0,resident=0;
 cudaError_t e=cudaGetDevice(&dev);if(e)return e;
 e=cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,dev);if(e)return e;
 if(!supported)return cudaErrorNotSupported;
 e=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);if(e)return e;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,moe_coop,128,4096);if(e)return e;
 int blocks=sms*resident;if(cap>0&&blocks>cap)blocks=cap;
 if(blocks<2)return cudaErrorCooperativeLaunchTooLarge;
 const uint4* W13=(const uint4*)w13;const uint2* S13=(const uint2*)s13;
 const uint4* W2=(const uint4*)w2;const uint2* S2=(const uint2*)s2;
 const bf16* X=(const bf16*)x;const bf16* NWT=(const bf16*)nw;
 bf16* INTER=(bf16*)inter;bf16* Y=(bf16*)y;bf16* OUT=(bf16*)out;
 void* args[]={&W13,&S13,&gs13,&W2,&S2,&gs2,&X,&ids,&tw,&NWT,&pids,&INTER,&Y,&nsq,&OUT,&eps,&prefetch};
 return cudaLaunchCooperativeKernel((void*)moe_coop,dim3(blocks),dim3(128),args,4096,stream);
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int coop_launch(const void*,const void*,const float*,const void*,const void*,const float*,const void*,const int*,const float*,const void*,const int*,void*,void*,float*,void*,float,int,int,cudaStream_t);
torch::Tensor cooperative_moe(torch::Tensor w13,torch::Tensor s13,torch::Tensor gs13,torch::Tensor w2,torch::Tensor s2,torch::Tensor gs2,torch::Tensor x,torch::Tensor ids,torch::Tensor tw,torch::Tensor nw,torch::Tensor pids,double eps,int64_t cap,int64_t prefetch){
 TORCH_CHECK(x.size(0)==1,"single-token prototype");
 auto inter=torch::empty({8,4096},x.options()),y=torch::empty_like(inter),out=torch::empty_like(x);
 auto nsq=torch::empty({8,256},x.options().dtype(torch::kFloat32));
 int e=coop_launch(w13.data_ptr(),s13.data_ptr(),gs13.data_ptr<float>(),w2.data_ptr(),s2.data_ptr(),gs2.data_ptr<float>(),x.data_ptr(),ids.data_ptr<int>(),tw.data_ptr<float>(),nw.data_ptr(),pids.data_ptr<int>(),inter.data_ptr(),y.data_ptr(),nsq.data_ptr<float>(),out.data_ptr(),eps,cap,prefetch,at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));return out;
}
'''
cpp='torch::Tensor cooperative_moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,int64_t,int64_t);'
Path('dev/moe_cooperative/kernel.cu').write_text(cu)
original_cpp=ast.literal_eval(next(n.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WM_CPP' for t in n.targets)))
a=original_cu.index('extern "C" int wm_gemm_c(');brace=original_cu.index('{',a);tail=original_cu.index('#include <torch/extension.h>',brace)
base=original_cu[:brace+1]+r'''
 if(nmax!=1 || nt!=2 || nw!=4 || ring!=0 || dbg!=0)return -1;
 if(mode==0)launch<0,2,1,4,0,0>(W,S,GS,X,OUT,cnt,lst,act,nact,woff,maxslots,lmax,g_pdl,st);
 else launch<1,2,1,4,0,0>(W,S,GS,X,OUT,cnt,lst,act,nact,woff,maxslots,lmax,g_pdl,st);
 return 0;
}
'''+original_cu[tail:]
Path('dev/moe_cooperative/probe.py').write_text('CPP='+repr(cpp)+'\nCUDA='+repr(cu)+'\nBASE_CPP='+repr(original_cpp)+'\nBASE_CUDA='+repr(base)+'\n')
