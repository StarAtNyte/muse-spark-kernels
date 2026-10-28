"""Transfer row/K warp partitioning to NVFP4 without expanding stored weights."""
from pathlib import Path
import ast,runpy
m=runpy.run_path('dev/moe_scale/probe.py');t=ast.parse(m['PARENT'])
cu=ast.literal_eval(next(n.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='_WM_CUDA' for v in n.targets)))
prefix=cu[:cu.index('__device__ __forceinline__ void cp16')]
kernel=r'''
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
'''
cuda=prefix+kernel
Path('dev/moe_wide/kernel.cu').write_text(cuda)
Path('dev/moe_wide/probe.py').write_text('PARENT='+repr(m['PARENT'])+'\nBASE_CPP='+repr(m['CPP'])+'\nBASE_CUDA='+repr(m['BASE'])+'\nCUDA='+repr(cuda)+'\nCPP='+repr('void wide_set_pdl(int64_t);\nvoid wm_wide_run(int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);')+'\n')
