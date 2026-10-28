
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
__global__ void __cluster_dims__(8,1,1) dsm_merge(const float* P,const float* M,const float* S,const __nv_bfloat16* G,__nv_bfloat16* O){
 __shared__ __align__(16) float sm[4096+128];
 cg::cluster_group cluster=cg::this_cluster();
 int rank=cluster.block_rank(),group=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 for(int i=tid;i<4096;i+=128)sm[i]=P[((long)group*8+rank)*4096+i];
 if(tid<64){sm[4096+tid]=M[(group*8+rank)*64+tid];sm[4160+tid]=S[(group*8+rank)*64+tid];}
 cluster.sync();
 int row=rank*8+warp*2+(lane>>4),d=(lane&15)*4;
 float high=-INFINITY;
#pragma unroll
 for(int r=0;r<8;r++){float* remote=cluster.map_shared_rank(sm,r);high=fmaxf(high,remote[4096+row]);}
 float sum=0.f,a=0.f,b=0.f,c=0.f,e=0.f;
#pragma unroll
 for(int r=0;r<8;r++){
  float* remote=cluster.map_shared_rank(sm,r);float mx=remote[4096+row],w=mx==-INFINITY?0.f:exp2f(mx-high);
  float4 v=*reinterpret_cast<float4*>(remote+row*64+d);sum=fmaf(remote[4160+row],w,sum);
  a=fmaf(v.x,w,a);b=fmaf(v.y,w,b);c=fmaf(v.z,w,c);e=fmaf(v.w,w,e);
 }
 float square=a*a+b*b+c*c+e*e;
#pragma unroll
 for(int x=1;x<=8;x*=2)square+=__shfl_xor_sync(0xffffffff,square,x);
 // rmsnorm(num/sum) == num / sqrt(mean(num^2) + eps*sum^2).
 float inv=rsqrtf(square*(1.f/64.f)+(sum>0.f?1.e-5f*sum*sum:1.f));
 long out=((long)group*64+row)*64+d;
 O[out]=__float2bfloat16_rn(a*inv/(1.f+expf(-__bfloat162float(G[out]))));
 O[out+1]=__float2bfloat16_rn(b*inv/(1.f+expf(-__bfloat162float(G[out+1]))));
 O[out+2]=__float2bfloat16_rn(c*inv/(1.f+expf(-__bfloat162float(G[out+2]))));
 O[out+3]=__float2bfloat16_rn(e*inv/(1.f+expf(-__bfloat162float(G[out+3]))));
 cluster.sync(); // Remote shared storage must outlive every reader.
}
extern "C" int dsm_launch(const float* p,const float* m,const float* s,const void* g,void* o,int groups,cudaStream_t stream){
 dsm_merge<<<dim3(8,groups),128,0,stream>>>(p,m,s,(const __nv_bfloat16*)g,(__nv_bfloat16*)o);return cudaGetLastError();}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int dsm_launch(const float*,const float*,const float*,const void*,void*,int,cudaStream_t);
torch::Tensor cluster_merge(torch::Tensor p,torch::Tensor m,torch::Tensor s,torch::Tensor gate){auto o=torch::empty({p.size(0),64,64},gate.options());int e=dsm_launch(p.data_ptr<float>(),m.data_ptr<float>(),s.data_ptr<float>(),gate.data_ptr(),o.data_ptr(),p.size(0),at::cuda::getCurrentCUDAStream());TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));return o;}
