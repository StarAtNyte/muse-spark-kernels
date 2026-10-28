"""One launch: distribute eight-token preparation over eight cluster ranks.

Each rank prepares one token's queries and its head's new KV. A cluster barrier
publishes cache writes; DSM broadcasts each rank's 512-byte query fragment.
Distinct clusters touch disjoint batch/KV-head cache regions.
"""
from pathlib import Path
import runpy
m=runpy.run_path('dev/cluster_attention/fused.py');s=m['CUDA']
helper=r'''
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
'''
idx=s.index('#define HD');s=s[:idx]+helper+s[idx:]
s=s.replace('const __nv_bfloat16* GATE, __nv_bfloat16* OUT, int stride) {','const __nv_bfloat16* GATE, __nv_bfloat16* OUT, int stride, const long* POS, const long* LOC, float theta) {')
a=s.index('  for (int c = tw; c < 64 * 4; c += 128)');b=s.index('  proxy_fence();',a)
s=s[:a]+r'''
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
''' +s[b:]
a=s.index('extern "C" int cluster_launch(')
s=s[:a]+r'''
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
'''
cpp='torch::Tensor fused_attention(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,double,int64_t,double);'
Path('dev/cluster_attention/prepare_fused.cu').write_text(s)
Path('dev/cluster_attention/prepare_fused.py').write_text('CPP='+repr(cpp)+'\nCUDA='+repr(s)+'\n')
