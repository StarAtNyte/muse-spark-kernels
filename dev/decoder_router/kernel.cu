/* Adapted from Nathen Fernandes ROUTER_SCALAR: FP32 activation reads.
MIT License

Copyright (c) 2026 Nathen Fernandes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

template<int M, int NT, int ST>
__global__ void __launch_bounds__(NT,1) fp32_router(const float* __restrict__ w, const float* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  float4 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const float4*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      float4 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        acc[t][0]=fmaf(v.x,a[t].x,acc[t][0]);
        acc[t][1]=fmaf(v.y,a[t].y,acc[t][1]);
        acc[t][2]=fmaf(v.z,a[t].z,acc[t][2]);
        acc[t][3]=fmaf(v.w,a[t].w,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

extern "C" int fp32_router_c(const float* W,const float* X,float* Y,int M,int cfg,cudaStream_t stream){
#define R(MM,NT,ST,C) if(M<=MM&&cfg==C){fp32_router<MM,NT,ST><<<256,NT,0,stream>>>(W,X,Y,M);return cudaGetLastError();}
 if(M==1){R(1,128,2,0) R(1,256,1,1) R(1,256,2,2) R(1,512,1,3) R(1,512,2,4)}
 else if(M<=8){R(8,128,2,0) R(8,256,1,1) R(8,256,2,2) R(8,512,1,3) R(8,512,2,4)}
#undef R
 return cudaErrorInvalidValue;
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int fp32_router_c(const float*,const float*,float*,int,int,cudaStream_t);
torch::Tensor fp32_router_run(torch::Tensor W,torch::Tensor X,int64_t cfg){
 auto Y=torch::empty({X.size(0),256},X.options());
 int e=fp32_router_c(W.data_ptr<float>(),X.data_ptr<float>(),Y.data_ptr<float>(),X.size(0),cfg,at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));return Y;
}
