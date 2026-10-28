"""Exact block-diagonal GQA packing for sparse FP8 tensor cores; no key pruning."""
CPP = 'torch::Tensor sparse_qk(torch::Tensor q, torch::Tensor k);'
CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>
__global__ void qk_sp(const uint8_t* Q,const uint8_t* K,float* O,int keys) {
  const int lane=threadIdx.x,g=lane>>2,t=lane&3,b=blockIdx.x,block=blockIdx.y;
  Q+=b*16*64;K+=(size_t)b*keys*128;
  float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
  const uint32_t meta=(t&1)?0xEEEEEEEEu:0x44444444u;
#pragma unroll
  for(int d0=0;d0<64;d0+=32) {
    const uint32_t a0=*reinterpret_cast<const uint32_t*>(Q+g*64+d0+t*4);
    const uint32_t a1=*reinterpret_cast<const uint32_t*>(Q+(g+8)*64+d0+t*4);
    const uint32_t a2=*reinterpret_cast<const uint32_t*>(Q+g*64+d0+16+t*4);
    const uint32_t a3=*reinterpret_cast<const uint32_t*>(Q+(g+8)*64+d0+16+t*4);
    uint32_t bv[4];
#pragma unroll
    for(int j=0;j<4;j++) {
      const int d=d0+8*j+2*t;
      const uint16_t lo=*reinterpret_cast<const uint16_t*>(K+(block*8+g)*128+d);
      const uint16_t hi=*reinterpret_cast<const uint16_t*>(K+(block*8+g)*128+64+d);
      bv[j]=uint32_t(lo)|(uint32_t(hi)<<16);
    }
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0;\n"
      : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(bv[0]),"r"(bv[1]),"r"(bv[2]),"r"(bv[3]),"r"(meta));
  }
  O+=(size_t)b*16*keys;
  O[g*keys+block*8+2*t]=c0;O[g*keys+block*8+2*t+1]=c1;
  O[(g+8)*keys+block*8+2*t]=c2;O[(g+8)*keys+block*8+2*t+1]=c3;
}
torch::Tensor sparse_qk(torch::Tensor q,torch::Tensor k) {
  TORCH_CHECK(q.is_cuda()&&k.is_cuda()&&q.is_contiguous()&&k.is_contiguous(),"CUDA contiguous");
  TORCH_CHECK(q.dim()==3&&q.size(1)==16&&q.size(2)==64,"q shape");
  TORCH_CHECK(k.dim()==4&&k.size(0)==q.size(0)&&k.size(2)==2&&k.size(3)==64&&k.size(1)%8==0,"k shape");
  auto out=torch::empty({q.size(0),16,k.size(1)},q.options().dtype(torch::kFloat32));
  qk_sp<<<dim3(q.size(0),k.size(1)/8),32,0,at::cuda::getCurrentCUDAStream()>>>(
    static_cast<const uint8_t*>(q.data_ptr()),static_cast<const uint8_t*>(k.data_ptr()),out.data_ptr<float>(),k.size(1));
  auto status=cudaGetLastError();TORCH_CHECK(status==cudaSuccess,cudaGetErrorString(status));
  return out;
}
'''
