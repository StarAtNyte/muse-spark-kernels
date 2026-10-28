"""Exact block-diagonal two-head FP16 P×V sparse tensor-core primitive."""
CPP = 'torch::Tensor sparse_pv(torch::Tensor p, torch::Tensor v);'
CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>
__global__ void pv_sp(const uint16_t* P,const uint16_t* V,float* O,int keys) {
  const int lane=threadIdx.x,g=lane>>2,t=lane&3,b=blockIdx.x,db=blockIdx.y*8;
  P+=(size_t)b*16*keys;V+=(size_t)b*keys*2*64;
  float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
  const uint32_t meta=0xEEEE4444u; // low16=row g/head0, high16=row g+8/head1
  for(int kb=0;kb<keys;kb+=16) {
    const uint32_t a0=*reinterpret_cast<const uint32_t*>(P+g*keys+kb+2*t);
    const uint32_t a1=*reinterpret_cast<const uint32_t*>(P+(g+8)*keys+kb+2*t);
    const uint32_t a2=*reinterpret_cast<const uint32_t*>(P+g*keys+kb+8+2*t);
    const uint32_t a3=*reinterpret_cast<const uint32_t*>(P+(g+8)*keys+kb+8+2*t);
    uint32_t bv[4];
#pragma unroll
    for(int j=0;j<4;j++) {
      const int key=kb+4*j+2*(t>>1),head=t&1;
      const uint16_t lo=V[(key*2+head)*64+db+g];
      const uint16_t hi=V[((key+1)*2+head)*64+db+g];
      bv[j]=uint32_t(lo)|(uint32_t(hi)<<16);
    }
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0;\n"
      : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(bv[0]),"r"(bv[1]),"r"(bv[2]),"r"(bv[3]),"r"(meta));
  }
  O+=(size_t)b*16*64;
  O[g*64+db+2*t]=c0;O[g*64+db+2*t+1]=c1;
  O[(g+8)*64+db+2*t]=c2;O[(g+8)*64+db+2*t+1]=c3;
}
torch::Tensor sparse_pv(torch::Tensor p,torch::Tensor v) {
  TORCH_CHECK(p.is_cuda()&&v.is_cuda()&&p.is_contiguous()&&v.is_contiguous(),"CUDA contiguous");
  TORCH_CHECK(p.dim()==3&&p.size(1)==16&&p.size(2)%16==0,"p shape");
  TORCH_CHECK(v.dim()==4&&v.size(0)==p.size(0)&&v.size(1)==p.size(2)&&v.size(2)==2&&v.size(3)==64,"v shape");
  auto out=torch::empty({p.size(0),16,64},p.options().dtype(torch::kFloat32));
  pv_sp<<<dim3(p.size(0),8),32,0,at::cuda::getCurrentCUDAStream()>>>(
    static_cast<const uint16_t*>(p.data_ptr()),static_cast<const uint16_t*>(v.data_ptr()),out.data_ptr<float>(),p.size(2));
  auto status=cudaGetLastError();TORCH_CHECK(status==cudaSuccess,cudaGetErrorString(status));
  return out;
}
'''
