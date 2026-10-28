"""Dense transposed FP8 QK / FP16 PV primitives for eight GQA query heads."""
CPP='std::vector<torch::Tensor> trans_probe(torch::Tensor q,torch::Tensor k,torch::Tensor p,torch::Tensor v);'
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ int off(int key,int byte){return key*128+(((byte>>4)^(key&7))<<4)+(byte&15);}
__device__ __forceinline__ uint32_t cv2(uint32_t x){uint32_t y;asm("{.reg .b16 a,b; mov.b32 {a,b},%1; cvt.rn.f16x2.e4m3x2 %0,a;}":"=r"(y):"r"(x));return y;}
__device__ __forceinline__ void ld(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}
__device__ __forceinline__ void ldt(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}
__device__ __forceinline__ void qmma(float* c,uint32_t* a,uint32_t b0,uint32_t b1){asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));}
__device__ __forceinline__ void pmma(float* c,uint32_t* a,uint32_t* b){asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));}
__global__ void probe(const uint8_t* Q,const uint8_t* K,const uint16_t* P,const uint8_t* V,float* S,float* O,int keys){
 __shared__ __align__(16) uint8_t mem[4096];
 int tid=threadIdx.x,lane=tid&31,h=tid>>5,g=lane>>2,t=lane&3,b=blockIdx.x;
 Q+=b*1024;K+=(long)b*keys*128;V+=(long)b*keys*128;P+=(long)b*16*keys;S+=(long)b*16*keys;O+=b*1024;
 float o[4][4]={};
 for(int kb=0;kb<keys;kb+=16){
  for(int i=tid;i<128;i+=64){int key=i>>3,byte=(i&7)*16;*reinterpret_cast<uint4*>(mem+off(key,byte))=*reinterpret_cast<const uint4*>(K+(kb+key)*128+byte);*reinterpret_cast<uint4*>(mem+2048+off(key,byte))=*reinterpret_cast<const uint4*>(V+(kb+key)*128+byte);}
  __syncthreads();
  float s[4]={};
#pragma unroll
  for(int k=0;k<2;k++){uint32_t a[4];ld(a,mem+off((lane&7)+8*((lane>>3)&1),h*64+k*32+16*(lane>>4)));const uint8_t* q=Q+(h*8+g)*64+k*32+t*4;qmma(s,a,*reinterpret_cast<const uint32_t*>(q),*reinterpret_cast<const uint32_t*>(q+16));}
  S[(h*8+2*t)*keys+kb+g]=s[0];S[(h*8+2*t+1)*keys+kb+g]=s[1];S[(h*8+2*t)*keys+kb+g+8]=s[2];S[(h*8+2*t+1)*keys+kb+g+8]=s[3];
  // Exercise the score-C to probability-B cross-lane permutation.
  uint32_t p0=P[(h*8+2*t)*keys+kb+g]|(uint32_t(P[(h*8+2*t+1)*keys+kb+g])<<16);
  uint32_t p8=P[(h*8+2*t)*keys+kb+g+8]|(uint32_t(P[(h*8+2*t+1)*keys+kb+g+8])<<16);
  int sa=8*t+(g>>1),sb=sa+4,shift=(g&1)*16;
  uint32_t z[2];z[0]=((__shfl_sync(0xffffffff,p0,sa)>>shift)&65535)|((__shfl_sync(0xffffffff,p0,sb)>>shift)<<16);z[1]=((__shfl_sync(0xffffffff,p8,sa)>>shift)&65535)|((__shfl_sync(0xffffffff,p8,sb)>>shift)<<16);
#pragma unroll
  for(int pr=0;pr<2;pr++){uint32_t r[4];ldt(r,mem+2048+off((lane&7)+8*((lane>>3)&1),h*64+pr*32+16*(lane>>4)));
#pragma unroll
   for(int odd=0;odd<2;odd++){uint32_t a[4];a[0]=cv2(__byte_perm(r[0],0,odd?0x4431:0x4420));a[1]=cv2(__byte_perm(r[2],0,odd?0x4431:0x4420));a[2]=cv2(__byte_perm(r[1],0,odd?0x4431:0x4420));a[3]=cv2(__byte_perm(r[3],0,odd?0x4431:0x4420));pmma(o[2*pr+odd],a,z);}}
  __syncthreads();
 }
#pragma unroll
 for(int pr=0;pr<2;pr++)for(int odd=0;odd<2;odd++){int d=pr*32+2*g+odd;float* c=o[2*pr+odd];O[(h*8+2*t)*64+d]=c[0];O[(h*8+2*t+1)*64+d]=c[1];O[(h*8+2*t)*64+d+16]=c[2];O[(h*8+2*t+1)*64+d+16]=c[3];}
}
std::vector<torch::Tensor> trans_probe(torch::Tensor q,torch::Tensor k,torch::Tensor p,torch::Tensor v){auto s=torch::empty({q.size(0),16,k.size(1)},q.options().dtype(torch::kFloat32));auto o=torch::empty({q.size(0),16,64},q.options().dtype(torch::kFloat32));probe<<<q.size(0),64,0,at::cuda::getCurrentCUDAStream()>>>((const uint8_t*)q.data_ptr(),(const uint8_t*)k.data_ptr(),(const uint16_t*)p.data_ptr(),(const uint8_t*)v.data_ptr(),s.data_ptr<float>(),o.data_ptr<float>(),k.size(1));auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));return {s,o};}
'''
