"""Native FP8 PV with exact byte transpose and derived C-to-A register packing."""
CPP='torch::Tensor pv8(torch::Tensor p,torch::Tensor v);'
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ int rawoff(int key,int byte){return key*64+(((byte>>4)^(key&3))<<4)+(byte&15);}
__device__ __forceinline__ int core(int row,int chunk){return (row>>3)*512+chunk*128+(row&7)*16;}
__device__ __forceinline__ void ldt(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}
__device__ __forceinline__ uint64_t desc(const void* p){uint32_t a=__cvta_generic_to_shared(p);return ((a>>4)&0x3fff)|((uint64_t)8<<16)|((uint64_t)32<<32);}
__device__ __forceinline__ void pv(float* c,uint32_t* a,uint64_t b){
 asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},{%32,%33,%34,%35},%36,1,1,1;"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}
__global__ void probe(const uint8_t* P,const uint8_t* V,float* O){
 __shared__ __align__(1024) uint8_t mem[8192];
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,t=lane&3,b=blockIdx.x;
 P+=(long)b*4096;V+=(long)b*4096;O+=(long)b*4096;
 for(int i=tid;i<256;i+=128){int key=i>>2,byte=(i&3)*16;*reinterpret_cast<uint4*>(mem+rawoff(key,byte))=*reinterpret_cast<const uint4*>(V+key*64+byte);}
 __syncthreads();
#pragma unroll
 for(int pr=0;pr<2;pr++){
  uint32_t r[4];ldt(r,mem+rawoff(warp*16+(lane&7)+8*((lane>>3)&1),pr*32+16*(lane>>4)));
#pragma unroll
  for(int j=0;j<4;j++){
   int d=pr*32+2*(g+(j>=2?8:0)),key=warp*16+(j&1)*8+2*t;
   *reinterpret_cast<uint16_t*>(mem+4096+core(d,key>>4)+(key&15))=__byte_perm(r[j],0,0x4420);
   *reinterpret_cast<uint16_t*>(mem+4096+core(d+1,key>>4)+(key&15))=__byte_perm(r[j],0,0x4431);
  }
 }
 asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
 float c[32]={};uint32_t pa[2][4];
#pragma unroll
 for(int kk=0;kk<2;kk++){
  int sa=g*4+2*(t&1),sb=sa+1,sh=(t>>1)*16;
#pragma unroll
  for(int h=0;h<2;h++){
   int row=warp*16+g+8*h;uint32_t v[4];
#pragma unroll
   for(int j=0;j<4;j++)v[j]=*reinterpret_cast<const uint16_t*>(P+row*64+kk*32+j*8+2*t);
   uint32_t z0=v[0]|(v[1]<<16),z1=v[2]|(v[3]<<16);
   pa[kk][h]=((__shfl_sync(0xffffffff,z0,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,z0,sb)>>sh)<<16);
   pa[kk][h+2]=((__shfl_sync(0xffffffff,z1,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,z1,sb)>>sh)<<16);
  }
 }
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
 pv(c,pa[0],desc(mem+4096));pv(c,pa[1],desc(mem+4096)+16);
 asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");
#pragma unroll
 for(int j=0;j<8;j++){
  O[(warp*16+g)*64+j*8+2*t]=c[4*j];O[(warp*16+g)*64+j*8+2*t+1]=c[4*j+1];
  O[(warp*16+g+8)*64+j*8+2*t]=c[4*j+2];O[(warp*16+g+8)*64+j*8+2*t+1]=c[4*j+3];
 }
}
torch::Tensor pv8(torch::Tensor p,torch::Tensor v){auto o=torch::empty({p.size(0),64,64},p.options().dtype(torch::kFloat32));probe<<<p.size(0),128,0,at::cuda::getCurrentCUDAStream()>>>((const uint8_t*)p.data_ptr(),(const uint8_t*)v.data_ptr(),o.data_ptr<float>());auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));return o;}
'''
