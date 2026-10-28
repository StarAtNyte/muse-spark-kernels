
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

__device__ __forceinline__ int voff(int key,int byte){return key*256+(((byte>>4)^(key&15))<<4)+(byte&15);}
template<int ST>
__global__ void __launch_bounds__(512,2) trans_verify(const uint8_t* Q,const uint8_t* KC,const uint8_t* VC,
 const int* TABLE,const int* SEQ,const float* KS,const float* VS,float* PART,float* LSE,
 int table_n,int page,int window,int splits,int chunk,float sm_scale,int scaled){
 extern __shared__ __align__(16) uint8_t mem[];
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,h=warp&1,qt=warp>>1,g=lane>>2,t=lane&3,b=blockIdx.y,split=blockIdx.x;
 int prefix=SEQ[b],first=window?max(0,prefix-window+1):0,begin=first+split*chunk;
 int end=min(begin+chunk,prefix+8),n=max(0,end-begin),tiles=(n+31)/32;
 int* rows=reinterpret_cast<int*>(mem+ST*8192+8192);
 int shift=__ffs(page)-1,mask=page-1;
 for(int i=tid;i<n;i+=512){int tok=begin+i,pg=tok>>shift,phys=pg<table_n?TABLE[b*table_n+pg]:-1;rows[i]=phys<0?0:(phys<<shift)+(tok&mask);}
 uint32_t qb[2][2];
#pragma unroll
 for(int k=0;k<2;k++){const uint8_t* q=Q+((long)b*8+qt)*1024+(h*8+g)*64+k*32+t*4;qb[k][0]=*reinterpret_cast<const uint32_t*>(q);qb[k][1]=*reinterpret_cast<const uint32_t*>(q+16);}
 float scale=sm_scale*(scaled?KS[0]:1.f)*1.4426950408889634f,vscale=scaled?VS[0]:1.f;
 __syncthreads();
 auto issue=[&](int tile){if(tile<tiles){uint8_t* dst=mem+(tile%ST)*8192;
#pragma unroll
 for(int i=tid;i<256;i+=512){int key=i>>3,byte=(i&7)*16,li=tile*32+key;long rr=li<n?rows[li]:0;uint32_t ka=__cvta_generic_to_shared(dst+off(key,byte)),va=__cvta_generic_to_shared(dst+4096+off(key,byte));asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ka),"l"(KC+rr*128+byte));asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(va),"l"(VC+rr*128+byte));}}
 asm volatile("cp.async.commit_group;");};
#pragma unroll
 for(int i=0;i<ST-1;i++)issue(i);
 float o[4][4]={},mm[2]={-INFINITY,-INFINITY},ll[2]={0.f,0.f};
#pragma unroll 1
 for(int tile=0;tile<tiles;tile++){
  asm volatile("cp.async.wait_group %0;"::"n"(ST-2));__syncthreads();issue(tile+ST-1);

  const uint8_t* raw=mem+(tile%ST)*8192;float s[2][4]={};
#pragma unroll
  for(int nt=0;nt<2;nt++){
#pragma unroll
   for(int k=0;k<2;k++){uint32_t a[4];ld(a,raw+off(nt*16+(lane&7)+8*((lane>>3)&1),h*64+k*32+16*(lane>>4)));qmma(s[nt],a,qb[k][0],qb[k][1]);}
#pragma unroll
   for(int r=0;r<4;r++)s[nt][r]=(tile*32+nt*16+g+(r>=2?8:0)<n && begin+tile*32+nt*16+g+(r>=2?8:0)<=prefix+qt)?s[nt][r]*scale:-INFINITY;
  }
  uint8_t* vf=mem+ST*8192;
  if(tid<256){int key=tid>>3,byte=(tid&7)*16;uint4 u=*reinterpret_cast<const uint4*>(raw+4096+off(key,byte));
   uint4 a=make_uint4(cv2(u.x),cv2(u.x>>16),cv2(u.y),cv2(u.y>>16));
   uint4 b4=make_uint4(cv2(u.z),cv2(u.z>>16),cv2(u.w),cv2(u.w>>16));
   *reinterpret_cast<uint4*>(vf+voff(key,byte*2))=a;*reinterpret_cast<uint4*>(vf+voff(key,byte*2+16))=b4;
  }
  float mx[2]={fmaxf(fmaxf(s[0][0],s[0][2]),fmaxf(s[1][0],s[1][2])),fmaxf(fmaxf(s[0][1],s[0][3]),fmaxf(s[1][1],s[1][3]))};
#pragma unroll
  for(int c=0;c<2;c++){
#pragma unroll
   for(int x=4;x<=16;x*=2)mx[c]=fmaxf(mx[c],__shfl_xor_sync(0xffffffff,mx[c],x));
   float nm=fmaxf(mm[c],mx[c]),safe=nm==-INFINITY?0.f:nm,corr=exp2f(mm[c]-safe);ll[c]*=corr;mm[c]=nm;
#pragma unroll
   for(int d=0;d<4;d++){o[d][c]*=corr;o[d][c+2]*=corr;}
  }

  float p[2][4];
#pragma unroll
  for(int nt=0;nt<2;nt++){
#pragma unroll
   for(int r=0;r<4;r++){float safe=mm[r&1]==-INFINITY?0.f:mm[r&1];p[nt][r]=exp2f(s[nt][r]-safe);}
  }
#pragma unroll
  for(int c=0;c<2;c++){float sum=p[0][c]+p[0][c+2]+p[1][c]+p[1][c+2];
#pragma unroll
   for(int x=4;x<=16;x*=2)sum+=__shfl_xor_sync(0xffffffff,sum,x);ll[c]+=sum;}
  __syncthreads();
#pragma unroll
  for(int nt=0;nt<2;nt++){
   __half2 hp0=__floats2half2_rn(p[nt][0],p[nt][1]),hp8=__floats2half2_rn(p[nt][2],p[nt][3]);
   uint32_t p0=*reinterpret_cast<uint32_t*>(&hp0),p8=*reinterpret_cast<uint32_t*>(&hp8),z[2];
   int sa=8*t+(g>>1),sb=sa+4,sh=(g&1)*16;
   z[0]=((__shfl_sync(0xffffffff,p0,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p0,sb)>>sh)<<16);
   z[1]=((__shfl_sync(0xffffffff,p8,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p8,sb)>>sh)<<16);
#pragma unroll
   for(int pr=0;pr<4;pr++){uint32_t r[4];ldt(r,vf+voff(nt*16+(lane&7)+8*((lane>>3)&1),h*128+pr*32+16*(lane>>4)));
    uint32_t a[4]={r[0],r[2],r[1],r[3]};pmma(o[pr],a,z);
   }
  }
  __syncthreads();
 }
 asm volatile("cp.async.wait_group 0;");__syncthreads();

#pragma unroll
 for(int c=0;c<2;c++){
  int row=qt*8+2*t+c;long base=((long)(b*2+h)*splits+split)*64+row;
  float inv=ll[c]>0.f?1.f/ll[c]:0.f;
  if(g==0)LSE[base]=ll[c]>0.f?(mm[c]+log2f(ll[c]))*0.6931471805599453f:-INFINITY;
#pragma unroll
  for(int pr=0;pr<4;pr++){int d=pr*16+g;PART[base*64+d]=o[pr][c]*inv*vscale;PART[base*64+d+8]=o[pr][c+2]*inv*vscale;}
 }
}

void trans_verify_run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor table,torch::Tensor seq,
 torch::Tensor ks,torch::Tensor vs,torch::Tensor part,torch::Tensor lse,int64_t page,int64_t window,
 int64_t splits,int64_t chunk,double scale,int64_t scaled){
 int dyn=3*8192+8192+((chunk+127)/128)*128*4;cudaFuncSetAttribute(trans_verify<3>,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);
 trans_verify<3><<<dim3(splits,seq.numel()),512,dyn,at::cuda::getCurrentCUDAStream()>>>((const uint8_t*)q.data_ptr(),(const uint8_t*)k.data_ptr(),(const uint8_t*)v.data_ptr(),table.data_ptr<int>(),seq.data_ptr<int>(),ks.data_ptr<float>(),vs.data_ptr<float>(),part.data_ptr<float>(),lse.data_ptr<float>(),table.size(1),page,window,splits,chunk,scale,scaled);
 auto st=cudaGetLastError();TORCH_CHECK(st==cudaSuccess,cudaGetErrorString(st));}
