"""Packed-head sparse tensor-core decode. Every key remains in dense attention."""
CPP = '''void sparse_decode_run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor table,torch::Tensor seq,
 torch::Tensor ks,torch::Tensor vs,torch::Tensor part,torch::Tensor lse,int64_t page,int64_t window,
 int64_t splits,int64_t chunk,double scale,int64_t scaled);'''
CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t ph(float a,float b) {__half2 h=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&h);}
__device__ __forceinline__ uint32_t cv2(uint32_t x) {
 uint32_t y;asm("{.reg .b16 a,b; mov.b32 {a,b},%1; cvt.rn.f16x2.e4m3x2 %0,a;}" :"=r"(y):"r"(x));return y;
}
__device__ __forceinline__ void spq(float* c,const uint32_t* a,const uint32_t* b,uint32_t e) {
 asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12,0;"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]),"r"(e));
}
__device__ __forceinline__ void spv(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,const uint32_t* b) {
 asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12,0;"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]),"r"(0xEEEE4444u));
}
__device__ __forceinline__ void ldt(uint32_t* r,const void* p) {
 uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
 :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
}
__device__ __forceinline__ int ko(int key,int byte) {return key*128+(((byte>>4)^(key&7))<<4)+(byte&15);}
template<int S>
__global__ void __launch_bounds__(128,3) sparse_decode(const uint8_t* Q,const uint8_t* KC,const uint8_t* VC,
 const int* TABLE,const int* SEQ,const float* KS,const float* VS,float* PART,float* LSE,
 int table_n,int page,int window,int splits,int chunk,float sm_scale,int scaled) {
 extern __shared__ __align__(16) uint8_t mem[];
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,t=lane&3,b=blockIdx.y,split=blockIdx.x;
 const int prefix=SEQ[b],first=window?max(0,prefix-window+1):0,begin=first+split*chunk;
 const int end=min(begin+chunk,prefix+1),n=max(0,end-begin),tiles=max(0,((n+31)/32-warp+3)/4);
 int* rows=reinterpret_cast<int*>(mem+S*4*8192);
 const int shift=__ffs(page)-1,mask=page-1;
 for(int i=tid;i<n;i+=128) {
  int tok=begin+i,pg=tok>>shift,phys=pg<table_n?TABLE[b*table_n+pg]:-1;
  rows[i]=phys<0?0:(phys<<shift)+(tok&mask);
 }
 const float sc2=sm_scale*(scaled?KS[0]:1.f)*1.4426950408889634f;
 const float vscale=scaled?VS[0]:1.f;
 uint32_t qa[2][4];
#pragma unroll
 for(int k=0;k<2;k++) {
  const uint8_t* q=Q+(long)b*1024+k*32+t*4;
  qa[k][0]=*reinterpret_cast<const uint32_t*>(q+g*64);
  qa[k][1]=*reinterpret_cast<const uint32_t*>(q+(g+8)*64);
  qa[k][2]=*reinterpret_cast<const uint32_t*>(q+g*64+16);
  qa[k][3]=*reinterpret_cast<const uint32_t*>(q+(g+8)*64+16);
 }
 __syncthreads();
 auto issue=[&](int s) {
  if(s<tiles) {
   uint8_t* dst=mem+((s%S)*4+warp)*8192;
#pragma unroll
   for(int i=lane;i<512;i+=32) {
    int key=i>>4,c=i&7,iv=(i>>3)&1,li=s*128+warp*32+key;
    long rr=li<n?rows[li]:0;
    int vr=(key>>1)*4+2*(c>>2)+(key&1);
    int off=iv?4096+vr*64+(((c&3)^(vr&3))*16):ko(key,c*16);
    const uint8_t* src=(iv?VC:KC)+rr*128+c*16;
    uint32_t ad=(uint32_t)__cvta_generic_to_shared(dst+off);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ad),"l"(src));
   }
  }
  asm volatile("cp.async.commit_group;");
 };
#pragma unroll
 for(int s=0;s<S-1;s++)issue(s);
 float o[8][4]={},mm[2]={-INFINITY,-INFINITY},ll[2]={0.f,0.f};
 const uint32_t em=(t&1)?0xEEEEEEEEu:0x44444444u;
#pragma unroll 1
 for(int s=0;s<tiles;s++) {
  asm volatile("cp.async.wait_group %0;"::"n"(S-2));__syncwarp();issue(s+S-1);
  const uint8_t* raw=mem+((s%S)*4+warp)*8192;
  const uint8_t* vr=raw+4096;
  int tb=begin+s*128+warp*32;
  float sv[4][4]={};
#pragma unroll
  for(int nt=0;nt<4;nt++) {
#pragma unroll
   for(int kb=0;kb<2;kb++) {
    uint32_t bv[4];
#pragma unroll
    for(int j=0;j<4;j++) {
     int d=kb*32+8*j+2*t,key=nt*8+g;
     uint16_t lo=*reinterpret_cast<const uint16_t*>(raw+ko(key,d));
     uint16_t hi=*reinterpret_cast<const uint16_t*>(raw+ko(key,64+d));
     bv[j]=uint32_t(lo)|(uint32_t(hi)<<16);
    }
    spq(sv[nt],qa[kb],bv,em);
   }
  }
  float mx[2]={-INFINITY,-INFINITY};
#pragma unroll
  for(int nt=0;nt<4;nt++) {
#pragma unroll
   for(int r=0;r<4;r++) {
    if(tb+nt*8+2*t+(r&1)>=end) sv[nt][r]=-INFINITY;
    mx[r>>1]=fmaxf(mx[r>>1],sv[nt][r]);
   }
  }
#pragma unroll
  for(int h=0;h<2;h++) {
   mx[h]=fmaxf(mx[h],__shfl_xor_sync(0xffffffffu,mx[h],1));
   mx[h]=fmaxf(mx[h],__shfl_xor_sync(0xffffffffu,mx[h],2));
   float nm=fmaxf(mm[h],mx[h]*sc2),safe=nm==-INFINITY?0.f:nm,corr=exp2f(mm[h]-safe);
   ll[h]*=corr;mm[h]=nm;
#pragma unroll
   for(int dp=0;dp<8;dp++){o[dp][2*h]*=corr;o[dp][2*h+1]*=corr;}
  }
  uint32_t pa[4][2];float sums[2]={0.f,0.f};
#pragma unroll
  for(int nt=0;nt<4;nt++) {
#pragma unroll
   for(int h=0;h<2;h++) {
    float safe=mm[h]==-INFINITY?0.f:mm[h];
    float p0=exp2f(fmaf(sv[nt][2*h],sc2,-safe)),p1=exp2f(fmaf(sv[nt][2*h+1],sc2,-safe));
    sums[h]+=p0+p1;pa[nt][h]=ph(p0,p1);
   }
  }
#pragma unroll
  for(int h=0;h<2;h++) {
   sums[h]+=__shfl_xor_sync(0xffffffffu,sums[h],1);
   sums[h]+=__shfl_xor_sync(0xffffffffu,sums[h],2);ll[h]+=sums[h];
  }
#pragma unroll
  for(int kk=0;kk<2;kk++) {
#pragma unroll
   for(int pr=0;pr<4;pr++) {
    int row=kk*32+(lane&7)+8*(lane>>3);
    uint32_t r[4],be[4],bo[4];ldt(r,vr+row*64+((pr^(row&3))*16));
#pragma unroll
    for(int j=0;j<4;j++) {
     be[j]=cv2(__byte_perm(r[j],0,0x4420));bo[j]=cv2(__byte_perm(r[j],0,0x4431));
    }
    spv(o[2*pr],pa[2*kk][0],pa[2*kk][1],pa[2*kk+1][0],pa[2*kk+1][1],be);
    spv(o[2*pr+1],pa[2*kk][0],pa[2*kk][1],pa[2*kk+1][0],pa[2*kk+1][1],bo);
   }
  }
  __syncwarp();
 }
 asm volatile("cp.async.wait_group 0;");__syncthreads();
 float* smm=reinterpret_cast<float*>(mem);float* sml=smm+64;float* smo=sml+64;
#pragma unroll
 for(int h=0;h<2;h++) {
  int row=g+8*h;
  if(t==0){smm[warp*16+row]=mm[h];sml[warp*16+row]=ll[h];}
#pragma unroll
  for(int pr=0;pr<4;pr++) {
#pragma unroll
   for(int c=0;c<2;c++) {
    int d=16*pr+4*t+2*c;
    smo[(warp*16+row)*64+d]=o[2*pr][2*h+c];
    smo[(warp*16+row)*64+d+1]=o[2*pr+1][2*h+c];
   }
  }
 }
 __syncthreads();
 if(tid<16) {
  float high=-INFINITY,ls=0.f;
#pragma unroll
  for(int w=0;w<4;w++)high=fmaxf(high,smm[w*16+tid]);
#pragma unroll
  for(int w=0;w<4;w++) {
   float m=smm[w*16+tid],f=m==-INFINITY?0.f:exp2f(m-high);
   ls+=sml[w*16+tid]*f;smm[w*16+tid]=f;
  }
  sml[tid]=ls>0.f?1.f/ls:0.f;
  long row=((long)(b*2+(tid>>3))*splits+split)*16+(tid&7);
  LSE[row]=ls>0.f?(high+log2f(ls))*0.6931471805599453f:-INFINITY;
 }
 __syncthreads();
 for(int i=tid;i<16*64;i+=128) {
  int row=i>>6,d=i&63;float sum=0.f;
#pragma unroll
  for(int w=0;w<4;w++)sum=fmaf(smo[w*16*64+i],smm[w*16+row],sum);
  long dst=(((long)(b*2+(row>>3))*splits+split)*16+(row&7))*64+d;
  PART[dst]=sum*sml[row]*vscale;
 }
}
void sparse_decode_run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor table,torch::Tensor seq,
 torch::Tensor ks,torch::Tensor vs,torch::Tensor part,torch::Tensor lse,int64_t page,int64_t window,
 int64_t splits,int64_t chunk,double scale,int64_t scaled) {
 int dyn=2*4*8192+((chunk+127)/128)*128*4;
 cudaFuncSetAttribute(sparse_decode<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);
 cudaFuncSetAttribute(sparse_decode<2>,cudaFuncAttributePreferredSharedMemoryCarveout,100);
 sparse_decode<2><<<dim3(splits,seq.numel()),128,dyn,at::cuda::getCurrentCUDAStream()>>>(
  static_cast<const uint8_t*>(q.data_ptr()),static_cast<const uint8_t*>(k.data_ptr()),static_cast<const uint8_t*>(v.data_ptr()),
  table.data_ptr<int>(),seq.data_ptr<int>(),ks.data_ptr<float>(),vs.data_ptr<float>(),part.data_ptr<float>(),lse.data_ptr<float>(),
  table.size(1),page,window,splits,chunk,scale,scaled);
 auto st=cudaGetLastError();TORCH_CHECK(st==cudaSuccess,cudaGetErrorString(st));
}
'''
