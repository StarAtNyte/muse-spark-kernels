from pathlib import Path
s=Path('dev/fp8_probability/qreg.cu').read_text();s=s[:s.index('#define LAZY_TH')]
s+=r'''
template<int MODE> __global__ void probe(const uint8_t* A,const uint8_t* B,float* O){
 __shared__ __align__(1024) uint8_t sm[8192];
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,t=lane&3;
 A+=blockIdx.x*4096;B+=blockIdx.x*4096;O+=blockIdx.x*4096;
 for(int i=tid;i<256;i+=128){int row=i>>2,ch=i&3;
  *reinterpret_cast<uint4*>(sm+core_off(row,ch))=*reinterpret_cast<const uint4*>(A+row*64+ch*16);
  *reinterpret_cast<uint4*>(sm+4096+core_off(row,ch))=*reinterpret_cast<const uint4*>(B+row*64+ch*16);
 }
 uint32_t qa[2][4];
#pragma unroll
 for(int k=0;k<2;k++){
  int row=warp*16+g,col=k*32+t*4;
  qa[k][0]=*reinterpret_cast<const uint32_t*>(A+row*64+col);
  qa[k][1]=*reinterpret_cast<const uint32_t*>(A+(row+8)*64+col);
  qa[k][2]=*reinterpret_cast<const uint32_t*>(A+row*64+col+16);
  qa[k][3]=*reinterpret_cast<const uint32_t*>(A+(row+8)*64+col+16);
 }
 proxy_fence();__syncthreads();float c[32]={};wg_fence();
 uint64_t bd=mkdesc(sm+4096,128,512);
 if constexpr(MODE==0){wg_pv8(c,qa[0],bd);wg_pv8(c,qa[1],bd+16);}
 if constexpr(MODE==1){wg_qk_reg_first(c,qa[0],bd);wg_pv8(c,qa[1],bd+16);}
 if constexpr(MODE==2){wg_qk_first(c,mkdesc(sm,128,512),bd);wg_qk_acc(c,mkdesc(sm+256,128,512),bd+16);}
 wg_commit_wait();
#pragma unroll
 for(int j=0;j<8;j++){
  O[(warp*16+g)*64+j*8+2*t]=c[4*j];O[(warp*16+g)*64+j*8+2*t+1]=c[4*j+1];
  O[(warp*16+g+8)*64+j*8+2*t]=c[4*j+2];O[(warp*16+g+8)*64+j*8+2*t+1]=c[4*j+3];
 }
}
extern "C" void probe_launch(const void* a,const void* b,void* o,int n,int mode,cudaStream_t stream){
 if(mode==0)probe<0><<<n,128,0,stream>>>((const uint8_t*)a,(const uint8_t*)b,(float*)o);
 if(mode==1)probe<1><<<n,128,0,stream>>>((const uint8_t*)a,(const uint8_t*)b,(float*)o);
 if(mode==2)probe<2><<<n,128,0,stream>>>((const uint8_t*)a,(const uint8_t*)b,(float*)o);
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" void probe_launch(const void*,const void*,void*,int,int,cudaStream_t);
torch::Tensor qreg_probe(torch::Tensor a,torch::Tensor b,int64_t mode){auto o=torch::empty({a.size(0),64,64},a.options().dtype(torch::kFloat32));probe_launch(a.data_ptr(),b.data_ptr(),o.data_ptr(),a.size(0),mode,at::cuda::getCurrentCUDAStream());return o;}
'''
Path('dev/fp8_probability/qreg_probe.py').write_text('CPP='+repr('torch::Tensor qreg_probe(torch::Tensor,torch::Tensor,int64_t);')+'\nCUDA='+repr(s)+'\n')
