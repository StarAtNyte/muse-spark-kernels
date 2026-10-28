"""Fuse the measured FP8 attention primitive with an eight-block DSM epilogue.

Q/K preparation remains an ordinary preceding launch. No cross-block cache-write
ordering assumption is introduced. The merge preserves the BF16 attention
rounding used by the production path before RMS normalization.
"""
from pathlib import Path
import ast

parent=Path('dev/ptx_campaign/attention_fp8p_joint_serial.py').read_text()
tree=ast.parse(parent)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets))
s=ast.literal_eval(node.value)
s='#include <cooperative_groups.h>\n'+s
s=s.replace('__global__ void __launch_bounds__(128, 4)','__global__ void __cluster_dims__(8,1,1) __launch_bounds__(128, 4)')
s=s.replace('int scaled, int dbg) {','int scaled, int dbg, const __nv_bfloat16* GATE, __nv_bfloat16* OUT, int stride) {',1)
a=s.index('  cp_wait<0>();',s.index('float o[32]'))
b=s.index('\n#endif\n}',a)
s=s[:a]+r'''
  cp_wait<0>();
  __syncthreads();
  float* shared=reinterpret_cast<float*>(smem);
#pragma unroll
  for(int h=0;h<2;h++){
    int row=warp*16+g+8*h;
    float ls=ls_run[h];ls+=__shfl_xor_sync(0xffffffff,ls,1);ls+=__shfl_xor_sync(0xffffffff,ls,2);
    float inv=ls>0.f?1.f/ls:1.f;
#pragma unroll
    for(int j=0;j<8;j++){
      shared[row*64+j*8+2*t]=o[4*j+2*h]*inv*value_scale;
      shared[row*64+j*8+2*t+1]=o[4*j+2*h+1]*inv*value_scale;
    }
    if(t==0)shared[4096+row]=ls>0.f?m_run[h]+log2f(ls)-8.f:-INFINITY;
  }
  auto cluster=cooperative_groups::this_cluster();
  cluster.sync();
  int row=split*8+warp*2+(lane>>4),d=(lane&15)*4;
  float high=-INFINITY;
#pragma unroll
  for(int r=0;r<8;r++)high=fmaxf(high,cluster.map_shared_rank(shared,r)[4096+row]);
  float den=0.f,x[4]={0.f,0.f,0.f,0.f};
#pragma unroll
  for(int r=0;r<8;r++){
    float* remote=cluster.map_shared_rank(shared,r);float mx=remote[4096+row];
    float w=mx==-INFINITY?0.f:exp2f(mx-high);den+=w;
    float4 v=*reinterpret_cast<float4*>(remote+row*64+d);
    x[0]=fmaf(v.x,w,x[0]);x[1]=fmaf(v.y,w,x[1]);x[2]=fmaf(v.z,w,x[2]);x[3]=fmaf(v.w,w,x[3]);
  }
  float square=0.f;
#pragma unroll
  for(int j=0;j<4;j++){x[j]=__bfloat162float(__float2bfloat16_rn(x[j]/(den>0.f?den:1.f)));square+=x[j]*x[j];}
#pragma unroll
  for(int k=1;k<=8;k*=2)square+=__shfl_xor_sync(0xffffffff,square,k);
  float inv=rsqrtf(square/64.f+1.e-5f);
  int token=row>>3,qh=kh*8+(row&7);
  long go=(long)(b*L+token)*stride+(HQ+4+qh)*64+d;
  long oo=((long)(b*L+token)*HQ+qh)*64+d;
#pragma unroll
  for(int j=0;j<4;j++)OUT[oo+j]=__float2bfloat16_rn(x[j]*inv/(1.f+expf(-__bfloat162float(GATE[go+j]))));
  cluster.sync();
''' +s[b:]
a=s.index('extern "C" int wg5_run(')
s=s[:a]+r'''
extern "C" int cluster_launch(const void* q,const void* kc,const void* vc,const int* table,const int* seq,const float* ks,const float* vs,const void* gate,void* out,int B,int HQ,int tn,int page,int window,int chunk,float scale,int scaled,int stride,cudaStream_t stream){
 int dyn=1024+4096+4096+WS*8192+((chunk+255)/256)*256*4;
 cudaFuncSetAttribute(wg5_partials,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);
 wg5_partials<<<dim3(8,2,B),128,dyn,stream>>>((const uint8_t*)q,(const uint8_t*)kc,(const uint8_t*)vc,table,seq,ks,vs,nullptr,nullptr,8,HQ,tn,page,window,8,chunk,scale,scaled,0,(const __nv_bfloat16*)gate,(__nv_bfloat16*)out,stride);
 return cudaGetLastError();
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int cluster_launch(const void*,const void*,const void*,const int*,const int*,const float*,const float*,const void*,void*,int,int,int,int,int,int,float,int,int,cudaStream_t);
torch::Tensor cluster_attention(torch::Tensor q,torch::Tensor kc,torch::Tensor vc,torch::Tensor table,torch::Tensor seq,torch::Tensor ks,torch::Tensor vs,torch::Tensor gate,int64_t page,int64_t window,int64_t chunk,double scale,int64_t scaled){
 auto out=torch::empty({q.size(0),q.size(1)*64},gate.options());
 int e=cluster_launch(q.data_ptr(),kc.data_ptr(),vc.data_ptr(),table.data_ptr<int>(),seq.data_ptr<int>(),ks.data_ptr<float>(),vs.data_ptr<float>(),gate.data_ptr(),out.data_ptr(),seq.numel(),q.size(1),table.size(1),page,window,chunk,scale,scaled,gate.stride(0),at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));return out;
}
'''
cpp='torch::Tensor cluster_attention(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,double,int64_t);'
Path('dev/cluster_attention/fused.cu').write_text(s)
Path('dev/cluster_attention/fused.py').write_text('CPP='+repr(cpp)+'\nCUDA='+repr(s)+'\n')
