"""Preserve BF16 norm weights; fuse small-token normalization with routing."""
from pathlib import Path
import ast

parent=Path('experiments/decoder_transpose_selected/submission.py').read_text()
tree=ast.parse(parent)
def val(name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
cu=val('_WM_CUDA');cpp=val('_WM_CPP')
a=cu.index('__global__ void route_kernel(')
b=cu.index('\n__device__ __forceinline__ void cp16',a)
route=cu[a:b].rstrip()
assert route.endswith('}')
fused=route.replace('__global__ void route_kernel(', '__global__ void __launch_bounds__(1024,1) norm_route_kernel(')
fused=fused.replace('int grp, int nslots) {','int grp, int nslots, const bf16* X, const bf16* W, bf16* Y, int M) {')
prefix=r'''
  __shared__ float partial[8][4];
  const int nr=threadIdx.x>>7,ni=threadIdx.x&127,wl=threadIdx.x&31,ww=(threadIdx.x&127)>>5;
  float sq=0.f;
  if(nr<M){
    for(int col=ni;col<4096;col+=128){float v=__bfloat162float(X[nr*4096+col]);sq=fmaf(v,v,sq);}
#pragma unroll
    for(int d=16;d>0;d>>=1)sq+=__shfl_down_sync(0xffffffff,sq,d);
    if(wl==0)partial[nr][ww]=sq;
  }
'''
assert fused.count('  pdl_trigger();')==1
fused=fused.replace('  pdl_trigger();','  pdl_trigger();\n'+prefix)
normalize=r'''
  if(nr<M){
    float sum=(partial[nr][0]+partial[nr][1])+(partial[nr][2]+partial[nr][3]);
    float inv=rsqrtf(sum*(1.f/4096.f)+1.e-5f);
    for(int col=ni;col<4096;col+=128){
      float v=__bfloat162float(X[nr*4096+col]);
      float w=__bfloat162float(W[col]);
      Y[nr*4096+col]=__float2bfloat16_rn((v*inv)*w);
    }
  }
'''
fused=fused.replace('  __syncthreads();','  __syncthreads();\n'+normalize,1)
launch=r'''
extern "C" int wm_norm_route_c(const int* ids,int npair,const int* ls,int lsv,int lmax,int* cnt,int* lst,int* act,int* nact,int* lids,int* woff,int grp,int nslots,const void* X,const void* W,void* Y,int M,cudaStream_t stream){
  if(M<1||M>8)return cudaErrorInvalidValue;
  launch_k(norm_route_kernel,dim3(1),dim3(1024),0,stream,g_pdl!=0,ids,npair,ls,lsv,lmax,cnt,lst,act,nact,lids,woff,grp,nslots,(const bf16*)X,(const bf16*)W,(bf16*)Y,M);
  return cudaGetLastError();
}
'''
binding=r'''
extern "C" int wm_norm_route_c(const int*,int,const int*,int,int,int*,int*,int*,int*,int*,int*,int,int,const void*,const void*,void*,int,cudaStream_t);
void wm_norm_route(torch::Tensor ids,torch::Tensor ls,int64_t lsv,int64_t lmax,torch::Tensor cnt,torch::Tensor lst,torch::Tensor act,torch::Tensor nact,torch::Tensor lids,torch::Tensor woff,int64_t grp,torch::Tensor X,torch::Tensor W,torch::Tensor Y){
 int e=wm_norm_route_c(ids.data_ptr<int>(),ids.numel(),ls.numel()?ls.data_ptr<int>():nullptr,lsv,lmax,cnt.data_ptr<int>(),lst.data_ptr<int>(),act.data_ptr<int>(),nact.data_ptr<int>(),lids.data_ptr<int>(),woff.data_ptr<int>(),grp,act.numel()/4,X.data_ptr(),W.data_ptr(),Y.data_ptr(),X.size(0),at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));
}
'''
declaration='void wm_norm_route(torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor,torch::Tensor,torch::Tensor);\n'
full=cu[:b]+'\n'+fused+'\n'+launch+cu[b:]+binding
thin=cu[:b]+'\n'+fused+'\n'+launch+r'''
extern "C" void wm_route_probe_c(const int* ids,int npair,int lmax,int* cnt,int* lst,int* act,int* nact,int* lids,int* woff,int grp,int nslots,cudaStream_t stream){
 launch_k(route_kernel,dim3(1),dim3(1024),0,stream,g_pdl!=0,ids,npair,(const int*)nullptr,0,lmax,cnt,lst,act,nact,lids,woff,grp,nslots);
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" void wm_set_pdl(int);
extern "C" void wm_route_probe_c(const int*,int,int,int*,int*,int*,int*,int*,int*,int,int,cudaStream_t);
void wm_set_pdl_t(int64_t v){wm_set_pdl(v);}
void wm_route(torch::Tensor ids,int64_t lmax,torch::Tensor cnt,torch::Tensor lst,torch::Tensor act,torch::Tensor nact,torch::Tensor lids,torch::Tensor woff,int64_t grp){
 wm_route_probe_c(ids.data_ptr<int>(),ids.numel(),lmax,cnt.data_ptr<int>(),lst.data_ptr<int>(),act.data_ptr<int>(),nact.data_ptr<int>(),lids.data_ptr<int>(),woff.data_ptr<int>(),grp,act.numel()/4,at::cuda::getCurrentCUDAStream());
}
'''+binding
norm=r'''
@triton.jit
def _pre_rms_kernel(X,W,Y,N:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,B)
    x=tl.load(X+row*N+c,c<N,0).to(tl.float32)
    w=tl.load(W+c,c<N,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/N+1.e-5)
    tl.store(Y+row*N+c,(x*inv)*w,c<N)
def _pre_rms(x,w,warps=4):
    y=torch.empty_like(x)
    _pre_rms_kernel[(x.shape[0],)](x,w,y,x.shape[1],triton.next_power_of_2(x.shape[1]),num_warps=warps,enable_fp_fusion=False)
    return y
'''
edits={}
for n in tree.body:
    if isinstance(n,ast.Assign):
        names=[t.id for t in n.targets if isinstance(t,ast.Name)]
        if '_WM_CUDA' in names:edits[n.lineno]=(n.end_lineno,'_WM_CUDA='+repr(full))
        if '_WM_CPP' in names:edits[n.lineno]=(n.end_lineno,'_WM_CPP='+repr(cpp+'\n'+declaration))
    if isinstance(n,ast.FunctionDef) and n.name=='_wm_module':
        for node in ast.walk(n):
            if isinstance(node,ast.Call):
                for kw in node.keywords:
                    if kw.arg=='functions' and isinstance(kw.value,ast.List):kw.value.elts.append(ast.Constant(value='wm_norm_route'))
        edits[n.lineno]=(n.end_lineno,ast.unparse(n).replace('muse_wm_moe','muse_wm_normroute'))
lines=parent.splitlines(keepends=True)
for start,(end,text) in sorted(edits.items(),reverse=True):lines[start-1:end]=[text+'\n']
standalone=''.join(lines)
old='_experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)'
assert standalone.count(old)==1
standalone=standalone.replace(old,'_experts_pre_normalized(expert_input, topk_weights, local_ids, state)')
n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_experts_wm')
forward=ast.get_source_segment(parent,n).replace('def _experts_wm(','def _experts_normroute(')
old='    mod.wm_route(ids, wm["nols"], 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)'
new='    xn=torch.empty_like(x)\n    mod.wm_norm_route(ids,wm["nols"],0,lmax,cnt,lst,act,nact,lids,woff,nch*8,x,state["pre_expert_eff"],xn)\n    x=xn'
assert forward.count(old)==1;forward=forward.replace(old,new)
standalone+='\n'+norm+'\n'+forward+r'''
_before_normroute_prepare=prepare
def prepare(weights):
    state=_before_normroute_prepare(weights)
    state['pre_expert_eff']=(weights['pre_expert_norm']+1).contiguous()
    return state
def _experts_pre_normalized(x,topk_weights,ids,state):
    if 1<=x.shape[0]<=8:
        return _experts_normroute(x,topk_weights,ids,state)
    return _experts(_pre_rms(x,state['pre_expert_eff']),topk_weights,ids,state)
'''
ast.parse(standalone)
Path('dev/ptx_campaign/decoder_normroute_v1.py').write_text(standalone)
Path('dev/decoder_norm/kernel.cu').write_text(thin)
Path('dev/decoder_norm/probe.py').write_text('CUDA='+repr(thin)+'\nNORM='+repr(norm)+'\n')
