"""Separate exact token-count tiers to avoid worst-case resource reservation."""
from pathlib import Path
import ast,runpy
p=Path('experiments/moe_wm_v22/submission.py').read_text();tree=ast.parse(p)
def value(name):return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
cu=value('_WM_CUDA');cpp=value('_WM_CPP')
route=cu.index('__global__ void route_kernel(')
end=cu.index('\n}\n\n__device__ __forceinline__ void cp16',route)
cu=cu[:end]+r'''
  // The extended work buffer is used only by the bucketed caller. Existing
  // callers retain their original 128-item allocation and one-element nact.
  if(nslots>=512){
    __syncthreads();
    if(threadIdx.x<32){
      int lane=threadIdx.x,total=*nact,carry[3]={0,0,0};
#pragma unroll
      for(int q=0;q<4;q++){
        int index=q*32+lane;
        int4 item=index<total?__ldcg(reinterpret_cast<const int4*>(act)+index):make_int4(0,0,0,0);
        int tier=item.z<=8?0:(item.z<=16?1:2);
#pragma unroll
        for(int b=0;b<3;b++){
          unsigned mask=__ballot_sync(0xffffffff,item.z>0&&tier==b);
          int rank=__popc(mask&((1u<<lane)-1));
          if(item.z>0&&tier==b)reinterpret_cast<int4*>(act+512*(b+1))[carry[b]+rank]=item;
          carry[b]+=__popc(mask);
        }
      }
      if(lane==0)for(int b=0;b<3;b++)nact[b+1]=carry[b];
    }
  }
''' +cu[end:]
idx=cu.index('extern "C" int wm_gemm_c(')
cu=cu[:idx]+r'''
template<int MODE,int NCH>
__global__ void __launch_bounds__(128,(NCH<=2?4:3))
wm_bucket(const uint4* W,const uint2* SF,const float* GS,const bf16* X,bf16* OUT,
 const int* lst,const int* act,const int* nact,int lmax){
 extern __shared__ uint4 smem[];
 pdl_wait();pdl_trigger();
 constexpr int tier=NCH==1?0:(NCH==2?1:2);
 constexpr int tiles=MODE==0?32:16;
 int count=__ldcg(nact+1+tier);
 if constexpr(NCH==4){
   int slot=blockIdx.y;if(slot>=count)return;
   int4 item=__ldcg(reinterpret_cast<const int4*>(act+512*(tier+1))+slot);
   wm_body_rs<MODE,4,NCH,4,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,smem,blockIdx.x);
 }else for(int job=blockIdx.x;job<count*tiles;job+=gridDim.x){
   int slot=job/tiles,bx=job%tiles;
   int4 item=__ldcg(reinterpret_cast<const int4*>(act+512*(tier+1))+slot);
   wm_body_rs<MODE,4,NCH,4,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,smem,bx);
   __syncthreads();
 }
}
template<int MODE,int NCH>
static void bucket_launch(const void* W,const void* SF,const float* GS,const void* X,void* OUT,
 const int* lst,const int* act,const int* nact,int lmax,int cap,cudaStream_t stream){
 constexpr int dyn=45056+4096*NCH;
 cudaFuncSetAttribute(wm_bucket<MODE,NCH>,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);
 cudaFuncSetAttribute(wm_bucket<MODE,NCH>,cudaFuncAttributePreferredSharedMemoryCarveout,100);
 dim3 grid=NCH==4?dim3(MODE==0?32:16,cap):dim3(cap);
 launch_k(wm_bucket<MODE,NCH>,grid,dim3(128),dyn,stream,g_pdl!=0,
 (const uint4*)W,(const uint2*)SF,GS,(const bf16*)X,(bf16*)OUT,lst,act,nact,lmax);
}
extern "C" int wm_bucket_c(int mode,int tier,int cap,const void* W,const void* SF,const float* GS,const void* X,void* OUT,
 const int* lst,const int* act,const int* nact,int lmax,cudaStream_t stream){
#define BKT(M,T,N) if(mode==M&&tier==T){bucket_launch<M,N>(W,SF,GS,X,OUT,lst,act,nact,lmax,cap,stream);return cudaGetLastError();}
 BKT(0,0,1) BKT(0,1,2) BKT(0,2,4) BKT(1,0,1) BKT(1,1,2) BKT(1,2,4)
#undef BKT
 return cudaErrorInvalidValue;
}
''' +cu[idx:]
cu+=r'''
extern "C" int wm_bucket_c(int,int,int,const void*,const void*,const float*,const void*,void*,const int*,const int*,const int*,int,cudaStream_t);
void wm_bucket_run(int64_t mode,int64_t tier,int64_t cap,torch::Tensor W,torch::Tensor SF,torch::Tensor GS,torch::Tensor X,torch::Tensor OUT,torch::Tensor lst,torch::Tensor act,torch::Tensor nact,int64_t lmax){
 int e=wm_bucket_c(mode,tier,cap,W.data_ptr(),SF.data_ptr(),GS.data_ptr<float>(),X.data_ptr(),OUT.data_ptr(),lst.data_ptr<int>(),act.data_ptr<int>(),nact.data_ptr<int>(),lmax,at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));
}
'''
decl='\nvoid wm_bucket_run(int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);\n'
cpp+=decl
forward=r'''
def _bucket_forward(data,state):
    x=data['hidden_states'].contiguous();m=x.shape[0];dev=x.device
    wm=state['wm'];mod=wm['mod'];lmax=(m+7)//8*8
    ids=data['topk_ids'].contiguous()
    cnt=torch.empty(32,dtype=torch.int32,device=dev)
    lst=torch.empty(32*lmax,dtype=torch.int32,device=dev)
    act=torch.empty(2048,dtype=torch.int32,device=dev)
    nact=torch.empty(4,dtype=torch.int32,device=dev)
    lids=torch.empty(m*TOP_K,dtype=torch.int32,device=dev)
    woff=torch.empty(128,dtype=torch.int32,device=dev)
    mod.wm_route(ids,_WM_NOLS,0,lmax,cnt,lst,act,nact,lids,woff,32)
    inter=torch.empty((m*TOP_K,4096),dtype=torch.bfloat16,device=dev)
    y=torch.empty_like(inter)
    main=torch.cuda.current_stream();parallel=state.get('_bucket_parallel',True)
    def run(tier):
        cap=state['_bucket_caps'][tier]
        if tier==2:cap=min(cap,32+(m*TOP_K)//32)
        mod.wm_bucket_run(0,tier,cap,wm['w13'],wm['s13'],wm['gs13'],x,inter,lst,act,nact,lmax)
        mod.wm_bucket_run(1,tier,cap,wm['w2'],wm['s2'],wm['gs2'],inter,y,lst,act,nact,lmax)
    if parallel:
        for tier in (2,1,0):
            stream=state['_bucket_streams'][tier];stream.wait_stream(main)
            with torch.cuda.stream(stream):run(tier)
        for stream in state['_bucket_streams']:main.wait_stream(stream)
    else:
        for tier in (2,1,0):run(tier)
    tw=data['topk_weights'].to(torch.float32).contiguous()
    return _finish_experts(y.view(m,TOP_K,4096),tw,norm_weight=state['norm_w'],eps=POST_NORM_EPS,weight_first=True,slot_expert_ids=lids.view(m,TOP_K))
'''
edits={}
for node in tree.body:
 if isinstance(node,ast.Assign):
  ns=[t.id for t in node.targets if isinstance(t,ast.Name)]
  if '_WM_CUDA' in ns:edits[node.lineno]=(node.end_lineno,'_WM_CUDA = '+repr(cu))
  if '_WM_CPP' in ns:edits[node.lineno]=(node.end_lineno,'_WM_CPP = '+repr(cpp))
 if isinstance(node,ast.FunctionDef) and node.name=='_wm_module':
  for call in ast.walk(node):
   if isinstance(call,ast.Call):
    for kw in call.keywords:
     if kw.arg=='functions' and isinstance(kw.value,ast.List):kw.value.elts.append(ast.Constant(value='wm_bucket_run'))
  edits[node.lineno]=(node.end_lineno,ast.unparse(node).replace('muse_wm_moe','muse_wm_bucket'))
lines=p.splitlines(keepends=True)
for start,(end,text) in sorted(edits.items(),reverse=True):lines[start-1:end]=[text+'\n']
standalone=''.join(lines)+'\n'+forward+r'''
_before_bucket_prepare=prepare
def prepare(weights):
    state=_before_bucket_prepare(weights)
    state['_bucket_streams']=[torch.cuda.Stream() for _ in range(3)]
    sms=torch.cuda.get_device_properties(weights['w13'].device).multi_processor_count
    state['_bucket_caps']=(sms*4,sms*2,sms*2)
    return state
_before_bucket_kernel_fn=kernel_fn
def kernel_fn(data,state):
    c=data['config'];m=data['hidden_states'].shape[0]
    if (128<=m<=256 and c['local_start']==0 and c['local_count']==32 and c['d_hidden']==4096 and c['d_expert']==4096):
        return _bucket_forward(data,state)
    return _before_bucket_kernel_fn(data,state)
'''
ast.parse(standalone)
Path('dev/ptx_campaign/moe_bucket_v1.py').write_text(standalone)
# Screen instantiates only selected bucket kernels; a winner must use the full
# standalone source above for official coverage, including original fallbacks.
a=cu.index('extern "C" int wm_gemm_c(');brace=cu.index('{',a);tail=cu.index('#include <torch/extension.h>',brace)
diagnostic=cu[:brace+1]+'return -1;\n}\n'+cu[tail:]
base=runpy.run_path('dev/moe_reuse/bundle.py')
Path('dev/moe_bucket/kernel.cu').write_text(diagnostic)
Path('dev/moe_bucket/probe.py').write_text('BASE_SUBMISSION='+repr(p)+'\nCPP='+repr(cpp)+'\nCUDA='+repr(diagnostic)+'\nBASE_CPP='+repr(base['CPP'])+'\nBASE_CUDA='+repr(base['CUDA_BASE'])+'\nFORWARD='+repr(forward)+'\n')
