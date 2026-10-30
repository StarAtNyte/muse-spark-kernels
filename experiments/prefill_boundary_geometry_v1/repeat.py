"""Paired screen of the exact prefill boundary body, varying launch geometry.

This isolates the boundary; full-layer correctness and timing remain mandatory.
"""
import importlib.util, json, statistics
import torch
from torch.utils.cpp_extension import load_inline

spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
source = s._CUDA_SRC

def part(start, end):
    assert source.count(start) == 1, start
    a = source.index(start)
    b = source.index(end, a)
    return source[a:b]

pdl = part('__device__ __forceinline__ void pdl_enter() {', '\n// Before lane 0')
g1 = part('DEVI float4 g1p4(', '\nDEVI void mbar_wait(')
body = part('#define BND_RC_MINB 2', '\n}  // namespace okc_moe')
body = body.replace('template <int TPB, bool GPOST, bool STREAMHINT, int MODE = 0>',
                    'template <int TPB, bool GPOST, bool STREAMHINT, int MODE = 0, int RC = 2>')
body = body.replace('MODE == 2 ? BND_RC_MINB :', 'MODE == 2 ? RC :')
cuda = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define DEVI __device__ __forceinline__
''' + pdl + '\n' + g1 + '\n' + body + r'''
template<int TPB, bool HINT, int RC>
void launch(int mode, torch::Tensor b, torch::Tensor st, torch::Tensor gp, torch::Tensor a, torch::Tensor be,
 torch::Tensor gn, torch::Tensor so, torch::Tensor no, torch::Tensor b1, torch::Tensor r, torch::Tensor a1, torch::Tensor be1) {
 auto stream = at::cuda::getCurrentCUDAStream(); int t = b.size(0);
 auto bp=(const __nv_bfloat16*)b.data_ptr(); auto sp=st.data_ptr<float>(); auto gpp=(const __nv_bfloat16*)gp.data_ptr();
 auto ap=a.data_ptr<float>(); auto bep=be.data_ptr<float>(); auto gnp=(const __nv_bfloat16*)gn.data_ptr();
 auto sop=so.data_ptr<float>(); auto nop=(__nv_bfloat16*)no.data_ptr();
 auto b1p=(const __nv_bfloat16*)b1.data_ptr(); auto rp=r.data_ptr<float>(); auto a1p=a1.data_ptr<float>(); auto be1p=be1.data_ptr<float>();
 if (mode == 1) boundary_kernel<TPB,false,HINT,1,RC><<<t,TPB,0,stream>>>(bp,sp,gpp,ap,bep,gnp,sop,nop,nullptr,0,b1p,rp,a1p,be1p);
 else boundary_kernel<TPB,true,HINT,2,RC><<<t,TPB,0,stream>>>(bp,sp,gpp,ap,bep,gnp,sop,nop,nullptr,0,b1p,rp,a1p,be1p);
 auto e=cudaGetLastError(); TORCH_CHECK(e==cudaSuccess,"boundary launch: ",cudaGetErrorString(e));
}
void run(int tpb, bool hint, int rc, int mode, torch::Tensor b, torch::Tensor st, torch::Tensor gp, torch::Tensor a,
 torch::Tensor be, torch::Tensor gn, torch::Tensor so, torch::Tensor no, torch::Tensor b1, torch::Tensor r, torch::Tensor a1, torch::Tensor be1) {
#define L(T,H,R) launch<T,H,R>(mode,b,st,gp,a,be,gn,so,no,b1,r,a1,be1)
 if (tpb==256 && !hint && rc==2) L(256,false,2);
 else if (tpb==256 && hint && rc==2) L(256,true,2);
 else if (tpb==512 && !hint && rc==2) L(512,false,2);
 else if (tpb==512 && hint && rc==2) L(512,true,2);
 else if (tpb==512 && !hint && rc==1) L(512,false,1);
 else if (tpb==1024 && !hint && rc==1) L(1024,false,1);
 else TORCH_CHECK(false,"unsupported config");
#undef L
}
'''
cpp = '''void run(int tpb, bool hint, int rc, int mode, torch::Tensor b, torch::Tensor st, torch::Tensor gp, torch::Tensor a, torch::Tensor be, torch::Tensor gn, torch::Tensor so, torch::Tensor no, torch::Tensor b1, torch::Tensor r, torch::Tensor a1, torch::Tensor be1);'''
mod = load_inline('prefill_boundary_geometry_01', cpp_sources=cpp, cuda_sources=cuda, functions=['run'],
                  extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '--use_fast_math'])
print('COMPILED', flush=True)
torch.manual_seed(1441)
flush=torch.zeros(256*1024*1024//4, device='cuda', dtype=torch.int32)
configs=[(512,False,2),(256,True,2)]

def cap(fn):
    fn(); torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g

def elapsed(g):
    flush.add_(1); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b)*1000

for seed,t in ((1441,16384),(2627,16384),(8675,16384)):
    torch.manual_seed(seed)
    shape=(t,8192)
    b=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    st=torch.randn(shape,device='cuda',dtype=torch.float32)
    b1=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    gp=torch.randn((8192,),device='cuda',dtype=torch.bfloat16)*0.1
    gn=torch.randn((8192,),device='cuda',dtype=torch.bfloat16)*0.1
    a=torch.randn((8192,),device='cuda',dtype=torch.float32)
    be=torch.randn((8192,),device='cuda',dtype=torch.float32)
    a1=torch.randn((8192,),device='cuda',dtype=torch.float32)
    be1=torch.randn((8192,),device='cuda',dtype=torch.float32)
    r=torch.rand((t,),device='cuda',dtype=torch.float32)+0.1
    for mode in (1,):
        so0=torch.empty(shape,device='cuda',dtype=torch.float32); no0=torch.empty(shape,device='cuda',dtype=torch.bfloat16); r0=r.clone()
        def func(cfg,so,no,rr):
            return lambda: mod.run(*cfg,mode,b,st,gp,a,be,gn,so,no,b1,rr,a1,be1)
        base=cap(func(configs[0],so0,no0,r0))
        for cfg in configs[1:]:
            so=torch.empty_like(so0); no=torch.empty_like(no0); rr=r.clone()
            try:
                g=cap(func(cfg,so,no,rr))
                for x,y in ((no,no0),(rr,r0)) if mode==1 else ((so,so0),(no,no0)):
                    # Compare sample rows including both ends. Full reference validation is a later gate.
                    idx=torch.tensor([0,t//2,t-1],device='cuda')
                    torch.testing.assert_close(x[idx].float(),y[idx].float(),rtol=0.02,atol=0.03)
                tb=[];tc=[]
                for rep in range(12):
                    if rep&1: tc.append(elapsed(g));tb.append(elapsed(base))
                    else: tb.append(elapsed(base));tc.append(elapsed(g))
                print('ROW',json.dumps({'seed':seed,'tokens':t,'mode':mode,'config':cfg,'base_us':statistics.median(tb),
                                        'candidate_us':statistics.median(tc),'gain':statistics.median(tb)/statistics.median(tc)}),flush=True)
            except Exception as e:
                print('FAIL',json.dumps({'tokens':t,'mode':mode,'config':cfg,'error':str(e)[:300]}),flush=True)
                raise
            del g,so,no,rr
        del base,so0,no0,r0
    del b,st,b1,gp,gn,a,be,a1,be1,r
print('DONE',flush=True)
