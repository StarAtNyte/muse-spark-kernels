"""Keep old row tiling for <=32-token experts; widen only busy expert items."""
from pathlib import Path
import runpy
v=runpy.run_path('dev/moe_reuse/build.py')
s=v['tiled'].replace(v['bound'],v['newbound'])
entry=s.index('extern "C" int wm_gemm_c(')
kernel=r'''
template<int MODE> __global__ void __launch_bounds__(128,3)
wm_hybrid(const uint4* W,const uint2* SF,const float* GS,const bf16* X,bf16* OUT,
 const int* lst,const int* act,int lmax){
 extern __shared__ uint4 smem[];
 pdl_wait();pdl_trigger();
 int4 item=__ldcg(reinterpret_cast<const int4*>(act)+blockIdx.y);
 int ntok=item.z;if(ntok==0)return;
 int e=item.x;const int* l=lst+e*lmax+item.y;int bx=blockIdx.x;
 if(ntok<=32){
   if(bx>=(MODE==0?32:16))return;
   if(ntok<=8)wm_body_rs<MODE,4,1,4,0>(W,SF,GS,X,OUT,l,e,ntok,smem,bx);
   else if(ntok<=16)wm_body_rs<MODE,4,2,4,0>(W,SF,GS,X,OUT,l,e,ntok,smem,bx);
   else wm_body_rs<MODE,4,4,4,0>(W,SF,GS,X,OUT,l,e,ntok,smem,bx);
 }else wm_body_rs<MODE,2,8,3,0>(W,SF,GS,X,OUT,l,e,ntok,smem,bx);
}
template<int MODE> static void launch_hybrid(const void* W,const void* SF,const float* GS,const void* X,void* OUT,
 const int* lst,const int* act,int slots,int lmax,cudaStream_t stream){
 constexpr int dyn=61440;
 cudaFuncSetAttribute(wm_hybrid<MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);
 launch_k(wm_hybrid<MODE>,dim3(MODE==0?64:32,slots),dim3(128),dyn,stream,g_pdl!=0,
 (const uint4*)W,(const uint2*)SF,GS,(const bf16*)X,(bf16*)OUT,lst,act,lmax);
}
'''
s=s[:entry]+kernel+s[entry:]
a=s.index('extern "C" int wm_gemm_c(');brace=s.index('{',a);tail=s.index('#include <torch/extension.h>',brace)
s=s[:brace+1]+r'''
 if(nmax!=8 || nt!=2 || ring!=9 || dbg!=0)return -1;
 if(mode==0)launch_hybrid<0>(W,S,GS,X,OUT,lst,act,maxslots,lmax,st);
 else launch_hybrid<1>(W,S,GS,X,OUT,lst,act,maxslots,lmax,st);
 return 0;
}
'''+s[tail:]
Path('dev/moe_reuse/hybrid.cu').write_text(s)
bundle=Path('dev/moe_reuse/bundle.py').read_text()
bundle+='CUDA_HYBRID='+repr(s)+'\n'
Path('dev/moe_reuse/hybrid_bundle.py').write_text(bundle)
