"""Specialize the actual M17..32 production algorithm by expert token count."""
from pathlib import Path
import runpy
m=runpy.run_path('dev/moe_bucket/probe.py');scale=runpy.run_path('dev/moe_scale/probe.py')
s=m['CUDA'];a=s.index('template<int MODE,int NCH>\n__global__');b=s.index('template<int MODE,int NCH>\nstatic void bucket_launch',a)
s=s[:a]+r'''
template<int MODE,int NCH>
__global__ void __launch_bounds__(128,(NCH==4?3:4))
wm_bucket(const uint4* W,const uint2* SF,const float* GS,const bf16* X,bf16* OUT,
 const int* lst,const int* act,const int* nact,int lmax){
 extern __shared__ uint4 smem[];
 pdl_wait();pdl_trigger();
 constexpr int tier=NCH==1?0:(NCH==2?1:2);
 constexpr int tiles=NCH==1?(MODE==0?256:128):(MODE==0?128:64);
 int count=__ldcg(nact+1+tier);
 for(int job=blockIdx.x;job<count*tiles;job+=gridDim.x){
   int slot=job/tiles,bx=job%tiles;
   int4 item=__ldcg(reinterpret_cast<const int4*>(act+512*(tier+1))+slot);
   if constexpr(NCH==1){
     wm_body<MODE,2,2,1,4,1,6,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,0,smem,bx);
   }else{
#pragma unroll 1
     for(int tb=0;tb<4;tb+=2)
       wm_body<MODE,4,2,NCH,4,1,6,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,tb,smem,bx);
   }
   __syncthreads();
 }
}
''' +s[b:]
s=s.replace('constexpr int dyn=45056+4096*NCH;', 'constexpr int dyn=24576;')
s=s.replace('dim3 grid=NCH==4?dim3(MODE==0?32:16,cap):dim3(cap);','dim3 grid(cap);')
Path('dev/moe_bucket/small.cu').write_text(s)
forward=m['FORWARD'].replace('        if tier==2:cap=min(cap,32+(m*TOP_K)//32)\n','')
direct=s.replace('wm_body<MODE,2,2,1,4,1,6,0>','wm_body<MODE,2,2,1,4,0,6,0>').replace('constexpr int dyn=24576;', 'constexpr int dyn=NCH==1?4096:24576;')
Path('dev/moe_bucket/small_probe.py').write_text('PARENT='+repr(scale['PARENT'])+'\nCPP='+repr(m['CPP'])+'\nCUDA='+repr(s)+'\nCUDA_DIRECT='+repr(direct)+'\nBASE_CPP='+repr(scale['CPP'])+'\nBASE_CUDA='+repr(scale['BASE'])+'\nFORWARD='+repr(forward)+'\n')
