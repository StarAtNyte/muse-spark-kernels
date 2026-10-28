"""Choose warp split-K GEMV for sparse experts, row-split for dense experts."""
from pathlib import Path
import runpy
m=runpy.run_path('dev/moe_bucket/probe.py');s=m['CUDA']
s=s.replace('__launch_bounds__(128,(NCH<=2?4:3))','__launch_bounds__((NCH==2?256:128),(NCH==2?2:(NCH==1?4:3)))')
s=s.replace('constexpr int tiles=MODE==0?32:16;','constexpr int tiles=NCH==4?(MODE==0?32:16):(MODE==0?256:128);')
old='wm_body_rs<MODE,4,NCH,4,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,smem,bx);'
# There are two occurrences, but the dense branch remains row-split.
needle='''   '''+old+'''
   __syncthreads();'''
replacement='''   if constexpr(NCH==1)wm_body<MODE,2,2,1,4,0,6,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,0,smem,bx);
   else wm_body<MODE,2,2,2,8,0,6,0>(W,SF,GS,X,OUT,lst+item.x*lmax+item.y,item.x,item.z,0,smem,bx);
   __syncthreads();'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
s=s.replace('constexpr int dyn=45056+4096*NCH;','constexpr int dyn=NCH==4?61440:(NCH==1?4096:16384);')
s=s.replace('launch_k(wm_bucket<MODE,NCH>,grid,dim3(128),dyn,stream,g_pdl!=0,','launch_k(wm_bucket<MODE,NCH>,grid,dim3(NCH==2?256:128),dyn,stream,g_pdl!=0,')
ca=s.replace('__ldcg(xv)', '__ldg(xv)').replace('__ldcg(xv + 1)', '__ldg(xv + 1)')
assert ca!=s
Path('dev/moe_bucket/native.cu').write_text(s)
Path('dev/moe_bucket/native_probe.py').write_text(Path('dev/moe_bucket/probe.py').read_text()+'\nCUDA_NATIVE='+repr(s)+'\nCUDA_NATIVE_CA='+repr(ca)+'\n')
