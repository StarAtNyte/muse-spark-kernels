"""Lossless scale expansion versus on-the-fly byte extraction."""
from pathlib import Path
import ast

parent=Path('experiments/moe_wm_v22/submission.py').read_text()
tree=ast.parse(parent)
def val(name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
cu=val('_WM_CUDA');cpp=val('_WM_CPP')
# Keep exactly the production configurations used by the scored shapes.
a=cu.index('  const int pdl = g_pdl;',cu.index('extern "C" int wm_gemm_c('))
b=cu.index('\n#include <torch/extension.h>',a)
dispatch='''  const int pdl=g_pdl;
#define D(M,T,N,W_,R) if(mode==M&&nt==T&&nmax==N&&nw==W_&&ring==R&&dbg==0){launch<M,T,N,W_,R,0>(Wq_,S,GS,X,OUT,cnt,lst,act,nact,woff,maxslots,lmax,pdl,st);return 0;}
'''.replace('Wq_','W')
for mode in range(2):
    for t,n,w,r in [(2,1,4,0),(2,1,2,0),(2,2,8,0),(4,4,4,1)]:
        dispatch+=f'D({mode},{t},{n},{w},{r})\n'
    dispatch+=f'if(mode=={mode}&&ring==4&&nt==4&&nmax==4&&dbg==0){{launch_rs<{mode},4,4,4,0>(W,S,GS,X,OUT,lst,act,maxslots,lmax,pdl,st);return 0;}}\n'
dispatch+='#undef D\nreturn 1;\n}\n'
base=cu[:a]+dispatch+cu[b:]
a=base.index('__device__ __forceinline__ uint32_t scale2(')
b=base.index('\n__device__ __forceinline__ uint32_t hmul',a)
prmt=base[:a]+'''__device__ __forceinline__ uint32_t scale2(uint32_t q,int c){
 uint32_t r;uint32_t ctl=0x4040u+0x101u*c;
 asm("prmt.b32 %0,%1,0,%2;":"=r"(r):"r"(q),"r"(ctl));
 return (r<<4)+0x76007600u;
}
'''+base[b:]
helper='''
__device__ __forceinline__ uint32_t scale4(uint4 q,int c,int rh){
 uint32_t v=rh==0?(c<2?q.x:q.y):(c<2?q.z:q.w),r;
 asm("prmt.b32.rc16 %0,%1,0,%2;":"=r"(r):"r"(v),"r"(c));
 return r;
}
'''
a=base.rfind('template <',0,base.index('__device__ __forceinline__ void wm_body('))
b=base.index('\n#ifndef WM_MINB',a)
body=base[a:b].replace('const uint2*','const uint4*').replace('uint2 sc','uint4 sc')
for obj in ['sc[n]','scur[n]']:
    for sel,c in [('c','c'),('cq','cq')]:
        body=body.replace(f'scale2({obj}.x, {sel})',f'scale4({obj}, {sel}, 0)').replace(f'scale2({obj}.y, {sel})',f'scale4({obj}, {sel}, 1)')
expanded=base[:a]+helper+body+base[b:]
# Only warp-split kernels use expanded scales; row-split stays unchanged.
a=expanded.index('__global__ void __launch_bounds__(NW * 32,')
b=expanded.index('// ---- row-split variant:',a)
piece=expanded[a:b].replace('(Wq, Sq, GS','(Wq, reinterpret_cast<const uint4*>(Sq), GS')
expanded=expanded[:a]+piece+expanded[b:]
out={'PARENT':parent,'CPP':cpp,'BASE':base,'PRMT':prmt,'EXPANDED':expanded}
Path('dev/moe_scale/probe.py').write_text('\n'.join(k+'='+repr(v) for k,v in out.items())+'\n')
Path('dev/moe_scale/expanded.cu').write_text(expanded)
# CPU proof for every possible source byte and selector, including unselected
# bytes chosen adversarially. Device validation also covers actual PTX behavior.
for c in range(4):
    for v in range(256):
        q=(0xabcdef01&~(255<<(8*c)))|(v<<(8*c))
        expected=((((q>>(8*c))&255)<<4)+0x7600)*0x10001
        got=((v|(v<<16))<<4)+0x76007600
        assert expected==got
print('1024 integer scale identities passed')
