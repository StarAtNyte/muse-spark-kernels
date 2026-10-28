"""Hopper warpgroup BF16 MoE with 8/16/32 token-column instructions."""
from pathlib import Path
import ast,runpy
m=runpy.run_path('dev/moe_scale/probe.py');parent=m['PARENT'];tree=ast.parse(parent)
cu=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WM_CUDA' for t in n.targets)))
helpers=''
for width in (8,16):
    regs=width//2
    operands=','.join('%'+str(i) for i in range(regs))
    aa=','.join('%'+str(regs+i) for i in range(4))
    outs=','.join('"+f"(d['+str(i)+'])' for i in range(regs))
    helpers+=f'''template <> struct WgN<{width}> {{
 static __device__ __forceinline__ void mma(float* d,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint64_t db){{
  asm volatile("wgmma.mma_async.sync.aligned.m64n{width}k16.f32.bf16.bf16 {{{operands}}}, {{{aa}}}, %{regs+4}, 1, 1, 1, 0;\\n"
   :{outs}:"r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(db));
 }}
}};
'''
cu=cu.replace('template <int N> struct WgN;','template <int N> struct WgN;\n'+helpers)
a=cu.index('__device__ __forceinline__ void wm_body_wg(');b=cu.index('template <int MODE, int NMAX, int S>',a)
body=cu[a:b].replace('NCH / 2','(NCH + 1) / 2').replace('xok[u] = col < ntok;', 'xok[u] = col < ntok && id < N * 8;')
body=body.replace('        cp16z(xring +', '        if(id < N * 8)cp16z(xring +')
cu=cu[:a]+body+cu[b:]
a=cu.index('  if (ntok <= 32) wm_body_wg',cu.index('wm_gemm_wg('))
cu=cu[:a]+'''  if(ntok<=8)wm_body_wg<MODE,1,S>(Wq,Sq,GS,X,OUT,l,e,ntok,smem_wg,blockIdx.x);
  else if(NMAX>=2&&ntok<=16)wm_body_wg<MODE,2,S>(Wq,Sq,GS,X,OUT,l,e,ntok,smem_wg,blockIdx.x);
  else '''+cu[a:]
cu=cu.replace('if (ntok <= 32) wm_body_wg','if (NMAX >= 4 && ntok <= 32) wm_body_wg')
a=cu.index('  const int pdl = g_pdl;',cu.index('extern "C" int wm_gemm_c('));b=cu.index('\n#include <torch/extension.h>',a)
dispatch='  const int pdl=g_pdl;\n'
for mode in (0,1):
    for n in (1,2,4):
        for stages in (3,4,6):
            dispatch+=f'if(mode=={mode}&&ring==7&&nmax=={n}&&nt=={stages}&&dbg==0){{launch_wg<{mode},{n},{stages}>(W,S,GS,X,OUT,lst,act,maxslots,lmax,pdl,st);return 0;}}\n'
dispatch+='return 1;\n}\n'
cu=cu[:a]+dispatch+cu[b:]
out={'PARENT':parent,'CPP':m['CPP'],'CUDA':cu,'BASE_CUDA':m['BASE']}
Path('dev/moe_wgnarrow/probe.py').write_text('\n'.join(k+'='+repr(v) for k,v in out.items())+'\n')
Path('dev/moe_wgnarrow/kernel.cu').write_text(cu)
