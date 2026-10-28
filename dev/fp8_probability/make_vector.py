"""Replace byte-scalar transpose stores with an exact warp matrix-store transpose."""
import runpy
from pathlib import Path
p=runpy.run_path('dev/fp8_probability/probe_pv.py');s=p['CUDA']
a=s.index('#pragma unroll\n for(int pr=0;pr<2;pr++)');b=s.index(' asm volatile("fence.proxy.async',a)
block='''#pragma unroll
 for(int pr=0;pr<2;pr++){
  uint32_t r[4];ldt(r,mem+rawoff(warp*16+(lane&7)+8*((lane>>3)&1),pr*32+16*(lane>>4)));
  uint32_t z[4];int sa=g*4+2*(t&1),sb=sa+1;
#pragma unroll
  for(int pair=0;pair<2;pair++){
   uint32_t a0=__shfl_sync(0xffffffff,r[pair*2],sa),b0=__shfl_sync(0xffffffff,r[pair*2],sb);
   uint32_t a1=__shfl_sync(0xffffffff,r[pair*2+1],sa),b1=__shfl_sync(0xffffffff,r[pair*2+1],sb);
   uint32_t aa=t>=2?a1:a0,bb=t>=2?b1:b0;
   z[pair*2]=__byte_perm(aa,bb,0x6420);z[pair*2+1]=__byte_perm(aa,bb,0x7531);
  }
  int matrix=lane>>3,d=pr*32+(matrix>>1)*16+2*(lane&7)+(matrix&1);
  uint32_t ad=__cvta_generic_to_shared(mem+4096+core(d,warp));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(ad),"r"(z[0]),"r"(z[1]),"r"(z[2]),"r"(z[3]):"memory");
 }
'''
s=s[:a]+block+s[b:]
Path('dev/fp8_probability/probe_pv_vector.py').write_text('CPP='+repr(p['CPP'])+'\nCUDA='+repr(s)+'\n')
Path('dev/fp8_probability/vector_transpose.cuh').write_text(block)
