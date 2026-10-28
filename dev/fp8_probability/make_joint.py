"""Joint P/V key permutation eliminates all transpose/operand warp shuffles.

Logical k in a 16-key block is mapped to physical
(k&1) | ((k&8)>>2) | ((k&6)<<1). P and V use the same permutation.
"""
from pathlib import Path
import runpy
p=runpy.run_path('dev/fp8_probability/probe_pv_vector.py');s=p['CUDA']
a=s.index('  uint32_t z[4];');b=s.index('  int matrix=lane>>3',a)
s=s[:a]+'''  uint32_t z[4]={__byte_perm(r[0],r[1],0x6420),__byte_perm(r[0],r[1],0x7531),
                 __byte_perm(r[2],r[3],0x6420),__byte_perm(r[2],r[3],0x7531)};
'''+s[b:]
a=s.index('   uint32_t z0=v[0]');b=s.index('\n  }\n }',a)
s=s[:a]+'''   pa[kk][h]=v[0]|(v[1]<<16);pa[kk][h+2]=v[2]|(v[3]<<16);'''+s[b:]
Path('dev/fp8_probability/probe_pv_joint.py').write_text('CPP='+repr(p['CPP'])+'\nCUDA='+repr(s)+'\n')
# Integrate into the native attention candidate without changing its math or masks.
parent=Path('dev/ptx_campaign/attention_fp8p_native_v1.py').read_text()
import ast
node=next(n for n in ast.parse(parent).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));cuda=ast.literal_eval(node.value)
a=cuda.index('    // Exact byte transpose');b=cuda.index('    proxy_fence();',a)
cuda=cuda[:a]+'''    // Key-permuted FP8 transpose: P uses the identical within-block order.
#pragma unroll
    for(int pr=0;pr<2;pr++){
      int key0=warp*16+(lane&7)+8*((lane>>3)&1),byte=pr*32+16*(lane>>4);
      uint32_t r[4];pv8_ldt(r,vst+key0*64+(((byte>>4)^(key0&3))<<4));
      uint32_t z[4]={__byte_perm(r[0],r[1],0x6420),__byte_perm(r[0],r[1],0x7531),
                     __byte_perm(r[2],r[3],0x6420),__byte_perm(r[2],r[3],0x7531)};
      int matrix=lane>>3,d=pr*32+(matrix>>1)*16+2*(lane&7)+(matrix&1);
      uint32_t ad=__cvta_generic_to_shared(vt+core_off(d,warp));
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(ad),"r"(z[0]),"r"(z[1]),"r"(z[2]),"r"(z[3]):"memory");
    }
'''+cuda[b:]
a=cuda.index('      int sa=g*4+');b=cuda.index('\n    wg_fence();wg_pv8',a)
cuda=cuda[:a]+'''#pragma unroll
      for(int h=0;h<2;h++){
        pp[kk][h]=pa[kk*4][h]|(pa[kk*4+1][h]<<16);
        pp[kk][h+2]=pa[kk*4+2][h]|(pa[kk*4+3][h]<<16);
      }
    }'''+cuda[b:]
lines=parent.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['_WG_CUDA = '+repr(cuda)+'\n'];out=''.join(lines).replace('muse_wg5_fp8p_native_v1','muse_wg5_fp8p_joint_v2')
ast.parse(out);Path('dev/ptx_campaign/attention_fp8p_joint_v2.py').write_text(out)
Path('dev/fp8_probability/joint_kernel.cu').write_text(cuda)
