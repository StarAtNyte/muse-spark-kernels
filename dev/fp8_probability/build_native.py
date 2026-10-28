"""Own FP8-P/V paged verification kernel with FP32 unquantized softmax sums."""
import ast,runpy
from pathlib import Path
parent=Path('experiments/attention_transpose_selected/submission.py').read_text();tree=ast.parse(parent)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));s=ast.literal_eval(node.value)
probe=runpy.run_path('dev/fp8_probability/probe_pv.py')['CUDA']
helpers=probe[probe.index('__device__ __forceinline__ void ldt'):probe.index('__global__ void probe')]
# Keep only ldmatrix and MMA helpers; use the parent's descriptor encoder.
a=helpers.index('__device__ __forceinline__ uint64_t desc');b=helpers.index('__device__ __forceinline__ void pv(',a)
helpers=helpers[:a]+helpers[b:];helpers=helpers.replace('void ldt(', 'void pv8_ldt(').replace('void pv(', 'void wg_pv8(')
helpers+='''
__device__ __forceinline__ uint32_t packp8(float a,float b){uint16_t y;
 asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(y):"f"(a),"f"(b));return y;}
'''
s=s.replace('#define WS 2',helpers+'\n#define WS 2')
s=s.replace('#define LAZY_TH 8.0f','#define LAZY_TH 0.0f')
a=s.index('  {\n    const uint32_t ad =');b=s.index('  proxy_fence();',a);s=s[:a]+s[b:]
s=s.replace('smem + 4096 + 9216','smem + 4096 + 4096').replace('1024 + 4096 + 9216','1024 + 4096 + 4096')
s=s.replace('const uint64_t vd0 = mkdesc(vt, 1152, 128);','const uint64_t vd0 = mkdesc(vt,128,512);')
s=s.replace('float o[36];','float o[32];\n  float ls_run[2]={0.f,0.f};').replace('i < 36','i < 32').replace('j < 9','j < 8')
a=s.index('    // V raw [key][d]');b=s.index('    proxy_fence();',a)
s=s[:a]+'''    // Exact byte transpose V[key,d] into the FP8 K-major WGMMA layout.
#pragma unroll
    for(int pr=0;pr<2;pr++){
      int key0=warp*16+(lane&7)+8*((lane>>3)&1),byte=pr*32+16*(lane>>4);
      uint32_t rr[4];pv8_ldt(rr,vst+key0*64+(((byte>>4)^(key0&3))<<4));
#pragma unroll
      for(int j=0;j<4;j++){
        int d=pr*32+2*(g+(j>=2?8:0)),key=warp*16+(j&1)*8+2*t;
        *reinterpret_cast<uint16_t*>(vt+core_off(d,key>>4)+(key&15))=__byte_perm(rr[j],0,0x4420);
        *reinterpret_cast<uint16_t*>(vt+core_off(d+1,key>>4)+(key&15))=__byte_perm(rr[j],0,0x4431);
      }
    }
'''+s[b:]
s=s.replace('m_run[0] = n0; m_run[1] = n1;', 'm_run[0] = n0; m_run[1] = n1;ls_run[0]*=c0;ls_run[1]*=c1;')
s=s.replace('sc2, -sf0)', 'sc2, 8.f-sf0)').replace('sc2, -sf1)', 'sc2, 8.f-sf1)')
s=s.replace('      pa[j][0] = pack_h2(p0, p1);','      ls_run[0]+=p0+p1;ls_run[1]+=p2+p3;\n      pa[j][0] = packp8(p0,p1);').replace('pa[j][1] = pack_h2(p2, p3);','pa[j][1] = packp8(p2,p3);')
a=s.index('    wg_fence();\n    const uint64_t vd = vd0;');b=s.index('    wg_commit_wait();',a)
s=s[:a]+'''    uint32_t pp[2][4];
#pragma unroll
    for(int kk=0;kk<2;kk++){
      int sa=g*4+2*(t&1),sb=sa+1,sh=(t>>1)*16;
#pragma unroll
      for(int h=0;h<2;h++){
        uint32_t z0=pa[kk*4][h]|(pa[kk*4+1][h]<<16),z1=pa[kk*4+2][h]|(pa[kk*4+3][h]<<16);
        pp[kk][h]=((__shfl_sync(0xffffffff,z0,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,z0,sb)>>sh)<<16);
        pp[kk][h+2]=((__shfl_sync(0xffffffff,z1,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,z1,sb)>>sh)<<16);
      }
    }
    wg_fence();wg_pv8(o,pp[0],vd0);wg_pv8(o,pp[1],vd0+16);
'''+s[b:]
s=s.replace('const float ls = o[32 + 2 * h];','float ls=ls_run[h];ls+=__shfl_xor_sync(0xffffffff,ls,1);ls+=__shfl_xor_sync(0xffffffff,ls,2);')
s=s.replace('(m_run[h] + log2f(ls))','(m_run[h] + log2f(ls) - 8.f)')
# Replace only the large/full verify implementation. All existing decode/small paths remain available.
lines=parent.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['_WG_CUDA = '+repr(s)+'\n'];out=''.join(lines).replace('load_inline("muse_wg5_onehead_s2",','load_inline("muse_wg5_fp8p_native_v1",')
ast.parse(out);Path('dev/ptx_campaign/attention_fp8p_native_v1.py').write_text(out)
Path('dev/fp8_probability/native_kernel.cu').write_text(s)
