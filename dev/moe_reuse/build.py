"""Reuse one FP4 weight stream for up to 64 tokens of a busy expert.

The existing NCH8 row-split kernel preloads a large activation register array.
For NCH8, stream just the required shared-memory uint2 fragment into the MMA.
NCH<=4 specializations retain their old code. Every routed token is computed.
"""
from pathlib import Path
import ast
p=Path('kernels/muse_moe_w4a16/submission.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WM_CUDA' for t in n.targets));s=ast.literal_eval(n.value)
a=s.index('__device__ __forceinline__ void wm_body_rs');b=s.index('template <int MODE, int NTB, int NMAX',a)
original_body=s[a:b]
body=original_body
before='''    uint4 xr[NCH][2];
#pragma unroll
    for (int h = 0; h < NCH; h++) {
      xr[h][0] = xring[(i % S) * XSLOTS + (h * 2) * 32 + lane];
      xr[h][1] = xring[(i % S) * XSLOTS + (h * 2 + 1) * 32 + lane];
    }'''
after='''    uint4 xr[NCH][2];
    if constexpr(NCH<=4) {
#pragma unroll
      for(int h=0;h<NCH;h++){
        xr[h][0]=xring[(i%S)*XSLOTS+(h*2)*32+lane];
        xr[h][1]=xring[(i%S)*XSLOTS+(h*2+1)*32+lane];
      }
    }'''
assert body.count(before)==1;body=body.replace(before,after)
before='''          const uint4 xs = xr[h][ks >> 1];
          if constexpr (DBG & 2) sink ^= a0 ^ a1 ^ a2 ^ a3 ^ ((ks & 1) ? xs.z : xs.x) ^ ((ks & 1) ? xs.w : xs.y);
          else mma16816(acc[n][h], a0, a1, a2, a3, (ks & 1) ? xs.z : xs.x, (ks & 1) ? xs.w : xs.y);'''
after='''          if constexpr(NCH>=8) { if(h*8>=ntok)continue; }
          uint32_t b0,b1;
          if constexpr(NCH<=4){
            const uint4 xs=xr[h][ks>>1];b0=(ks&1)?xs.z:xs.x;b1=(ks&1)?xs.w:xs.y;
          }else{
            const uint4* ptr=&xring[(i%S)*XSLOTS+(h*2+(ks>>1))*32+lane];
            const uint2 xs=reinterpret_cast<const uint2*>(ptr)[ks&1];b0=xs.x;b1=xs.y;
          }
          if constexpr(DBG&2)sink^=a0^a1^a2^a3^b0^b1;
          else mma16816(acc[n][h],a0,a1,a2,a3,b0,b1);'''
assert body.count(before)==1;body=body.replace(before,after)
updated=s[:a]+body+s[b:]
def emit(cuda,name):
 lines=p.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['_WM_CUDA = '+repr(cuda)+'\n'];out=''.join(lines)
 out=out.replace('nch = 1 if m <= 8 else 2 if m <= 16 else 4','nch = 1 if m <= 8 else 2 if m <= 16 else 4 if m < 128 else 8')
 out=out.replace('    nt1 = _c[3] if len(_c) > 3 else nt','    if m >= 128: nt,nw,ring=2,4,9\n    nt1 = _c[3] if len(_c) > 3 else nt')
 out=out.replace('"muse_wm_moe"','"muse_wm_'+name+'"');ast.parse(out)
 Path('dev/ptx_campaign/'+name+'.py').write_text(out)
emit(updated,'moe_reuse64_stream')
emit(s,'moe_reuse64_control')
Path('dev/moe_reuse/stream.cu').write_text(updated)

# Paired-K activation streaming: load a full shared vector once and reuse it
# for both K16 MMAs and all weight tiles. Accumulators retain their K order.
start=original_body.index('    uint4 xr[NCH][2];')
end=original_body.index('\n  }\n  cp_wait<0>();',start)
inner=original_body[start:end]
pair=r'''
    if constexpr(NCH<=4) {
'''+inner+r'''
    }else{
#pragma unroll
      for(int kp=0;kp<2;kp++){
        uint32_t a[NTB][2][4];
#pragma unroll
        for(int n=0;n<NTB;n++){
          uint4 wv=wring[(((i%S)*4+warp)*NTB+n)*32+lane];
          uint2 sc=sring[((((i>>2)%3)*4+warp)*NTB+n)*32+lane];
          uint32_t s0=scale2(sc.x,cq),s1=scale2(sc.y,cq);
#pragma unroll
          for(int j=0;j<2;j++){
            uint32_t w=kp==0?(j==0?wv.x:wv.y):(j==0?wv.z:wv.w);
            a[n][j][0]=hmul(dec<0>(w),s0);a[n][j][1]=hmul(dec<1>(w),s1);
            a[n][j][2]=hmul(dec<2>(w),s0);a[n][j][3]=hmul(dec<3>(w),s1);
          }
        }
#pragma unroll
        for(int h=0;h<NCH;h++){
          if(h*8>=ntok)continue;
          uint4 xs=xring[(i%S)*XSLOTS+(h*2+kp)*32+lane];
#pragma unroll
          for(int n=0;n<NTB;n++){
            mma16816(acc[n][h],a[n][0][0],a[n][0][1],a[n][0][2],a[n][0][3],xs.x,xs.y);
            mma16816(acc[n][h],a[n][1][0],a[n][1][1],a[n][1][2],a[n][1][3],xs.z,xs.w);
          }
        }
      }
    }
'''
tiled=s[:a]+original_body[:start]+pair+original_body[end:]+s[b:]
bound='__launch_bounds__(128, (S <= 3 ? 4 : 3))'
newbound='__launch_bounds__(128, (NMAX>=8 ? 3 : (S<=3 ? 4 : 3)))'
def prune(cuda,nmax=8,ntb=2,stages=3,ring=9):
    # Diagnostic device dispatch: instantiate only the exact tested kernels.
    # Full standalone sources are also emitted above/below for later validation.
    f=cuda.index('extern "C" int wm_gemm_c(');brace=cuda.index('{',f)
    tail=cuda.index('#include <torch/extension.h>',brace)
    dispatch=r'''
      if(nmax!=_N_ || nt!=_T_ || ring!=_R_ || dbg!=0)return -1;
      if(mode==0)launch_rs<0,_T_,_N_,_S_,0>(W,S,GS,X,OUT,lst,act,maxslots,lmax,g_pdl,st);
      else launch_rs<1,_T_,_N_,_S_,0>(W,S,GS,X,OUT,lst,act,maxslots,lmax,g_pdl,st);
      return 0;
    }
    '''
    for key,val in [('_N_',nmax),('_T_',ntb),('_S_',stages),('_R_',ring)]:dispatch=dispatch.replace(key,str(val))
    return cuda[:brace+1]+dispatch+cuda[tail:]
emit(updated.replace(bound,newbound),'moe_reuse64_stream_b3')
emit(tiled.replace(bound,newbound),'moe_reuse64_paired_b3')
bundle='CPP='+repr(ast.literal_eval(next(v.value for v in ast.parse(p).body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WM_CPP' for t in v.targets))))+'\n'
bundle+='CUDA_BASE='+repr(prune(s,4,4,4,4))+'\n'
for name,cu in [('control',s),('stream',updated.replace(bound,newbound)),('paired',tiled.replace(bound,newbound))]:
    diagnostic=prune(cu)
    Path('dev/moe_reuse/'+name+'_probe.cu').write_text(diagnostic)
    bundle+='CUDA_'+name.upper()+'='+repr(diagnostic)+'\n'
Path('dev/moe_reuse/bundle.py').write_text(bundle)
