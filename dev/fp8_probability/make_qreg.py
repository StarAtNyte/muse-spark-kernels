"""Keep the invariant FP8 query operand in registers across all KV tiles."""
from pathlib import Path
import ast
p=Path('dev/ptx_campaign/attention_fp8p_joint_serial.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets))
s=ast.literal_eval(n.value)
a=s.index('__device__ __forceinline__ void wg_pv8(');b=s.index('\n}\n',a)+3
first=s[a:b].replace('wg_pv8(', 'wg_qk_reg_first(').replace('"+f"','"=f"').replace('%36,1,1,1;', '%36,0,1,1;')
s=s[:b]+first+s[b:]
a=s.index('  for (int c = tw; c < 64 * 4; c += 128)');b=s.index('  proxy_fence();',a)
s=s[:a]+r'''
  uint32_t qa[2][4];
  const int qr0=warp*16+g,qr1=qr0+8,qh=kh*8+g;
#pragma unroll
  for(int kk=0;kk<2;kk++){
    const int col=kk*32+t*4;
    const uint8_t* q0=Q+((long)(b*L+(qr0>>3))*HQ+qh)*64;
    const uint8_t* q1=Q+((long)(b*L+(qr1>>3))*HQ+qh)*64;
    qa[kk][0]=*reinterpret_cast<const uint32_t*>(q0+col);
    qa[kk][1]=*reinterpret_cast<const uint32_t*>(q1+col);
    qa[kk][2]=*reinterpret_cast<const uint32_t*>(q0+col+16);
    qa[kk][3]=*reinterpret_cast<const uint32_t*>(q1+col+16);
  }
''' +s[b:]
s=s.replace('wg_qk_first(sacc, qd0, kdesc);','wg_qk_reg_first(sacc, qa[0], kdesc);')
s=s.replace('wg_qk_acc(sacc, qd1, kdesc + 16);','wg_pv8(sacc, qa[1], kdesc + 16);')
lines=p.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['_WG_CUDA = '+repr(s)+'\n']
p=''.join(lines).replace('muse_wg5_fp8p_joint_serial','muse_wg5_fp8p_qreg')
ast.parse(p)
Path('dev/fp8_probability/qreg.cu').write_text(s)
Path('dev/ptx_campaign/attention_fp8p_qreg.py').write_text(p)
