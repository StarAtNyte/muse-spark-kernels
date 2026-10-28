from pathlib import Path
import ast
p=Path('dev/ptx_campaign/attention_fp8p_qreg.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets))
s=ast.literal_eval(n.value)
marker='    asm volatile("bar.sync %0, 128;\\n" ::"r"(1 + wgi) : "memory");'
assert s.count(marker)==1
s=s.replace(marker,marker+r'''
    if(dbg&1024){
      cp_wait<0>();__syncthreads();
#pragma unroll
      for(int j=0;j<8;j++){
        PART[(base_row+warp*16+g)*64+j*8+2*t]=sacc[4*j];
        PART[(base_row+warp*16+g)*64+j*8+2*t+1]=sacc[4*j+1];
        PART[(base_row+warp*16+g+8)*64+j*8+2*t]=sacc[4*j+2];
        PART[(base_row+warp*16+g+8)*64+j*8+2*t+1]=sacc[4*j+3];
      }
      return;
    }
''')
lines=p.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['_WG_CUDA = '+repr(s)+'\n']
Path('dev/ptx_campaign/attention_fp8p_qreg_debug.py').write_text(''.join(lines))
