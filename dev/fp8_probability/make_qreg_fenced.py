"""Make invariant register operands explicit compiler dependencies at waits.

NVIDIA CuTe uses empty read/write asm fences for WGMMA register operands:
https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp
"""
from pathlib import Path
import ast
p=Path('dev/ptx_campaign/attention_fp8p_qreg.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets))
s=ast.literal_eval(n.value)
fence='''
#pragma unroll
    for(int k=0;k<2;k++)
#pragma unroll
      for(int r=0;r<4;r++)asm volatile("" : "+r"(qa[k][r]) :: "memory");
'''
s=s.replace('    wg_fence();\n    wg_qk_reg_first',fence+'    wg_fence();\n    wg_qk_reg_first')
s=s.replace('    wg_wait0();','    wg_wait0();'+fence)
lines=p.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['_WG_CUDA = '+repr(s)+'\n']
p=''.join(lines).replace('muse_wg5_fp8p_qreg','muse_wg5_fp8p_qreg_fenced')
Path('dev/ptx_campaign/attention_fp8p_qreg_fenced.py').write_text(p)
Path('dev/fp8_probability/qreg_fenced.cu').write_text(s)
