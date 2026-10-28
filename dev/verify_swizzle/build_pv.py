"""Apply canonical MN-major 64B swizzle to the FP16 value operand."""
import ast
from pathlib import Path
s=Path('dev/ptx_campaign/attention_verify_qksw64.py').read_text();tree=ast.parse(s)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));cuda=ast.literal_eval(node.value)
replacements={
'return (key >> 3) * 1152 + (d >> 3) * 128 + (key & 7) * 16 + (d & 7) * 2;':'int a=(key>>3)*1536+(d>>5)*512+(key&7)*64+(d&31)*2;return a ^ ((a>>3)&0x30);',
'smem + 4096 + 9216':'smem + 4096 + 12288',
'vt + (tw >> 3) * 1152 + 1024 + (tw & 7) * 16':'vt + vmn_off(tw,64)',
'const uint64_t vd0 = mkdesc(vt, 1152, 128);':'const uint64_t vd0 = mkdesc(vt,512,1536)|(2ull<<62);',
'(kk * 144)':'(kk * 192)',
'1024 + 4096 + 9216':'1024 + 4096 + 12288',
}
for old,new in replacements.items():assert cuda.count(old)==1,(old,cuda.count(old));cuda=cuda.replace(old,new)
lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['_WG_CUDA = '+repr(cuda)+'\n'];s=''.join(lines)
s=s.replace('load_inline("muse_wg5_qk_sw64",','load_inline("muse_wg5_qkpv_sw64",')
Path('dev/verify_swizzle/qkpv_sw64.cu').write_text(cuda)
Path('dev/ptx_campaign/attention_verify_qkpvsw64.py').write_text(s)
ast.parse(s)
