"""Use NVIDIA's canonical 64B K-major WGMMA shared layout for Q/K."""
import ast
from pathlib import Path
s=Path('experiments/attention_transpose_selected/submission.py').read_text();tree=ast.parse(s)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));cuda=ast.literal_eval(node.value)
replacements={
'return (r >> 3) * 512 + chunk * 128 + (r & 7) * 16;':'int a=r*64+chunk*16;return a ^ ((a>>3)&0x30);',
'const uint64_t qd0 = mkdesc(qs, 128, 512), qd1 = mkdesc(qs + 256, 128, 512);':'const uint64_t qd0 = mkdesc(qs,16,512)|(2ull<<62), qd1 = mkdesc(qs+32,16,512)|(2ull<<62);',
'const uint64_t kd0 = mkdesc(stages, 128, 512);':'const uint64_t kd0 = mkdesc(stages,16,512)|(2ull<<62);',
'wg_qk_acc(sacc, qd1, kdesc + 16);':'wg_qk_acc(sacc, qd1, kdesc + 2);',
}
for old,new in replacements.items():assert cuda.count(old)==1,old;cuda=cuda.replace(old,new)
lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['_WG_CUDA = '+repr(cuda)+'\n'];s=''.join(lines)
s=s.replace('load_inline("muse_wg5_onehead_s2",','load_inline("muse_wg5_qk_sw64",')
Path('dev/verify_swizzle/qk_sw64.cu').write_text(cuda)
Path('dev/ptx_campaign/attention_verify_qksw64.py').write_text(s)
ast.parse(s)
