"""Double keys per WGMMA tile while retaining independent KV-head CTAs."""
from pathlib import Path
import ast
s=Path('experiments/attention_verify_onehead_s2/submission.py').read_text()
a=s.index('_WG_CUDA =');b=s.index('_WG_MOD =',a)
x=s[a:b]
x=x.replace('#define WT 64', '#define WT 128').replace('__launch_bounds__(128, 4)', '__launch_bounds__(128, 2)')
x=x.replace('smem + 4096 + 9216','smem + 4096 + 18432').replace('stages + WS * 8192','stages + WS * 16384')
x=x.replace('if (tw < 64) asm volatile', 'asm volatile')
x=x.replace('dst_off[4]', 'dst_off[8]').replace('key4[4]', 'key4[8]').replace('i < 4; i++', 'i < 8; i++')
x=x.replace('(4096 + key * 64', '(8192 + key * 64').replace('stage * 8192', 'stage * 16384')
x=x.replace('const int vkey = (tw & 31) + 32 * (tw >> 6);', 'const int vkey = tw;')
x=x.replace('const int vcb = ((tw >> 5) & 1) * 2;', 'const int vcb = 0;')
x=x.replace('vsrc_off[2]', 'vsrc_off[4]').replace('j < 2; j++', 'j < 4; j++')
x=x.replace('rd_stage * 16384 + 4096', 'rd_stage * 16384 + 8192').replace('(rd_stage * 512)', '(rd_stage * 1024)')
x=x.replace('float sacc[32];', 'float sacc[64] = {};')
x=x.replace('wg_qk_first(sacc, qd0, kdesc);','wg_qk128(sacc, qd0, kdesc, 0);').replace('wg_qk_acc(sacc, qd1, kdesc + 16);','wg_qk128(sacc, qd1, kdesc + 16, 1);')
x=x.replace('h2 < ((dbg & 16) ? 0 : 2)', 'h2 < ((dbg & 16) ? 0 : 4)')
x=x.replace('i < 32; i++) asm volatile("" : "+f"(sacc[i])', 'i < 64; i++) asm volatile("" : "+f"(sacc[i])')
x=x.replace('j < 8; j++)\n#pragma unroll\n        for (int r', 'j < 16; j++)\n#pragma unroll\n        for (int r')
x=x.replace('j < ((dbg & 64) ? 0 : 8)', 'j < ((dbg & 64) ? 0 : 16)')
x=x.replace('uint32_t pa[8][2];', 'uint32_t pa[16][2];').replace('for (int j = 0; j < 8; j++) {\n      const float p0', 'for (int j = 0; j < 16; j++) {\n      const float p0')
x=x.replace('kk < 4; kk++', 'kk < 8; kk++')
x=x.replace('1024 + 4096 + 9216 + (size_t)WS * 8192', '1024 + 4096 + 18432 + (size_t)WS * 16384')
s=s[:a]+x+s[b:]
a=s.index('def _wg_forward(');b=s.index('_FD_CUDA =',a)
x=s[a:b].replace('triton.cdiv(triton.cdiv(span, splits), 64) * 64','triton.cdiv(triton.cdiv(span, splits), 128) * 128')
s=s[:a]+x+s[b:]
s=s.replace('"muse_wg5_onehead_s2"','"muse_wg5_onehead_tile128"')
ast.parse(s)
Path('dev/ptx_campaign/attention_verify_tile128.py').write_text(s)
