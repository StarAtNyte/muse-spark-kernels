"""Stack baseline-matched FP8 P/V with MIT bigboateng's warp-specialized attention."""
from pathlib import Path
import ast
s=Path('dev/public_attention/operatorstack_v1.py').read_text()
a=s.index('@gluon.jit\ndef _gv_softmax_pv(');b=s.index('\n\ndef _counter(',a)
part=s[a:b]
assert 'p = gl.exp2(sc - mu[:, None])' in part
part=part.replace('p = gl.exp2(sc - mu[:, None])','p = gl.exp2(sc - mu[:, None] + 8.0)')
part=part.replace('gl.DotOperandLayout(0, mma_o, 2)','gl.DotOperandLayout(0, mma_o, 4)').replace('p.to(gl.float16)','p.to(gl.float8e4nv)')
part=part.replace('instr_shape=[16, D, 16]','instr_shape=[16, D, 32]')
part=part.replace('gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)','gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2, transposed=True)')
part=part.replace('v16_s.store(vt.to(gl.float16))','v16_s.store(vt)')
part=part.replace('gl.allocate_shared_memory(gl.float16, [BN, D], v_sl)','gl.allocate_shared_memory(gl.float8e4nv, [BN, D], v_sl)')
part=part.replace('gl.allocate_shared_memory(gl.float16, [NP, D], v_sl, v8.to(gl.float16))','gl.allocate_shared_memory(gl.float8e4nv, [NP, D], v_sl, v8)')
part=part.replace('NP: gl.constexpr = 16','NP: gl.constexpr = 32')
part=part.replace('    v_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2, transposed=True)',
                  '    v_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2, transposed=True)\n    vnew_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=8, rank=2, transposed=True)')
part=part.replace('gl.allocate_shared_memory(gl.float8e4nv, [NP, D], v_sl, v8)', 'gl.allocate_shared_memory(gl.float8e4nv, [NP, D], vnew_sl, v8)')
s=s[:a]+part+s[b:]
s+='\n# FP8 probability/value adaptation: exponent offset8 and FP32 softmax sums.\n# Original warp-specialized pipeline by team bigboateng, MIT notice above.\n'
ast.parse(s);Path('dev/gluon_fp8p/candidate.py').write_text(s)
