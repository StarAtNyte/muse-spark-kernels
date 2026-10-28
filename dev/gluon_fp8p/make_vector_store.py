"""Target measured shared-store bank conflicts in the transposed FP8 V tile."""
from pathlib import Path
s=Path('dev/gluon_fp8p/candidate.py').read_text()
a=s.index('@gluon.jit\ndef _gv_consumer(');b=s.index('@gluon.jit\ndef _g_full_verify(',a)
part=s[a:b].replace('    mma_s: gl.constexpr =','    vstore: gl.constexpr = gl.BlockedLayout([16, 1], [8, 4], [1, 4], [0, 1])\n    mma_s: gl.constexpr =').replace('        v16_s.store(vt)','        v16_s.store(gl.convert_layout(vt, vstore))')
s=s[:a]+part+s[b:]
s=s.replace('v_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2, transposed=True)','v_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2, transposed=True)')
Path('dev/gluon_fp8p/candidate_vector_store.py').write_text(s)
