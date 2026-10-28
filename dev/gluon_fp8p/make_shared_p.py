"""Avoid FP8 DotOperandLayout redistribution by supplying P as shared A."""
from pathlib import Path
s=Path('dev/gluon_fp8p/candidate_vector_store.py').read_text()
old='''    pa = gl.convert_layout(p.to(gl.float8e4nv), gl.DotOperandLayout(0, mma_o, 4))
    o = warpgroup_mma(pa, v16_s, o * ao[:, None])'''
new='''    p_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=min(128, BN), element_bitwidth=8, rank=2)
    pa = gl.allocate_shared_memory(gl.float8e4nv, [R, BN], p_layout, p.to(gl.float8e4nv))
    fence_async_shared()
    gl.barrier()
    o = warpgroup_mma(pa, v16_s, o * ao[:, None])'''
assert old in s;s=s.replace(old,new)
Path('dev/gluon_fp8p/candidate_shared_p.py').write_text(s)
