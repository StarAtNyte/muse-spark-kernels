from pathlib import Path
import ast
p=Path('kernels/muse_attention_decode/submission.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));s=ast.literal_eval(n.value)
assert 'cp.async.cg.shared.global [' in s
grid=s.replace('const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;','const int split = blockIdx.y, kh = blockIdx.x, b = blockIdx.z;').replace('dim3 grid(splits, 2, B);','dim3 grid(2, splits, B);')
assert grid!=s and 'dim3 grid(2, splits, B);' in grid
v={'h64':s.replace('cp.async.cg.shared.global [','cp.async.cg.shared.global.L2::64B ['),'h128':s.replace('cp.async.cg.shared.global [','cp.async.cg.shared.global.L2::128B ['),'interleave':grid,'h128_interleave':grid.replace('cp.async.cg.shared.global [','cp.async.cg.shared.global.L2::128B [')}
Path('dev/ptx_campaign/attention_fp8p_cache_bundle.py').write_text(p+'\n_CACHE_VARIANTS='+repr(v)+'\n')
