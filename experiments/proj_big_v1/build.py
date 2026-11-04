"""submission.py = parent + pb_part + wrap_part (routes from argv json or {})"""
import sys, json, os
here = os.path.dirname(os.path.abspath(__file__))
parent = open(os.path.join(here, '../projection_fp32_compactq128_bucket_v7/submission.py')).read()
routes = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
r = "{" + ", ".join(f"({k.split(':')[0]!r}, {int(k.split(':')[1])}): {tuple(v)}" for k, v in routes.items()) + "}"
lines = parent.split('\n')
lines.insert(1, "# + proj_big (team keell0renz): m=128/256 BF16 projections as a TMA + register-A wgmma GEMM on the lossless 12-bit weight codec of vjeux/okc-2026 (Apache-2.0), split-K combined through cluster DSMEM")
src = '\n'.join(lines) + open(os.path.join(here, 'pb_part.py')).read() + open(os.path.join(here, 'wrap_part.py')).read().replace('@@ROUTES@@', r)
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, 'submission.py')
open(out, 'w').write(src)
print(out, len(src.encode()))
