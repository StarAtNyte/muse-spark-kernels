"""Reduce live weight-register staging in the native small-batch MoE."""
from pathlib import Path
import ast
s=Path('experiments/moe_wm_v22/submission.py').read_text()
a=s.index('  if (RING == 0) {');b=s.index('  } else if (RING == 2)',a)
x=s[a:b]
old='''      uint4 wv[NTB][4];
#pragma unroll
      for (int n = 0; n < NTB; n++) {
        sc[n] = __ldg(ssrc[n] + q * 32);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          if constexpr (DBG & 8) wv[n][c] = make_uint4(lane * 7 + c, lane * 13 + q, lane + n, lane * 3);
          else wv[n][c] = __ldg(wsrc[n] + (q * 4 + c) * 32);
        }
      }
#pragma unroll
      for (int c = 0; c < 4; c++) {'''
new='''#pragma unroll
      for (int n = 0; n < NTB; n++) sc[n] = __ldg(ssrc[n] + q * 32);
#pragma unroll
      for (int cp = 0; cp < 4; cp += 2) {
      uint4 wv[NTB][2];
#pragma unroll
      for (int n = 0; n < NTB; n++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const int c = cp + j;
          if constexpr (DBG & 8) wv[n][j] = make_uint4(lane * 7 + c, lane * 13 + q, lane + n, lane * 3);
          else wv[n][j] = __ldg(wsrc[n] + (q * 4 + c) * 32);
        }
      }
#pragma unroll
      for (int c = cp; c < cp + 2; c++) {'''
assert x.count(old)==1
x=x.replace(old,new).replace('{wv[n][c].x, wv[n][c].y, wv[n][c].z, wv[n][c].w}', '{wv[n][c-cp].x, wv[n][c-cp].y, wv[n][c-cp].z, wv[n][c-cp].w}')
assert x.endswith('    }\n')
x=x[:-len('    }\n')]+'      }\n    }\n'
s=s[:a]+x+s[b:]
old='  if (NT == 8) dyn = NW * 3 * 8 * 512 + 1024;'
new=old+'\n  if (RING == 0) dyn = NW * (NT > 2 * NMAX ? NT : 2 * NMAX) * 4 * 32 * sizeof(float);'
assert s.count(old)==1
s=s.replace(old,new)
ast.parse(s)
Path('dev/ptx_campaign/moe_preload2.py').write_text(s)
p=Path('dev/modal/drv_moe_ptx.py').read_text()
a=p.index('variants =');b=p.index('flush =',a)
p=p[:a]+"variants = [('preload2', open('/w/run/sub.py').read(), [])]\n"+p[b:]
# Candidate is already the runtime module; validated parent source is mounted
# separately for a fair original-module baseline.
p=p.replace("spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')", "spec = importlib.util.spec_from_file_location('sub', '/w/kernels/muse_moe_w4a16/submission.py')")
p=p.replace("open('/w/run/sub.py').read(), []", "candidate_source, []")
insert="""candidate_source = open('/w/run/sub.py').read()
import ast
candidate_cuda = next(ast.literal_eval(n.value) for n in ast.parse(candidate_source).body
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_WM_CUDA' for t in n.targets))
"""
p=p.replace('variants =',insert+'variants =').replace("('preload2', candidate_source, [])", "('preload2', candidate_cuda, [])")
p=p.replace('for batch in (1, 8, 16, 32, 128, 256):','for batch in (1, 8, 16, 32, 128):')
Path('dev/modal/drv_moe_preload2.py').write_text(p)
ast.parse(p)
