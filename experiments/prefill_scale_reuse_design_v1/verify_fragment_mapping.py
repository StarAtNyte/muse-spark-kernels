"""Verify the proposed permutation against the packer's explicit index order.

Pure index proof; does not model WGMMA's hardware fragment layout.
"""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
s=spec_from_file_location('perm',Path(__file__).with_name('derive_permutation.py'))
m=module_from_spec(s);s.loader.exec_module(m)
# _repack uses k=(j,hk,t,b): k=16*j+8*hk+2*t+b.
# Its packed register is indexed by lane subgroup t, j, then half/hk/b.
for t in range(4):
 for j in range(4):
  for hk in range(2):
   for b in range(2):
    p=16*j+8*hk+2*t+b
    k=m.perm[p]
    assert k==16*t+4*j+2*hk+b
    assert k//16==t
# A single scale pair (rows g and g+8) addressed by lane t can therefore
# serve every j, assuming activation fragments use the identical permutation.
for t in range(4):
 for j in range(4):
  values={m.perm[16*j+8*hk+2*t+b]//16 for hk in range(2) for b in range(2)}
  assert values=={t}
print('PASS: all 64 actual packer (j,hk,t,b) positions map to source scale block t')
print('WGMMA fragment correspondence and intermediate publication remain unverified')
