"""Derive a non-overflowing lazy-max range for FP8 probability operands."""
from pathlib import Path
import ast,math
p=Path('kernels/muse_attention_decode/submission.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));s=ast.literal_eval(n.value)
assert s.count('#define LAZY_TH 0.0f')==1
assert 2**(8+.75)<448
variants={th:s.replace('#define LAZY_TH 0.0f',f'#define LAZY_TH {th}f') for th in (0.,.25,.5,.75)}
Path('dev/ptx_campaign/attention_fp8p_lazy_bundle.py').write_text(p+'\n_LAZY_VARIANTS='+repr(variants)+'\n')
print('Safe theoretical limit:',math.log2(448)-8,'; tested maximum:',.75)
