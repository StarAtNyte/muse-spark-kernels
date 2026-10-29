"""Resource usage (registers, stack, local spills) of the scored MoE builds."""
import importlib.util, subprocess, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
st = S.prepare(R.generate_weights(R.WEIGHTS_SEED))
for defs, e in S._EXTS.items():
    out = subprocess.run(f"cuobjdump --dump-resource-usage {e.__file__} | grep -E 'Function|REG'", shell=True, capture_output=True, text=True).stdout
    print('BUILD', ' '.join(d for d in defs if 'NW' in d or 'TINY' in d or 'WGS' in d or 'SPLIT' in d), flush=True)
    print(out[:3000], flush=True)
