import os, shutil, subprocess, sys
if not os.environ.get('PREFILL_NCU_INNER'):
    ncu = shutil.which('ncu')
    if not ncu:
        print('NCU_MISSING', flush=True); raise SystemExit(1)
    env = dict(os.environ, PREFILL_NCU_INNER='1')
    cmd = [ncu, '--clock-control', 'none', '--target-processes', 'all', '--profile-from-start', 'off', '--kernel-name', 'regex:moe_fused_persistent', '--launch-count', '1', '--section', 'SpeedOfLight', '--section', 'SchedulerStats', '--section', 'WarpStateStats', sys.executable, __file__]
    p = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=275)
    print('NCU_RETURNCODE',p.returncode,flush=True);print(p.stdout,flush=True);print('NCU_STDERR',p.stderr,flush=True)
    raise SystemExit(p.returncode)
"""ncu target: one prefill layer call (first turn, 16384 tokens, full) so the fused MoE kernel can be profiled.
Attention choice pinned after prepare (the in-prepare FA3 validation cannot run on Modal)."""
import importlib.util, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_prefill_layer')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
st['attn_choice'] = dict(S.ATTN_FIXED)
c = dict(batch=1, ctx=80000, layer='sliding', page=1, dist='soft', q_len=753, seed=9400)
if hasattr(R, 'case_fields'): c.update(R.case_fields(c, w))
d = R.generate_input(**c)
o = S.kernel_fn(d, st); torch.cuda.synchronize()
print('CHECK', R.check_implementation(d, R.collect(d, o), w), flush=True)
torch.cuda.profiler.start(); S.kernel_fn(d, st); torch.cuda.synchronize(); torch.cuda.profiler.stop()
print('DONE', flush=True)
