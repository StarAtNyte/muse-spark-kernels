"""ncu DRAM-bytes probe: per (bs, seed) one profiled kernel_fn launch (ncu flushes caches => cold L2)."""
import importlib.util, sys, torch
from unittest.mock import MagicMock
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sglang", "sgl_kernel"):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, module): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, "/w/kernels/muse_moe_w4a16")
import reference as R
sp = importlib.util.spec_from_file_location("sub", "/w/run/sub.py")
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
w = R.generate_weights(R.WEIGHTS_SEED); st = m.prepare(w)
EXP = 3 * 4096 * 4096 * (0.5 + 1 / 16)
for bs in (8, 16, 32, 128, 256):
    for seed in (403, 419, 1234):
        d = R.generate_input(bs, "ep8", seed)
        ids = d["topk_ids"]; loc = ids[ids < 32]; c = torch.bincount(loc.flatten(), minlength=32)
        U = int((c > 0).sum()); jobs = int(((c + 63) // 64).sum())
        o = m.kernel_fn(d, st); torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, o, w)
        print(f"CASE bs={bs} seed={seed} U={U} jobs64={jobs} uniqueMB={U*EXP/1e6:.1f} jobMB={jobs*EXP/1e6:.1f} ok={ok}", flush=True)
        torch.cuda.profiler.start(); m.kernel_fn(d, st); torch.cuda.synchronize(); torch.cuda.profiler.stop()
print("DONE", flush=True)
