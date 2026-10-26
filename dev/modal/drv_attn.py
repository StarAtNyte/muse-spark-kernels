import sys, os, importlib.util, torch, types
from unittest.mock import MagicMock
class _F:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sglang", "sgl_kernel"):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, _F())
sys.path.insert(0, "/w/kernels/__PROBLEM__")
import reference as R
spec = importlib.util.spec_from_file_location("sub", "/w/run/sub.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
B = int(os.environ.get("BS", "16")); QL = int(os.environ.get("QL", "8")); LAYER = os.environ.get("LAYER", "full")
d = R.generate_input(B, 80000, "tp8", LAYER, 1, q_len=QL, spread="agentic", dist="coherent" if LAYER == "full" else "soft", page=1)
for _ in range(2):
    o = m.kernel_fn(d)
torch.cuda.synchronize()
print("OUT", type(o), getattr(o, "shape", None), getattr(o, "dtype", None))
try:
    print("CHECK", R.check_implementation(d, o))
except Exception as ex:
    print("CHECK ERR", str(ex)[:200])
