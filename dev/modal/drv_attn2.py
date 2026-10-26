import sys, os, importlib.util, torch, types, time
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
B = int(os.environ.get("BS", "16")); QL = int(os.environ.get("QL", "1")); LAYER = os.environ.get("LAYER", "full")
d = R.generate_input(B, 80000, "tp8", LAYER, 3, q_len=QL, spread="agentic", dist="coherent" if LAYER == "full" else "soft", page=1)
flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
o = m.kernel_fn(d); torch.cuda.synchronize()
if os.environ.get("TIME", "1") == "1":
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        m.kernel_fn(d)
    ts = []
    for _ in range(15):
        flush.add_(1); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); print("TIME B=%d QL=%d %s median=%.1fus" % (B, QL, LAYER, ts[len(ts) // 2]), flush=True)
