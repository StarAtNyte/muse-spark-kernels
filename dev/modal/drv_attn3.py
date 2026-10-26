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
flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
def timeit(fn, reps=12):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[len(ts) // 2]
LAYER = os.environ.get("LAYER", "full")
for B in [int(x) for x in os.environ.get("BATCHES", "16,32").split(",")]:
    d = R.generate_input(B, 80000, "tp8", LAYER, 3, q_len=int(os.environ.get("QL","1")), spread="agentic" if B > 1 else "fixed", dist="coherent" if LAYER == "full" else "soft", page=1)
    res = []
    m._FD_POLICY = {('full', False): [(1, 64, 2, 32, 3)], ('sliding', False): [(1, 64, 2, 32, 3)]}
    base = m.kernel_fn(d).float().clone()
    for stages, splits, var in eval(os.environ.get("CFGS", "[(2,32,3),(2,64,3),(3,32,3),(3,64,3),(2,16,3)]")):
        m._FD_POLICY = {('full', False): [(1, 64, stages, splits, var)], ('sliding', False): [(1, 64, stages, splits, var)]}
        try:
            o = m.kernel_fn(d).float()
            err = ((o - base).norm() / base.norm()).item()
            res.append("s%d/p%d/v%d=%.1f(e%.0e)" % (stages, splits, var, timeit(lambda: m.kernel_fn(d)), err))
        except Exception as ex:
            res.append("s%d/p%d/v%d=ERR(%s)" % (stages, splits, var, str(ex)[:40]))
    print("SWEEP", LAYER, "B=%d" % B, " ".join(res), flush=True)
    del d; torch.cuda.empty_cache()
