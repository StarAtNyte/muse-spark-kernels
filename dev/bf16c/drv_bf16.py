import sys, os, importlib.util, torch, math
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
sys.path.insert(0, "/w/kernels/muse_bf16_proj")
import reference as R
spec = importlib.util.spec_from_file_location("sub", "/w/run/sub.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
w = R.generate_weights(R.WEIGHTS_SEED)
st = m.prepare(w)
flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
def timeit(fn, reps=15):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.sum(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[len(ts) // 2]
tests = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"), (130, "unpooler", "normal"), (5, "unpooler", "normal"),
         (8, "router", "normal"), (9, "router", "outliers"), (512, "qkv_gate", "outliers"), (1000, "unpooler", "outliers")]
bad = 0
for mm, layer, dist in tests:
    d = R.generate_input(mm, layer, 7 + mm, dist=dist)
    out = m.kernel_fn(d, st)
    ok, msg = R.check_implementation(d, out, w)
    bad += (not ok)
    print("TEST", mm, layer, dist, ok, msg[:60], flush=True)
print("TESTS bad=%d" % bad)
sp = []
for load, ms in (("decode1", 1), ("decode16", 16), ("decode32", 32), ("verify1", 8), ("verify16", 128), ("verify32", 256)):
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        d = R.generate_input(ms, layer, 11 + ms)
        n = d["config"]["n"]
        x = d["x"]
        if layer == "router":
            continue
        base = timeit(lambda: torch.nn.functional.linear(x, w[layer][:n]))
        mine = timeit(lambda: m.kernel_fn(d, st))
        sp.append(base / mine)
        print("CASE %-9s %-9s base=%.1f mine=%.1f x%.3f" % (load, layer, base, mine, base / mine), flush=True)
print("GEOMEAN(non-router, raw event timings incl fixed overhead) = %.4f" % math.exp(sum(math.log(s) for s in sp) / len(sp)))
