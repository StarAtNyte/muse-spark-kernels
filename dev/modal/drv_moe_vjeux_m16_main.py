"""Compare the established M=16 skinny split kernel with the WGS main kernel."""
import importlib.util
import json
import statistics
import sys
from unittest.mock import MagicMock

import torch


class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sglang", "sgl_kernel"):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, "/w/kernels/muse_moe_w4a16")
import reference as R

spec = importlib.util.spec_from_file_location("sub", "/w/run/sub.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.TINY_DEFINES = (); m.TINY2_DEFINES = ()
weights = R.generate_weights(5252)
state = m.prepare(weights)
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)


def capture(data, mid_max):
    m.MID_MAX_M = mid_max
    out = m.kernel_fn(data, state)
    torch.cuda.synchronize()
    ok, msg = R.check_implementation(data, out, weights)
    if not ok:
        raise AssertionError(("reference", mid_max, msg))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = m.kernel_fn(data, state)
    graph.replay(); torch.cuda.synchronize()
    return graph, out


def elapsed(g):
    flush.add_(1)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000


print("READY", torch.cuda.get_device_name(), "sfx", state["sfx"], flush=True)
for seed in (419, 7919, 17319):
    data = R.generate_input(16, "ep8", seed)
    gb, ob = capture(data, 16)
    gc, oc = capture(data, 15)
    err = ((ob.float() - oc.float()).norm() / ob.float().norm().clamp_min(1e-20)).item()
    bt, ct, ratios = [], [], []
    for i in range(24):
        if i & 1:
            c, b = elapsed(gc), elapsed(gb)
        else:
            b, c = elapsed(gb), elapsed(gc)
        bt.append(b); ct.append(c); ratios.append(b / c)
    print("RESULT", json.dumps({"seed": seed, "mid_us": statistics.median(bt),
          "main_us": statistics.median(ct), "mid_over_main": statistics.median(ratios),
          "relative_l2": err}), flush=True)
    del data, gb, gc, ob, oc
print("MOE_M16_MAIN_SCREEN_DONE", flush=True)
