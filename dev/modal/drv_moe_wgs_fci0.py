"""Bounded A/B screen for the WGS activation-gather batch size (8 -> 16)."""
import importlib.util
import json
import os
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
m.TINY_DEFINES = (); m.TINY2_DEFINES = (); m.MID_DEFINES = ()
base_defines = tuple(m.MAIN_DEFINES)
gath_batch = 2
variant = "wgsfci0"
m.ALT_SRCS[variant] = m.CUDA_SRC.replace(
    "#define OKC_WGSFCI 3", "#define OKC_WGSFCI 0", 1)
assert m.ALT_SRCS[variant] != m.CUDA_SRC
candidate_defines = base_defines + (f"@{variant}",)
weights = R.generate_weights(5252)
state = m.prepare(weights)
print("READY", torch.cuda.get_device_name(), "sfx", state["sfx"], flush=True)
flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device="cuda")


def capture(data, defines):
    m.MAIN_DEFINES = defines
    out = m.kernel_fn(data, state)
    torch.cuda.synchronize()
    ok, msg = R.check_implementation(data, out, weights)
    if not ok:
        raise AssertionError((defines, msg))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = m.kernel_fn(data, state)
    graph.replay(); torch.cuda.synchronize()
    return graph, out


def elapsed(graph):
    flush.add_(1)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); graph.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000


for batch in (32, 128, 256):
    for seed in (419, 7919):
        data = R.generate_input(batch, "ep8", seed)
        gb, ob = capture(data, base_defines)
        gc, oc = capture(data, candidate_defines)
        rel = ((ob.float() - oc.float()).norm() / ob.float().norm().clamp_min(1e-20)).item()
        bt, ct, ratios = [], [], []
        for i in range(16):
            if i & 1:
                c, b = elapsed(gc), elapsed(gb)
            else:
                b, c = elapsed(gb), elapsed(gc)
            bt.append(b); ct.append(c); ratios.append(b / c)
        print("RESULT", json.dumps({"batch": batch, "seed": seed,
              "base_us": statistics.median(bt), f"gath{gath_batch}_us": statistics.median(ct),
              "base_over_candidate": statistics.median(ratios), "relative_l2": rel}), flush=True)
        del data, gb, gc, ob, oc
print("MOE_WGS_GATHB16_DONE", flush=True)
