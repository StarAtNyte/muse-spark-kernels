"""Paired screen of stage scale-pair reuse on the scored KPERM3 WGS path."""
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
base_defines = tuple(m.MAIN_DEFINES)
anchor = "    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);"
old = """const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;"""
hoist = """
    uint32_t stage_scale[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t sw = j < 2 ? scs.x : scs.y, sb = (uint32_t)(j & 1) * 2u;
      stage_scale[j] = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
    }"""
assert m.CUDA_SRC.count(anchor) == 1 and m.CUDA_SRC.count(old) == 1
m.ALT_SRCS['scalehoist'] = m.CUDA_SRC.replace(anchor, anchor + hoist).replace(old, 'const uint32_t sgp = stage_scale[js];')
candidate_defines = base_defines + ('@scalehoist',)
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
              "base_us": statistics.median(bt), "candidate_us": statistics.median(ct),
              "base_over_candidate": statistics.median(ratios), "relative_l2": rel}), flush=True)
        del data, gb, gc, ob, oc
print("MOE_SCALE_HOIST_DONE", flush=True)
