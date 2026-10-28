"""Screen split-KV plans for B32, q_len=1 sliding decode on H200."""
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
        m = MagicMock()
        m.__path__ = []
        m.__name__ = spec.name
        m.__spec__ = spec
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, "/w/kernels/muse_attention_decode")
import reference as R

spec = importlib.util.spec_from_file_location("cand", "/w/run/sub.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)


def cap(plan, seed):
    data = R.generate_input(32, 80000, "tp8", "sliding", seed,
                            spread="agentic", dist="soft", page=1)
    out = m._launch(data, **plan)
    torch.cuda.synchronize()
    ok, msg = R.check_implementation(data, R.collect(data, out))
    if not ok:
        raise AssertionError(("reference", plan, seed, msg))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        m._launch(data, **plan)
    graph.replay()
    torch.cuda.synchronize()
    # Keep tensors alive: CUDA graphs retain their addresses, not Python refs.
    return graph, data, out


def elapsed(graph):
    flush.add_(1)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); graph.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000


seed = 56321
base = m._plan(32, 1, 1, 2048, 80000, 1, torch.device("cuda"))
configs = [("base", base)]
for splits in (4, 6, 8, 12, 16, 24, 32):
    if splits == base["n_splits"]:
        continue
    configs.append((f"split{splits}", {**base, "n_splits": splits, "flat": 0,
                                         "cs": 0, "ncl": 1, "nc": 1}))
graphs = {}
for name, plan in configs:
    graphs[name] = cap(plan, seed)
    print("CAPTURED", json.dumps({"name": name, "plan": plan}), flush=True)

base_graph = graphs["base"][0]
for name, _ in configs[1:]:
    ratios, base_us, candidate_us = [], [], []
    for i in range(20):
        if i & 1:
            ct, bt = elapsed(graphs[name][0]), elapsed(base_graph)
        else:
            bt, ct = elapsed(base_graph), elapsed(graphs[name][0])
        ratios.append(bt / ct); base_us.append(bt); candidate_us.append(ct)
    print("RESULT", json.dumps({"name": name, "speedup": statistics.median(ratios),
          "base_us": statistics.median(base_us),
          "candidate_us": statistics.median(candidate_us)}), flush=True)
print("SCREEN_DONE", torch.cuda.get_device_name(), flush=True)
