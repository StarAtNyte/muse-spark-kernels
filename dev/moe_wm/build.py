"""Assemble submission.py = v5 base (Marlin fallback) + weight-major CUDA path."""
import re
import sys

import wm_cuda

base = open("base_v5.py").read()
base = base.replace("def prepare(weights):", "def _prepare_marlin(weights):", 1)
base = base.replace("def kernel_fn(data, state):", "def _marlin_kernel_fn(data, state):", 1)
assert "_prepare_marlin" in base and "_marlin_kernel_fn" in base

repack_src = open("wm_cuda.py").read().split("def repack(w, sf):", 1)[1]
repack_src = "def _wm_repack(w, sf):" + repack_src
repack_src = repack_src.replace("    import torch\n\n", "")

import os
cfg = os.environ.get("WM_CFG_SPEC", "[(8, (2, 4, 0)), (16, (2, 4, 0)), (10**9, (4, 4, 1))]")
glue = open("wm_glue.py").read().replace("__WM_CFG__", cfg)
out = base + "\n\n# ---- weight-major NVFP4 path (CUDA) ----\n"
out += "_WM_CUDA = r'''" + wm_cuda.CUDA_SRC + wm_cuda.TORCH_GLUE + "'''\n"
out += "_WM_CPP = r'''" + wm_cuda.CPP_SRC + "'''\n\n"
out += repack_src + "\n\n" + glue
if os.environ.get("WM_LAB"):
    out += "\n\n_LAB_CFGS = " + os.environ["WM_LAB"] + "\n" + open("wm_lab.py").read() + "\n"
    out = out.replace("    torch.cuda.synchronize()\n    return state\n\n\nWM_MAX_TOKENS", "    torch.cuda.synchronize()\n    _lab(state)\n    return state\n\n\nWM_MAX_TOKENS")
open(sys.argv[1] if len(sys.argv) > 1 else "submission.py", "w").write(out)
print("wrote")
