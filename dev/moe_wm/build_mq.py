"""submission = v5 base (Marlin fallback) + Marlin-layout CUDA MoE path (mq)."""
import ast
import inspect
import os
import sys

import mq_cuda

base = open("base_v5.py").read()
base = base.replace("def prepare(weights):", "def _prepare_marlin(weights):", 1)
base = base.replace("def kernel_fn(data, state):", "def _marlin_kernel_fn(data, state):", 1)
assert "_prepare_marlin" in base and "_marlin_kernel_fn" in base

pack_src = inspect.getsource(mq_cuda.marlin_pack).replace("def marlin_pack(", "def _mq_pack_torch(", 1)
glue = open("mq_glue.py").read().replace("__MQ_S__", os.environ.get("MQ_S", "4"))
out = base + "\n\n# ---- Marlin-layout NVFP4 MoE path (CUDA) ----\n"
out += "_MQ_CUDA = r'''" + mq_cuda.CUDA_SRC + mq_cuda.TORCH_GLUE + "'''\n"
out += "_MQ_CPP = r'''" + mq_cuda.CPP_SRC + "'''\n\n"
out += pack_src + "\n\n" + glue
if os.environ.get("MQ_LAB"):
    out = out.replace("_MQ_LAB = False", "_MQ_LAB = True")
    lab = open("wm_lab.py").read()
    tree = ast.parse(lab)
    helpers = [ast.get_source_segment(lab, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_lab_inputs", "_lab_time")]
    out += "\n\n" + "\n\n\n".join(helpers) + "\n\n\n" + open(os.environ.get("MQ_LABFILE","mq_lab.py")).read() + "\n"
open(sys.argv[1] if len(sys.argv) > 1 else "submission_mq.py", "w").write(out)
print("wrote")
