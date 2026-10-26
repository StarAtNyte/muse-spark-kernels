import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf16c_cuda as B

base = open("../../experiments/scored_bf16_v5/submission.py").read()
assert base.count("\ndef prepare(") == 1 and base.count("\ndef kernel_fn(") == 1
base = base.replace("\ndef prepare(", "\ndef _prepare_v5(", 1).replace("\ndef kernel_fn(", "\ndef _kernel_v5(", 1)
# helper: reuse torch repack from bf16c_cuda.pack
import inspect
pack_src = inspect.getsource(B.pack).replace("def pack(", "def _bc_pack(", 1)
out = base + "\n\n_BC_CUDA = r'''" + B.CUDA_SRC + B.TORCH_GLUE + "'''\n_BC_CPP = r'''" + B.CPP_SRC + "'''\n\n" + pack_src + "\n" + open("glue_bf16c.py").read()
open(sys.argv[1], "w").write(out)
print("wrote", len(out))
