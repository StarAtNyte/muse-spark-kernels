import os, sys
import dense_cuda as D

base = open("../../experiments/bf16_hybrid_v5/submission.py").read()
base = base.replace("def prepare(weights: Weights):", "def _old_prepare(weights: Weights):", 1)
base = base.replace("def kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:", "def _old_kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:", 1)
assert "_old_prepare" in base and "_old_kernel_fn" in base
default = "{'qkv_gate': [(10**9, (1, 4, 4, 4, 4))], 'o': [(10**9, (1, 4, 4, 4, 1))], 'pooler': [(10**9, (1, 4, 4, 4, 2))], 'unpooler': [(8, (1, 1, 4, 4, 4)), (10**9, (1, 4, 4, 4, 1))], 'router': [(10**9, (1, 4, 4, 4, 8))]}"
cfg = os.environ.get("DENSE_CFG", default)
repack = open("dense_cuda.py").read().split("def dense_repack(w):", 1)[1]
repack = "def _dense_repack(w):" + repack.split("def dense_crepack(w):")[0]
crep = "def _dense_crepack(w):" + open("dense_cuda.py").read().split("def dense_crepack(w):", 1)[1].replace("dense_repack(", "_dense_repack(")
repack += "\n\n" + crep
out = base + "\n\n# ---- weight-streaming dense GEMM (CUDA) ----\n_DENSE_CUDA = r'''" + D.CUDA_SRC + D.TORCH_GLUE + "'''\n_DENSE_CPP = r'''" + D.CPP_SRC + "'''\n\n" + repack + "\n"
out += open("glue_bf16.py").read().replace("__CFG__", cfg)
if os.environ.get("DENSE_LAB"):
    out += open(os.environ.get("LABFILE", "lab_bf16.py")).read() + "\n_DENSE_LAB = True\n"
open(sys.argv[1], "w").write(out)
print("wrote")
