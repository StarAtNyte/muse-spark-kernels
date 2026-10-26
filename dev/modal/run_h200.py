"""Run a problem submission/lab file on a Modal H200 (budget-capped by the user).
usage: modal run dev/modal/run_h200.py --problem muse_moe_w4a16 --file /abs/submission.py [--ncu 0]
Executes prepare(weights) (labs print LAB lines there) then a timed kernel_fn loop."""
import modal, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("cuda-nsight-compute-12-8", "ninja-build", "libnuma1")
         .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy", "triton")
         .add_local_dir(f"{ROOT}/kernels", "/w/kernels")
         .add_local_dir(f"{ROOT}/experiments/attention_fp8_hybrid_v7", "/w/v7")
         .add_local_dir(f"{ROOT}/dev/wg2", "/w/wg2")
         .add_local_dir(f"{ROOT}/dev/wg3", "/w/wg3")
         .add_local_dir(f"{ROOT}/dev/wg4", "/w/wg4")
         .add_local_dir(f"{ROOT}/dev/wg5", "/w/wg5")
         .add_local_dir(f"{ROOT}/dev/wg6", "/w/wg6")
         .add_local_dir(f"{ROOT}/dev/wg", "/w/wg")
         .add_local_dir(f"{ROOT}/dev/modal", "/w/modal")
         .add_local_dir(f"{ROOT}/dev/bw", "/w/bw")
         .add_local_dir(f"{ROOT}/dev/bf16c", "/w/bf16c"))
image_sgl = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
             .apt_install("cuda-nsight-compute-12-8", "ninja-build", "libnuma1")
             .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
             .pip_install("numpy", "triton", "sgl-kernel")
             .add_local_dir(f"{ROOT}/kernels", "/w/kernels")
             .add_local_dir(f"{ROOT}/experiments/attention_fp8_hybrid_v7", "/w/v7")
             .add_local_dir(f"{ROOT}/dev/modal", "/w/modal")
             .add_local_dir(f"{ROOT}/dev/wg", "/w/wg")
             .add_local_dir(f"{ROOT}/dev/wg5", "/w/wg5"))
image_cute = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
              .apt_install("cuda-nsight-compute-12-8", "ninja-build", "libnuma1", "git")
              .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
              .pip_install("numpy", "triton", "nvidia-cutlass-dsl")
              .add_local_dir(f"{ROOT}/kernels", "/w/kernels")
              .add_local_dir(f"{ROOT}/dev/modal", "/w/modal")
              .add_local_dir(os.path.expanduser("~/agent-skills-src/cutlass/examples/python/CuTeDSL"), "/w/cute_examples"))
app = modal.App("muse-h200", image=image)

@app.function(gpu="H200", timeout=1800, image=image_sgl)
def run_sgl(problem: str, src: str, cmd: str = "", drv_src: str = ""):
    return _run(problem, src, cmd, drv_src)


@app.function(gpu="H200", timeout=1800, image=image_cute)
def run_cute(problem: str, src: str, cmd: str = "", drv_src: str = ""):
    return _run(problem, src, cmd, drv_src)


@app.function(gpu="H200", timeout=1800)
def run(problem: str, src: str, cmd: str = "", drv_src: str = ""):
    return _run(problem, src, cmd, drv_src)


def _run(problem: str, src: str, cmd: str = "", drv_src: str = ""):
    import subprocess, sys, os
    os.makedirs("/w/run", exist_ok=True)
    open("/w/run/sub.py", "w").write(src)
    drv = r'''
import sys, os, importlib.util, torch, types
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
sys.path.insert(0, "/w/kernels/%s")
import reference as R
spec = importlib.util.spec_from_file_location("sub", "/w/run/sub.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
w = R.generate_weights(1234)
st = m.prepare(w)
d = R.generate_input(int(os.environ.get("BS","32")), "ep8", seed=1)
o = m.kernel_fn(d, st); print(R.check_implementation(d, o, w))
''' % problem
    open("/w/run/drv.py", "w").write(drv_src.replace('__PROBLEM__', problem) if drv_src else drv)
    full = cmd or "python /w/run/drv.py"
    p = subprocess.run(full, shell=True, cwd="/w/run", capture_output=True, text=True)
    return p.stdout[-900000:] + "\nSTDERR:\n" + p.stderr[-4000:]

@app.local_entrypoint()
def main(problem: str, file: str, cmd: str = "", drv: str = "", sgl: bool = False, cute: bool = False):
    fn = run_cute if cute else (run_sgl if sgl else run)
    print(fn.remote(problem, open(file).read(), cmd, open(drv).read() if drv else ""))
