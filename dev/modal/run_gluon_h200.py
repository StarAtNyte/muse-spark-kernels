"""H200 diagnostic image with the Gluon 3.8 API used by official passing kernels."""
import os,sys,modal
sys.path.insert(0,os.path.dirname(__file__))
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
image=(modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
 .apt_install('cuda-nsight-compute-12-8','ninja-build','libnuma1')
 .pip_install('torch==2.9.1',index_url='https://download.pytorch.org/whl/cu128')
 .pip_install('numpy','sgl-kernel')
 .pip_install('triton==3.8.0',extra_options='--no-deps')
 .add_local_dir(f'{ROOT}/kernels','/w/kernels')
 .add_local_dir(f'{ROOT}/dev/modal','/w/modal'))
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

app=modal.App('muse-gluon-h200',image=image)
@app.function(gpu='H200',timeout=1200)
def run(problem:str,src:str,drv_src:str,cmd:str=""):
 return _run(problem,src,cmd=cmd,drv_src=drv_src)
@app.local_entrypoint()
def main(problem:str,file:str,drv:str,cmd:str=""):
 print(run.remote(problem,open(file).read(),open(drv).read(),cmd))
