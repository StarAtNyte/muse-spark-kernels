"""H200 screen using Torch 2.11 plus the Triton Gluon API used by current kernels."""
import os, sys, modal
sys.path.insert(0,os.path.dirname(__file__))
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
image=(modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
 .apt_install('cuda-nsight-compute-12-8','ninja-build','libnuma1')
 .pip_install('torch==2.11.0',index_url='https://download.pytorch.org/whl/cu128')
 .pip_install('numpy','sgl-kernel')
 .pip_install('triton==3.8.0',extra_options='--no-deps')
 .add_local_dir(f'{ROOT}/kernels','/w/kernels')
 .add_local_dir(f'{ROOT}/dev/modal','/w/modal'))
app=modal.App('muse-gluon211-h200',image=image)
def _run(problem,src,drv_src):
 import subprocess,os
 os.makedirs('/w/run',exist_ok=True)
 open('/w/run/sub.py','w').write(src)
 open('/w/run/drv.py','w').write(drv_src.replace('__PROBLEM__',problem))
 p=subprocess.run('python /w/run/drv.py',shell=True,cwd='/w/run',capture_output=True,text=True)
 return p.stdout[-900000:]+'\nSTDERR:\n'+p.stderr[-4000:]
@app.function(gpu='H200',timeout=1200)
def run(problem,src,drv_src):return _run(problem,src,drv_src)
@app.local_entrypoint()
def main(problem,file,drv):print(run.remote(problem,open(file).read(),open(drv).read()))
