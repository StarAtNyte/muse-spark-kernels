"""CPU-only nvcc/PTXAS triage; does not allocate a GPU."""
from pathlib import Path
import modal
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
app=modal.App('muse-compile-triage',image=image)
@app.function(cpu=4,memory=4096,timeout=240)
def compile_device(source:str,opt:int=3):
 import subprocess,tempfile,os
 with tempfile.TemporaryDirectory() as d:
  cu=os.path.join(d,'kernel.cu');ptx=os.path.join(d,'kernel.ptx');cubin=os.path.join(d,'kernel.cubin')
  open(cu,'w').write(source)
  logs=[]
  for name,cmd in (
   ('nvcc_ptx',['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_90a','--ptx',cu,'-o',ptx]),
   ('ptxas',['/usr/local/cuda/bin/ptxas','-arch=sm_90a',f'-O{opt}','-v',ptx,'-o',cubin])):
   r=subprocess.run(cmd,capture_output=True,text=True,timeout=150)
   logs.append(dict(stage=name,returncode=r.returncode,stdout=r.stdout[-2000:],stderr=r.stderr[-10000:]))
   if r.returncode:break
  return logs
@app.local_entrypoint()
def main(file:str,variants:bool=False):
 import json
 src=Path(file).read_text();marker='#include <torch/extension.h>'
 if marker in src:src=src[:src.index(marker)]
 if variants:
  for bound,opt in ((4,2),(2,3),(1,3)):
   candidate=src.replace('__launch_bounds__(128, 4)',f'__launch_bounds__(128, {bound})')
   print('VARIANT',bound,opt,json.dumps(compile_device.remote(candidate,opt),indent=2),flush=True)
 else:print(json.dumps(compile_device.remote(src),indent=2))
