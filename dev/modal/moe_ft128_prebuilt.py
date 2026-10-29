"""Compile MoE extensions on CPUs, then test the binaries on one bounded H200."""
import os
import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("ninja-build", "libnuma1")
         .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy", "triton")
         .add_local_dir(f"{ROOT}/kernels", "/w/kernels"))
app = modal.App("muse-ft128-prebuilt", image=image)


@app.function(cpu=8, memory=32768, timeout=1000)
def compile_one(src: str):
    import subprocess
    os.makedirs("/w/run", exist_ok=True)
    open("/w/run/sub.py", "w").write(src)
    driver = '''import importlib.util, torch
torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
spec = importlib.util.spec_from_file_location("candidate", "/w/run/sub.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ext = m._ext(tuple(m.MAIN_DEFINES) + ("-DOKC_PRMTSC=1",))
print("EXT_PATH=" + ext.__file__, flush=True)
'''
    open("/w/run/compile.py", "w").write(driver)
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST="9.0", MAX_JOBS="4")
    p = subprocess.run(["timeout", "750s", "python", "/w/run/compile.py"], capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError(f"compile exit {p.returncode}: {p.stdout[-2000:]} {p.stderr[-8000:]}")
    path = next(line[9:] for line in p.stdout.splitlines() if line.startswith("EXT_PATH="))
    return os.path.basename(path), open(path, "rb").read()


@app.function(gpu="H200", timeout=180)
def screen(base_src: str, candidate_src: str, blobs):
    import subprocess
    os.makedirs("/w/run", exist_ok=True)
    open("/w/run/base.py", "w").write(base_src)
    open("/w/run/cand.py", "w").write(candidate_src)
    for name, blob in blobs:
        open("/w/run/" + name, "wb").write(blob)
    driver = r'''import importlib.util, json, statistics, sys
from unittest.mock import MagicMock
import torch
class Stub:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang','sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self, module): pass
sys.meta_path.insert(0, Stub());sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def ext(path):
    name=path.rsplit('/',1)[1].split('.')[0]
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=load('base','/w/run/base.py');cand=load('cand','/w/run/cand.py')
be=ext('/w/run/' + sys.argv[1]);ce=ext('/w/run/' + sys.argv[2])
for m,e in ((base,be),(cand,ce)):
    m.TINY_DEFINES=();m.TINY2_DEFINES=();m.MID_DEFINES=()
    m._ext=lambda defines=None,e=e:e
    e.okc_set_pdl(1 if m.PDL else 0)
weights=R.generate_weights(5252);sb=base.prepare(weights);sc=cand.prepare(weights)
print('READY',torch.cuda.get_device_name(),sb['sfx'],sc['sfx'],flush=True)
assert sb['sfx']==('-DOKC_PRMTSC=1',) and sc['sfx']==sb['sfx']
flush=torch.empty(64*1024*1024,dtype=torch.int32,device='cuda')
def capture(m,state,data):
    out=m.kernel_fn(data,state);torch.cuda.synchronize()
    ok,msg=R.check_implementation(data,out,weights)
    if not ok:raise AssertionError((m.__name__,msg))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=m.kernel_fn(data,state)
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for bs in (32,128,256):
  for seed in (419,7919):
    data=R.generate_input(bs,'ep8',seed);gb,ob=capture(base,sb,data);gc,oc=capture(cand,sc,data)
    err=((ob.float()-oc.float()).norm()/ob.float().norm().clamp_min(1e-20)).item()
    bt=[];ct=[];rat=[]
    for i in range(20):
      if i&1:c,b=elapsed(gc),elapsed(gb)
      else:b,c=elapsed(gb),elapsed(gc)
      bt.append(b);ct.append(c);rat.append(b/c)
    print('RESULT',json.dumps(dict(batch=bs,seed=seed,baseline_us=statistics.median(bt),candidate_us=statistics.median(ct),base_over_candidate=statistics.median(rat),relative_l2=err)),flush=True)
print('SCREEN_DONE',flush=True)
'''
    open('/w/run/screen.py', 'w').write(driver)
    p = subprocess.run(['timeout', '120s', 'python', '/w/run/screen.py', blobs[0][0], blobs[1][0]], capture_output=True, text=True)
    return {"returncode": p.returncode, "stdout": p.stdout[-10000:], "stderr": p.stderr[-10000:]}


@app.local_entrypoint()
def main(base: str, candidate: str):
    import json
    b, c = open(base).read(), open(candidate).read()
    blobs = [compile_one.remote(b), compile_one.remote(c)]
    print('COMPILED', [(name, len(data)) for name, data in blobs], flush=True)
    print(json.dumps(screen.remote(b, c, blobs)))
