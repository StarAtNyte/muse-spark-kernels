"""Paired H200 screen for a revised WGS cost curve."""
import importlib.util, json, statistics, sys
from unittest.mock import MagicMock
import torch

class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sglang", "sgl_kernel"):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m=MagicMock(); m.__path__=[]; m.__name__=spec.name; m.__spec__=spec; return m
    def exec_module(self, module): pass

sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0,"/w/kernels/muse_moe_w4a16")
import reference as R

def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
base=load("base","/w/kernels/muse_moe_w4a16/submission.py")
cand=load("cand","/w/run/sub.py")
for m in (base,cand): m.TINY_DEFINES=(); m.TINY2_DEFINES=(); m.MID_DEFINES=()
weights=R.generate_weights(5252)
sb=base.prepare(weights); sc=cand.prepare(weights)
print("READY",torch.cuda.get_device_name(),flush=True)
flush=torch.empty(64*1024*1024,dtype=torch.int32,device="cuda")

def capture(m,state,data):
    out=m.kernel_fn(data,state); torch.cuda.synchronize()
    ok,msg=R.check_implementation(data,out,weights)
    if not ok: raise AssertionError((m.__name__,msg))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=m.kernel_fn(data,state)
    g.replay(); torch.cuda.synchronize()
    return g,out
def elapsed(g):
    flush.add_(1)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000

for bs in (32,128,256):
  for seed in (419,7919):
    data=R.generate_input(bs,"ep8",seed)
    gb,ob=capture(base,sb,data); gc,oc=capture(cand,sc,data)
    err=((ob.float()-oc.float()).norm()/ob.float().norm().clamp_min(1e-20)).item()
    bt=[];ct=[];rat=[]
    for i in range(20):
      if i&1: c,b=elapsed(gc),elapsed(gb)
      else: b,c=elapsed(gb),elapsed(gc)
      bt.append(b);ct.append(c);rat.append(b/c)
    print("RESULT",json.dumps({"batch":bs,"seed":seed,"base_us":statistics.median(bt),"candidate_us":statistics.median(ct),"base_over_candidate":statistics.median(rat),"relative_l2":err}),flush=True)
    del data,gb,gc,ob,oc
print("COSTCURVE_SCREEN_DONE",flush=True)
