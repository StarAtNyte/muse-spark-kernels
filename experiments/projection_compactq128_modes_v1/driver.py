"""Paired targeted qkv32 screen with only the two required backend preparations."""
import importlib.util, json, statistics, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m=MagicMock(); m.__path__=[]; m.__name__=spec.name; m.__spec__=spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
sp=importlib.util.spec_from_file_location('candidate','/w/run/sub.py'); S=importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w=R.generate_weights(R.WEIGHTS_SEED)
configs=[(64,36,128,3,8,539,5),(64,36,128,3,8,6683,5)]
st=S.tt_v14_qkv_prepare_qbig_candidates(w,configs)
base={'kernel_fn':lambda d, state:S.tt_v14_qkv_run_qbig_candidate(d['x'],state,0)}
cand={'kernel_fn':lambda d, state:S.tt_v14_qkv_run_qbig_candidate(d['x'],state,1)}
sb=sc=st
flush=torch.zeros(256*1024*1024//4,device='cuda')
def cap(K,state,d):
    out=K['kernel_fn'](d,state); torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=K['kernel_fn'](d,state)
    g.replay(); torch.cuda.synchronize(); return g,out
def time(g):
    flush.sum(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b)*1000
for seed in (61,62,63):
    d=R.generate_input(128,'qkv_gate',seed); gb,ob=cap(base,sb,d); gc,oc=cap(cand,sc,d)
    checks={}
    for name,out in [('base',ob),('cand',oc)]: checks[name]=R.check_implementation(d,out,w)[0]
    times={'base':[],'cand':[]}
    for i in range(20):
        for name,g in ([('base',gb),('cand',gc)] if i%2==0 else [('cand',gc),('base',gb)]): times[name].append(time(g))
    x,y=(statistics.median(times[n]) for n in ('base','cand'))
    print('ROW',json.dumps({'seed':seed,'base_us':round(x,3),'cand_us':round(y,3),'gain':round(x/y,4),'correct':checks}),flush=True)
print('DONE',flush=True)
