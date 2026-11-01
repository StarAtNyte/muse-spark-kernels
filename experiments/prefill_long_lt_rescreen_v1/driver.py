"""Screen long prefill dense GEMM plans against the prepared scored baseline."""
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
    def exec_module(self, module): pass
sys.meta_path.insert(0,StubSGLang()); sys.path.insert(0,'/w/kernels/muse_prefill_layer')
import reference as R
spec=importlib.util.spec_from_file_location('sub','/w/run/sub.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
# Only scored long-prefill plans are needed in this diagnostic.
S.LT_TOKENS=(16384,)
w=R.generate_weights(R.WEIGHTS_SEED); st=S.prepare(w)
assert st.get('lt') is not None
print('READY',torch.cuda.get_device_name(),flush=True)
flush=torch.empty(256*1024*1024//4,dtype=torch.int32,device='cuda')
def capture(fn):
    out=fn();torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for name in ('qkv','pooler','unpooler'):
    wn, f32=S.LT_GEMMS[name]; W=st[wn]; N,K=W.shape; T=16384
    torch.manual_seed(411); x=torch.randn((T,K),device='cuda',dtype=torch.bfloat16)
    gb, ref=capture(lambda:S._mm(st,x,name)); plan=st['lt'].lt_plan(T,N,K,f32,st['lt_ws'].numel(),32)
    print('PLAN',json.dumps({'name':name,'N':N,'K':K,'algos':st['lt'].lt_nalgo(plan),'parent_choice':str(st['lt_choice'].get((T,name)))}),flush=True)
    out=torch.empty((T,N),device='cuda',dtype=torch.bfloat16)
    for a in range(st['lt'].lt_nalgo(plan)):
        try:
            fn=lambda:st['lt'].lt_run(plan,a,x,W,out,st['lt_ws'])
            gc,_=capture(fn)
            err=S._lt_rel_err(out,ref)
            if err >= S.LT_ALGO_TOL: raise ValueError(f'error={err}')
            bt=[];ct=[];rat=[]
            for i in range(6):
                if i&1: c,b=elapsed(gc),elapsed(gb)
                else: b,c=elapsed(gb),elapsed(gc)
                bt.append(b);ct.append(c);rat.append(b/c)
            print('RESULT',json.dumps({'name':name,'algo':a,'info':list(st['lt'].lt_info(plan,a)),'parent_us':statistics.median(bt),'candidate_us':statistics.median(ct),'ratio':statistics.median(rat),'max_relative':err}),flush=True)
            del gc
        except Exception as e: print('ERROR',name,a,repr(e),flush=True)
    del gb,ref,out,x
print('DONE',flush=True)
