"""Screen long prefill dense GEMM plans against the prepared scored baseline."""
import importlib.util, json, statistics, sys, random
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
    flush.sum();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000

name='qkv'; wn,f32=S.LT_GEMMS[name]; W=st[wn]; N,K=W.shape; T=16384
plan=st['lt'].lt_plan(T,N,K,f32,st['lt_ws'].numel(),32)
infos=[list(st['lt'].lt_info(plan,i)) for i in range(st['lt'].lt_nalgo(plan))]
print('INFOS',json.dumps(infos),flush=True)
wanted=[i for i,v in enumerate(infos) if (v[0],v[1],v[-2]) in ((24,1,7),(184,1,2))]
if len(wanted)!=2: raise RuntimeError('target descriptors unavailable')
for seed in (411,719,1013):
    torch.manual_seed(seed); x=torch.randn((T,K),device='cuda',dtype=torch.bfloat16)
    ga,ref=capture(lambda:S._mm(st,x,name)); gb,ref2=capture(lambda:S._mm(st,x,name))
    graphs={'base_a':ga,'base_b':gb}; outs={}
    for idx in wanted:
        out=torch.empty((T,N),device='cuda',dtype=torch.bfloat16);outs[idx]=out
        g,_=capture(lambda idx=idx,out=out:st['lt'].lt_run(plan,idx,x,W,out,st['lt_ws']))
        err=S._lt_rel_err(out,ref)
        if err >= S.LT_ALGO_TOL: raise RuntimeError(f'algo {idx} err {err}')
        graphs[f'algo_{infos[idx][0]}']=g
    ts={k:[] for k in graphs}; rng=random.Random(seed)
    for rep in range(24):
        order=list(graphs);rng.shuffle(order)
        for k in order: ts[k].append(elapsed(graphs[k]))
    med={k:statistics.median(v) for k,v in ts.items()}
    print('ROW',json.dumps({'seed':seed,'us':med,'base_a_over_b':med['base_a']/med['base_b'],'ratios':{k:med['base_a']/v for k,v in med.items()}}),flush=True)
    torch.cuda.synchronize()
    graphs.clear(); del ga, gb, g, ref, ref2, outs, out, x
    torch.cuda.empty_cache()
print('DONE',flush=True)
