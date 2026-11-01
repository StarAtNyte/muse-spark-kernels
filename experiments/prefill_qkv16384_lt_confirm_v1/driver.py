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
import random
name='qkv'; T=16384; wn,f32=S.LT_GEMMS[name]; W=st[wn]; N,K=W.shape
plan=st['lt'].lt_plan(T,N,K,f32,st['lt_ws'].numel(),32)
sigs={'plan24':(24,1,0,7),'plan184':(184,1,0,2)}
choices={name:S._lt_pin_index(st['lt'],plan,sig) for name,sig in sigs.items()}
assert all(v is not None and v>=0 for v in choices.values()),choices
print('PLAN',json.dumps({'parent':str(st['lt_choice'].get((T,name))),'candidates':choices,'signatures':sigs}),flush=True)
for seed in (411, 7919, 22073):
    torch.manual_seed(seed); x=torch.randn((T,K),device='cuda',dtype=torch.bfloat16)
    states={'base1':st,'base2':st}
    for label,idx in choices.items():
        q=dict(st);q['lt_choice']=dict(st['lt_choice']);q['lt_choice'][(T,name)]=(plan,idx);states[label]=q
    graphs={};outs={}
    for label,state in states.items(): graphs[label],outs[label]=capture(lambda state=state:S._mm(state,x,name))
    for label in ('base2',*choices):
        err=S._lt_rel_err(outs[label],outs['base1'])
        if err >= S.LT_ALGO_TOL: raise AssertionError((label,err))
        print('CHECK',json.dumps({'seed':seed,'variant':label,'max_relative':err}),flush=True)
    vals={label:[] for label in states}; ratios={label:[] for label in states if label!='base1'}
    rng=random.Random(seed)
    for rep in range(20):
        order=list(states);rng.shuffle(order); round_times={label:elapsed(graphs[label]) for label in order}
        for label,t in round_times.items(): vals[label].append(t)
        for label in ratios: ratios[label].append(round_times['base1']/round_times[label])
    print('RESULT',json.dumps({'seed':seed,'medians_us':{k:statistics.median(v) for k,v in vals.items()},'paired_ratios':{k:statistics.median(v) for k,v in ratios.items()},'samples':ratios}),flush=True)
    del graphs,outs,x,states
print('DONE',flush=True)
