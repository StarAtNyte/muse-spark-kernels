"""Whole-decoder paired screen: scored source (/w/kernels) vs candidate (/w/run/sub.py) on given shapes.
Each output reference-checked; alternating cold-L2 graph replays. Shapes from env CASES (json list) or defaults."""
import importlib.util, json, os, statistics, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base', '/w/kernels/muse_decode_layer/submission.py')
C = load('cand', '/w/run/sub.py')
w = R.generate_weights(R.WEIGHTS_SEED)
sb = B.prepare(w); base_ext = B._att7(); cand_ext = C._att7()
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, d, st):
    o = M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
import random
cases=[dict(batch=b,q_len=1,layer=l) for b,l in ((1,'sliding'),(16,'sliding'),(16,'full'))]
for i,c in enumerate(cases):
    for seed in (4001,4002,4021):
        cc=dict(ctx=80000,page=1,dist='soft',spread=('agentic' if c['batch']>1 else 'fixed'),seed=seed+10*i,**c)
        ls=R.case_fields(cc,w)['local_start'];d=R.generate_input(**cc,local_start=ls)
        graphs={};outs={}
        B._ATT7=base_ext
        for name in ('base1','base2'):graphs[name],outs[name]=cap(B,d,sb)
        B._ATT7=cand_ext;graphs['candidate'],outs['candidate']=cap(B,d,sb);B._ATT7=base_ext
        for name,out in outs.items():
            ok,msg=R.check_implementation(d,R.collect(d,out),w)
            print('CHECK',c,seed,name,ok,str(msg)[:120],flush=True)
            if not ok:raise RuntimeError((c,seed,name,msg))
        vals={k:[] for k in graphs};rat={k:[] for k in graphs if k!='base1'};rng=random.Random(seed+i)
        for rep in range(24):
            order=list(graphs);rng.shuffle(order);row={k:t1(graphs[k]) for k in order}
            for k,v in row.items():vals[k].append(v)
            for k in rat:rat[k].append(row['base1']/row[k])
        print('RESULT',json.dumps({'case':c,'seed':seed,'medians_us':{k:statistics.median(v) for k,v in vals.items()},'paired_ratios':{k:statistics.median(v) for k,v in rat.items()},'samples':rat}),flush=True)
        del d,graphs,outs
print('DONE',flush=True)
