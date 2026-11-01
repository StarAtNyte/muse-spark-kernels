"""Whole-layer screen of pooler SM targets and router overlap."""
import importlib.util, json, statistics, sys
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
sys.path.insert(0, '/w/kernels/muse_prefill_layer')
import reference as R
def load(name):
    sp = importlib.util.spec_from_file_location(name, '/w/run/sub.py'); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base')
B.LT_TOKENS=(753,3012)
w = R.generate_weights(R.WEIGHTS_SEED)
sb = B.prepare(w); sb['attn_choice'] = dict(B.ATTN_FIXED)
print('BASE_PLANS',str(sb.get('lt_choice')),flush=True)
print('SHARED_PLANS', flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, st, d):
    M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, out
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
import random
for i,c in enumerate((dict(batch=1,ctx=80000,q_len=753,layer='sliding'),dict(batch=4,ctx=80000,q_len=753,layer='full'))):
    T=c['batch']*c['q_len']; n,k=sb['pooler'].shape
    states={'base1':sb,'base2':sb}
    for reserve in (16,32,48):
        pid=sb['lt'].lt_plan_sm(T,n,k,False,sb['lt_ws'].numel(),4,sb['nsm']-reserve)
        if sb['lt'].lt_nalgo(pid)==0:
            print('NO_PLAN',T,reserve,flush=True);continue
        sc=dict(sb);sc['lt_choice']=dict(sb['lt_choice']);sc['lt_choice'][(T,'pooler')]=(pid,0)
        states[f'reserve{reserve}']=sc
        print('PLAN',T,reserve,list(sb['lt'].lt_info(pid,0)),flush=True)
    for seed in (9400+i, 9420+i):
        cc=dict(page=1,dist='soft',seed=seed,**c)
        if hasattr(R,'case_fields'): cc.update(R.case_fields(cc,w))
        d=R.generate_input(**cc);graphs={};outs={}
        for name,st in states.items(): graphs[name],outs[name]=cap(B,st,d)
        for name,out in outs.items():
            ok,msg=R.check_implementation(d,R.collect(d,out),w)
            print('CHECK',T,seed,name,ok,str(msg)[:140],flush=True)
            if not ok: raise RuntimeError((name,msg))
        vals={k:[] for k in graphs};rat={k:[] for k in graphs if k!='base1'};rng=random.Random(seed)
        for rep in range(12):
            order=list(graphs);rng.shuffle(order); row={k:t1(graphs[k]) for k in order}
            for k,v in row.items(): vals[k].append(v)
            for k in rat: rat[k].append(row['base1']/row[k])
        print('RESULT',json.dumps({'T':T,'seed':seed,'case':c,'medians_us':{k:statistics.median(v) for k,v in vals.items()},'paired_ratios':{k:statistics.median(v) for k,v in rat.items()},'samples':rat}),flush=True)
        del graphs,outs,d
print('DONE',flush=True)
