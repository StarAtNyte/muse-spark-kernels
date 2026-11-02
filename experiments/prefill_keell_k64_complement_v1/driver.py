"""Screen K64 permutation transfer on the current prefill baseline."""
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
# Use the exact current best source as parent and the patched source as candidate.
sp=importlib.util.spec_from_file_location('base','/w/kernels/muse_prefill_layer/submission.py')
B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
C=load('cand')
w=R.generate_weights(R.WEIGHTS_SEED)
sb=B.prepare(w);sb['attn_choice']=dict(B.ATTN_FIXED)
sc=dict(sb);sc['mod']=C._build(C._fast_scales_ok(w));sc['mod'].moe_init(sc['smem']);sc['mod'].attn_init(sc['attn_smem'])
sc['w13r']=C._repack(w['w13'],w['w13_sf'],True);sc['w2r']=C._repack(w['w2'],w['w2_sf'],False)
C.SIDE_STREAM=B.SIDE_STREAM
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
cases=(dict(batch=1,ctx=80000,q_len=753,layer='full'),dict(batch=4,ctx=80000,q_len=753,layer='sliding'))
for i,c in enumerate(cases):
    for seed in (9400+i,9417+i,9429+i):
        cc=dict(page=1,dist='soft',seed=seed,**c)
        if hasattr(R,'case_fields'):cc.update(R.case_fields(cc,w))
        d=R.generate_input(**cc);graphs={};outs={}
        for name,M,st in [('base1',B,sb),('base2',B,sb),('candidate',C,sc)]:graphs[name],outs[name]=cap(M,st,d)
        for name,out in outs.items():
            ok,msg=R.check_implementation(d,R.collect(d,out),w)
            print('CHECK',seed,name,ok,str(msg)[:140],flush=True)
            if not ok:raise RuntimeError((name,msg))
        vals={k:[] for k in graphs};rat={k:[] for k in graphs if k!='base1'};rng=random.Random(seed)
        for rep in range(16):
            order=list(graphs);rng.shuffle(order);row={k:t1(graphs[k]) for k in order}
            for k,v in row.items():vals[k].append(v)
            for k in rat:rat[k].append(row['base1']/row[k])
        print('RESULT',json.dumps({'seed':seed,'case':c,'medians_us':{k:statistics.median(v) for k,v in vals.items()},'paired_ratios':{k:statistics.median(v) for k,v in rat.items()},'samples':rat}),flush=True)
        del graphs,outs,d
print('DONE',flush=True)
