"""Paired H200 test of 128-key full-verify B32 against 64-key public source."""
import importlib.util,json,statistics,sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('sglang','sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self,module):pass
sys.meta_path.insert(0,StubSGLang());sys.path.insert(0,'/w/kernels/muse_attention_decode')
import reference as R
sp=importlib.util.spec_from_file_location('cand','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
flush=torch.empty(256*1024*1024//4,device='cuda',dtype=torch.int32)
def cap(fn):
    out=fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
def paired(a,b,reps=20):
    rat=[];at=[];bt=[]
    for i in range(reps):
        if i&1:y,x=elapsed(b),elapsed(a)
        else:x,y=elapsed(a),elapsed(b)
        rat.append(x/y);at.append(x);bt.append(y)
    return statistics.median(rat),statistics.median(at),statistics.median(bt)
def gen(seed):return R.generate_input(32,80000,'tp8','full',seed,q_len=8,spread='agentic',dist='coherent',page=1)
for seed in (56321,881,3491):
    for tile,fn in ((64,lambda d:m._launch(d,_mod=m._module())),(128,lambda d:m._launch(d,_mod=m._module_tile128()))):
        d=gen(seed);o=fn(d);torch.cuda.synchronize();ok,msg=R.check_implementation(d,R.collect(d,o))
        if not ok:raise AssertionError((f'tile{tile} direct reference',seed,msg))
    d64=gen(seed);d128=gen(seed);plan=m._plan(32,1,8,0,d64['page_table'].shape[1],1,d64['qkv'].device)
    g64,o64=cap(lambda:m._launch(d64,_mod=m._module(),**plan))
    g128,o128=cap(lambda:m._launch(d128,_mod=m._module_tile128(),**plan))
    ratio,a,b=paired(g64,g128)
    print('RESULT',json.dumps({'seed':seed,'tile64_us':a,'tile128_us':b,'speedup':ratio}),flush=True)
    del d64,d128,g64,g128,o64,o128
print('TILE128_SCREEN_PASSED',torch.cuda.get_device_name(),flush=True)
