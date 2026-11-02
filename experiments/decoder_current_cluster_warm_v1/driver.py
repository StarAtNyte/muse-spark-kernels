"""Reference-check and time the decoder's attention module with cluster warmup."""
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
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
B=load('base','/w/kernels/muse_decode_layer/submission.py');C=load('candidate','/w/run/sub.py')
B._att7();C._att7();print('READY',torch.cuda.get_device_name(),flush=True)
flush=torch.empty(256*1024*1024//4,dtype=torch.int32,device='cuda')
def cap(M,d):
    out=M._launch7(d);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=M._launch7(d)
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for batch,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
    for seed in (419,7919):
        d=R.generate_input(batch,80000,'tp8','sliding',seed,q_len=q,spread='agentic' if batch>1 else 'fixed',dist='soft',page=1)
        B._inv_freq(d['config']['rope_theta'],d['qkv'].device);C._inv_freq(d['config']['rope_theta'],d['qkv'].device)
        gb,ob=cap(B,d);gc,oc=cap(C,d)
        for name,out in [('base',ob),('candidate',oc)]:
            ok,msg=R.check_implementation(d,R.collect(d,out));print('CHECK',batch,q,seed,name,ok,str(msg)[:120],flush=True)
            if not ok:raise RuntimeError((batch,q,seed,name,msg))
        bt=[];ct=[];rat=[]
        for i in range(20):
            if i&1:c,b=elapsed(gc),elapsed(gb)
            else:b,c=elapsed(gb),elapsed(gc)
            bt.append(b);ct.append(c);rat.append(b/c)
        print('RESULT',json.dumps({'batch':batch,'q_len':q,'seed':seed,'base_us':statistics.median(bt),'candidate_us':statistics.median(ct),'paired_ratio':statistics.median(rat)}),flush=True)
        del d,gb,gc,ob,oc
print('DONE',flush=True)
