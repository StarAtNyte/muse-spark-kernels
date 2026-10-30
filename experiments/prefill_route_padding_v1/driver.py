"""Measure real routed token counts and conditional tile-padding opportunities."""
import importlib.util, json, sys
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
spec=importlib.util.spec_from_file_location('candidate','/w/run/sub.py'); S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w=R.generate_weights(R.WEIGHTS_SEED); st=S.prepare(w); st['attn_choice']=dict(S.ATTN_FIXED)
def tile(r, fine):
    if not r:return 0
    if r<=32:return 32
    if r<=64:return 64
    if not fine:return 128 if r<=128 else 256
    for n in (128,160,192,256):
        if r<=n:return n
def plan(c, fine):
    n,r=divmod(c,256)
    if not fine and 128<r<=192:return 256*n+128+tile(r-128,False)
    return 256*n+tile(r,fine)
for i,c in enumerate((dict(batch=1,ctx=80000,q_len=753,layer='sliding'),dict(batch=4,ctx=80000,q_len=753,layer='full'),dict(batch=1,ctx=32768,q_len=16384,layer='sliding'))):
    cc=dict(page=1,dist='soft',seed=9400+i,**c)
    if hasattr(R,'case_fields'):cc.update(R.case_fields(cc,w))
    d=R.generate_input(**cc); out=S.kernel_fn(d,st); torch.cuda.synchronize()
    ok,msg=R.check_implementation(d,R.collect(d,out),w);print('CHECK',ok,str(msg)[:200],flush=True)
    if not ok:raise RuntimeError('reference check failed')
    ids=out[2].flatten().to(torch.int64); start=int(d['local_start']); local=ids[(ids>=start)&(ids<start+32)]-start
    counts=torch.bincount(local,minlength=32).cpu().tolist(); fine=c['batch']*c['q_len']>=S.FINE_MIN_T
    current=sum(plan(n,fine) for n in counts); ideal=sum(((n+31)//32)*32 for n in counts)
    print('PADDING',json.dumps(dict(case=c,seed=9400+i,local_start=start,counts=counts,actual=sum(counts),padded=current,ideal_32=ideal,excess=current-ideal,fine=fine)),flush=True)
print('DONE',flush=True)
