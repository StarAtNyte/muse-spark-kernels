"""Check correctness and record graph latency on non-scored projection sizes."""
import importlib.util, json, statistics, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('sglang','sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self,m): pass
sys.meta_path.insert(0,StubSGLang())
sys.path.insert(0,'/w/kernels/muse_bf16_proj')
import reference as R
spec=importlib.util.spec_from_file_location('sub','/w/run/sub.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
w=R.generate_weights(R.WEIGHTS_SEED);st=S.prepare(w)
flush=torch.zeros(256*1024*1024//4,device='cuda')
for layer in ('qkv_gate','o','pooler','unpooler','router'):
    for m in (7,9,15,17,31,33,63,127,129,200,255,300):
        d=R.generate_input(m,layer,9000+m)
        try:
            S._ctx_bind(d['x'])
            base=S._K[S._DEFAULT]; bs=st[S._DEFAULT]
            def parent(): return base['kernel_fn'](d,bs)
            def candidate(): return S.kernel_fn(d,st)
            graphs={}; checks={}
            for name,fn in (('base',parent),('cand',candidate)):
                out=fn();torch.cuda.synchronize();checks[name]=R.check_implementation(d,out,w)[0]
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): out=fn()
                g.replay();torch.cuda.synchronize();graphs[name]=g
            times={k:[] for k in graphs}
            for i in range(8):
                for name in (('base','cand') if i%2==0 else ('cand','base')):
                    flush.sum();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graphs[name].replay();b.record();b.synchronize();times[name].append(a.elapsed_time(b)*1000)
            x,y=(statistics.median(times[k]) for k in ('base','cand'))
            print('ROW',json.dumps({'layer':layer,'m':m,'correct':checks,'base_us':round(x,3),'cand_us':round(y,3),'gain':round(x/y,4)}),flush=True)
        except Exception as e: print('ROW',json.dumps({'layer':layer,'m':m,'error':str(e)[:180]}),flush=True)
print('DONE',flush=True)
