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
    for m in (7,17,63,127,129,200,300):
        d=R.generate_input(m,layer,9000+m)
        try:
            out=S.kernel_fn(d,st);torch.cuda.synchronize();ok,msg=R.check_implementation(d,out,w)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out=S.kernel_fn(d,st)
            g.replay();torch.cuda.synchronize();ts=[]
            for i in range(8):
                flush.sum();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000)
            print('ROW',json.dumps({'layer':layer,'m':m,'ok':ok,'detail':str(msg)[:160],'us':round(statistics.median(ts),3)}),flush=True)
        except Exception as e:
            print('ROW',json.dumps({'layer':layer,'m':m,'error':str(e)[:180]}),flush=True)
print('DONE',flush=True)
