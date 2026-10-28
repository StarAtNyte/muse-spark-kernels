import importlib.util, json, statistics, sys, torch
from unittest.mock import MagicMock
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        module=MagicMock();module.__path__=[];module.__name__=spec.name;module.__spec__=spec;return module
    def exec_module(self, module): pass
sys.meta_path.insert(0,StubSGLang())
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
base=load('base','/w/kernels/muse_moe_w4a16/submission.py')
cand=load('cand','/w/run/sub.py')
cand._om_IL_M=(8,32)
weights=R.generate_weights(R.WEIGHTS_SEED)
st0=base.prepare(weights);st1=cand.prepare(weights)
print('GPU',torch.cuda.get_device_name(),flush=True)
flush=torch.empty(256*1024*1024//4,device='cuda')
def capture(mod,state,data):
    out=mod.kernel_fn(data,state);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=mod.kernel_fn(data,state)
    graph.replay();torch.cuda.synchronize();return graph,out
def elapsed(graph):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for seed in (419,7919,15241):
    data=R.generate_input(32,'ep8',seed)
    g0,o0=capture(base,st0,data);g1,o1=capture(cand,st1,data)
    ref=R.ref_kernel(data,weights)
    print('DIAG',json.dumps(dict(local_ids=int((data['topk_ids']<32).sum().item()),ref_norm=ref.float().norm().item(),base_norm=o0.float().norm().item(),cand_norm=o1.float().norm().item(),base_max=o0.float().abs().max().item(),cand_max=o1.float().abs().max().item())),flush=True)
    ok,msg=R.check_implementation(data,o1,weights)
    err=((o1.float()-o0.float()).norm()/o0.float().norm().clamp_min(1e-20)).item()
    print('CHECK',json.dumps(dict(seed=seed,ok=ok,message=msg,relative_error_vs_base=err,bit_equal=torch.equal(o0,o1))),flush=True)
    if not ok: continue
    ratios=[];t0=[];t1=[]
    for i in range(32):
        if i%2:
            b=elapsed(g0);c=elapsed(g1)
        else:
            c=elapsed(g1);b=elapsed(g0)
        t0.append(b);t1.append(c);ratios.append(b/c)
    print('RESULT',json.dumps(dict(seed=seed,base_us=statistics.median(t0),candidate_us=statistics.median(t1),speedup=statistics.median(ratios))),flush=True)
print('DONE',flush=True)
