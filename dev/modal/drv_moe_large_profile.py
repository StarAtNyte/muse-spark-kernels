# Remove corrupt bytecode in disposable diagnostic container before Triton import.
from pathlib import Path
for cached in Path('/usr/local/lib/python3.12/site-packages/triton').rglob('*.pyc'):
    cached.unlink(missing_ok=True)
"""Diagnostic ablations only. Incorrect ablated outputs are never candidates."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
import ast
sp=importlib.util.spec_from_file_location('bundle','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
open('/w/run/frozen_parent.py','w').write(m.PARENT)
sp=importlib.util.spec_from_file_location('frozen_parent','/w/run/frozen_parent.py');sub=importlib.util.module_from_spec(sp);sys.modules['frozen_parent']=sub;sp.loader.exec_module(sub)
t=ast.parse(m.PARENT);n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_finish_experts')
exec(ast.get_source_segment(m.PARENT,n).replace('if tokens >= 32:','if False:'),sub.__dict__)
sys.path.insert(0,'/w/modal');from compile_split import build
mod=build('moe_large_profile_v1',m.CPP,m.CUDA,['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf'],['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a'])
mod.wm_set_pdl_t(1);sub._wm_module=lambda:mod
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights);wm=state['wm'];flush=torch.zeros(256*1024*1024//4,device='cuda')
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000



data=R.generate_input(256,'ep8',419)
sub.kernel_fn(data,state);torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):out=sub.kernel_fn(data,state)
g.replay();torch.cuda.synchronize()
ok,msg=R.check_implementation(data,out,weights);print('CHECK',ok,msg,flush=True);assert ok,msg
flush.add_(1);torch.cuda.synchronize()
torch.cuda.profiler.start();g.replay();torch.cuda.synchronize();torch.cuda.profiler.stop()
print('LARGE_MOE_PROFILE_COMPLETE',flush=True)
