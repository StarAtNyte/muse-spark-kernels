"""Narrow-column WGMMA versus the production warp-MMA MoE, full pipeline."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
import ast
sp=importlib.util.spec_from_file_location('wg_bundle','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
open('/w/run/wg_parent.py','w').write(m.PARENT)
sp=importlib.util.spec_from_file_location('wg_parent','/w/run/wg_parent.py');sub=importlib.util.module_from_spec(sp);sys.modules['wg_parent']=sub;sp.loader.exec_module(sub)
tree=ast.parse(m.PARENT)
n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_finish_experts')
exec(ast.get_source_segment(m.PARENT,n).replace('if tokens >= 32:','if False:'),sub.__dict__)
n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
forward=ast.get_source_segment(m.PARENT,n).replace('def kernel_fn(', 'def _wg_forward(')
forward=forward.replace('    x = x.contiguous()', '    x = x.contiguous()\n    xp=torch.empty_like(x);mod.wm_permx(x,xp);x=xp')
assert 'mod.wm_permx(x,xp)' in forward
exec(forward,sub.__dict__)
sys.path.insert(0,'/w/modal');from compile_split import build
flags=['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a']
funcs=['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf']
base_mod=build('muse_wgn_base',m.CPP,m.BASE_CUDA,funcs,flags)
wg_mod=build('muse_wgn_pipeline',m.CPP,m.PIPELINE,funcs+['wm_permx'],flags)
for mod in (base_mod,wg_mod):mod.wm_set_pdl_t(1)
sub._wm_module=lambda:base_mod
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights);base_cfg=sub._WM_CFG
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(data,stages):
    state['wm']['mod']=base_mod if not stages else wg_mod
    sub._WM_CFG=base_cfg if not stages else [(10**9,(stages,4,7))]
    fn=sub.kernel_fn if not stages else sub._wg_forward
    fn(data,state);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn(data,state)
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for seed in (419,7919):
    for bs in (1,8,16,32):
        data=R.generate_input(bs,'ep8',seed);gb,ob=capture(data,0)
        for stages in (3,4,6):
            g,o=capture(data,stages);ok,msg=R.check_implementation(data,o,weights)
            err=((o.float()-ob.float()).norm()/ob.float().norm()).item()
            print('CHECK',bs,seed,stages,ok,msg,'base_error',err,flush=True);assert ok,msg
            bt,ct,ratios=[],[],[]
            for rep in range(20):
                if rep%2:c,b=elapsed(g),elapsed(gb)
                else:b,c=elapsed(gb),elapsed(g)
                bt.append(b);ct.append(c);ratios.append(b/c)
            print('RESULT',json.dumps(dict(batch=bs,seed=seed,stages=stages,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
            del g,o
        del gb,ob,data
print('MOE_WGPIPELINE_SCREEN_PASSED',flush=True)
