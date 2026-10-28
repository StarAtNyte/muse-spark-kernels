"""Separate row/K warp ownership; preserve NVFP4 and the full routed pipeline."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
import ast
sp=importlib.util.spec_from_file_location('wide_bundle','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
open('/w/run/wide_parent.py','w').write(m.PARENT)
sp=importlib.util.spec_from_file_location('wide_parent','/w/run/wide_parent.py');sub=importlib.util.module_from_spec(sp);sys.modules['wide_parent']=sub;sp.loader.exec_module(sub)
t=ast.parse(m.PARENT)
n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_finish_experts')
exec(ast.get_source_segment(m.PARENT,n).replace('if tokens >= 32:','if False:'),sub.__dict__)
n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
fn=ast.get_source_segment(m.PARENT,n).replace('def kernel_fn(','def _wide_forward(')
for mode,ins,gs,x,y in [(0,'w13','gs13','x','inter'),(1,'w2','gs2','inter','y')]:
    sf='s13' if mode==0 else 's2'
    old=f'    mod.wm_gemm_run({mode}, nch, nt'+('1' if mode else '')+f', nw, ring, _WM_DBG, wm["{ins}"], wm["{sf}"], wm["{gs}"], {x}, {y}, cnt, lst, act, nact, woff, maxslots, lmax)'
    assert old in fn,old
    new=f'    state["wide_mod"].wm_wide_run({mode},state["wide_wk"],state["wide_cap"],wm["{ins}"],wm["{sf}"],wm["{gs}"],{x},{y},lst,act,nact,lmax)'
    fn=fn.replace(old,new)
exec(fn,sub.__dict__)
sys.path.insert(0,'/w/modal');from compile_split import build
flags=['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a']
funcs=['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf']
base_mod=build('muse_wide_base',m.BASE_CPP,m.BASE_CUDA,funcs,flags)
wide_mod=build('muse_wide_row_k',m.CPP,m.CUDA,['wide_set_pdl','wm_wide_run'],flags)
base_mod.wm_set_pdl_t(1);wide_mod.wide_set_pdl(1)
sub._wm_module=lambda:base_mod
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights);state['wide_mod']=wide_mod
sms=torch.cuda.get_device_properties(0).multi_processor_count
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(data,wk):
    state['wide_wk']=wk;state['wide_cap']=sms*(16//wk) if wk else sms
    fn=sub._wide_forward if wk else sub.kernel_fn
    fn(data,state);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn(data,state)
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for seed in (419,7919,15241):
    for bs in (1,8):
        data=R.generate_input(bs,'ep8',seed);gb,ob=capture(data,0)
        for wk in (4,8,16):
            g,o=capture(data,wk);ok,msg=R.check_implementation(data,o,weights)
            err=((o.float()-ob.float()).norm()/ob.float().norm()).item()
            print('CHECK',bs,seed,wk,ok,msg,'base_error',err,flush=True);assert ok,msg
            bt,ct,ratios=[],[],[]
            for rep in range(24):
                if rep%2:c,b=elapsed(g),elapsed(gb)
                else:b,c=elapsed(gb),elapsed(g)
                bt.append(b);ct.append(c);ratios.append(b/c)
            print('RESULT',json.dumps(dict(batch=bs,seed=seed,wk=wk,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
            del g,o
        del gb,ob,data
print('MOE_WIDE_SCREEN_PASSED',flush=True)
