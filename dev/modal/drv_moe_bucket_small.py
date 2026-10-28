"""Count-specialized M17..32 kernels versus production and persistence control."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
import ast
sp=importlib.util.spec_from_file_location('small_bundle','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
open('/w/run/small_parent.py','w').write(m.PARENT)
sp=importlib.util.spec_from_file_location('small_parent','/w/run/small_parent.py');sub=importlib.util.module_from_spec(sp);sys.modules['small_parent']=sub;sp.loader.exec_module(sub)
n=next(n for n in ast.parse(m.PARENT).body if isinstance(n,ast.FunctionDef) and n.name=='_finish_experts')
exec(ast.get_source_segment(m.PARENT,n).replace('if tokens >= 32:','if False:'),sub.__dict__)
sys.path.insert(0,'/w/modal');from compile_split import build
flags=['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a']
funcs=['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf']
base_mod=build('muse_small_base',m.BASE_CPP,m.BASE_CUDA,funcs,flags)
async_mod=build('muse_small_async',m.CPP,m.CUDA,funcs+['wm_permx','wm_bucket_run'],flags)
direct_mod=build('muse_small_direct',m.CPP,m.CUDA_DIRECT,funcs+['wm_permx','wm_bucket_run'],flags)
for mod in (base_mod,async_mod,direct_mod):mod.wm_set_pdl_t(1)
sub._wm_module=lambda:base_mod
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights);exec(m.FORWARD,sub.__dict__)
state['_bucket_streams']=[torch.cuda.Stream() for _ in range(3)]
sms=torch.cuda.get_device_properties(0).multi_processor_count
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(data,name):
    base_mod.wm_set_persist_t(4 if name=='persistent' else 0)
    state['wm']['mod']=base_mod if name in ('base','persistent') else direct_mod if name.startswith('direct') else async_mod
    c1,c2,c3=(6,4,3) if name.endswith('high') else (4,2,2)
    state['_bucket_caps']=(sms*c1,sms*c2,sms*c3)
    fn=sub.kernel_fn if name in ('base','persistent') else sub._bucket_forward
    fn(data,state);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn(data,state)
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
cases=[(32,'normal',s) for s in (101,419,2387,7919,15241,49157)]+[(17,'normal',173),(24,'normal',2387),(32,'skewed',1487)]
for bs,dist,seed in cases:
    data=R.generate_input(bs,'ep8',seed,dist=dist);gb,ob=capture(data,'base')
    for name in ('persistent','async_low','async_high','direct_low','direct_high'):
        g,o=capture(data,name);ok,msg=R.check_implementation(data,o,weights)
        equal=torch.equal(o,ob)
        print('CHECK',name,bs,dist,seed,ok,msg,'bit_identical',equal,flush=True);assert ok and equal,msg
        bt,ct,ratios=[],[],[]
        for rep in range(20):
            if rep%2:c,b=elapsed(g),elapsed(gb)
            else:b,c=elapsed(gb),elapsed(g)
            bt.append(b);ct.append(c);ratios.append(b/c)
        print('RESULT',json.dumps(dict(variant=name,batch=bs,dist=dist,seed=seed,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
        del g,o
    del gb,ob,data
print('MOE_SMALL_BUCKET_SCREEN_PASSED',flush=True)
