"""End-to-end paired screen of exact scale-decoding alternatives."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
import ast,types
spec=importlib.util.spec_from_file_location('bundle','/w/run/sub.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
open('/w/run/scale_parent.py','w').write(m.PARENT)
parent_spec=importlib.util.spec_from_file_location('scale_parent','/w/run/scale_parent.py')
sub=importlib.util.module_from_spec(parent_spec);sys.modules['scale_parent']=sub;parent_spec.loader.exec_module(sub)
node=next(n for n in ast.parse(m.PARENT).body if isinstance(n,ast.FunctionDef) and n.name=='_finish_experts')
exec(ast.get_source_segment(m.PARENT,node).replace('if tokens >= 32:','if False:'),sub.__dict__)
sys.path.insert(0,'/w/modal');from compile_split import build
functions=['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf']
flags=['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a']
mods={}
for name,src in [('base',m.BASE),('prmt',m.PRMT),('expanded',m.EXPANDED),('expanded_m2',m.EXPANDED)]:
    mods[name]=build('muse_scale_'+name,m.CPP,src,functions,flags+(['-DWM_MINB=2'] if name=='expanded_m2' else []))
    mods[name].wm_set_pdl_t(1)
    print('COMPILED',name,flush=True)
sub._wm_module=lambda:mods['base']
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights)
original=(state['wm']['s13'],state['wm']['s2'])
expanded=tuple((s.to(torch.int32)*16+0x7600).to(torch.int16).contiguous().view(torch.uint8) for s in original)
for src,dst in zip(original,expanded):
    assert torch.equal(dst.view(torch.int16).to(torch.int32),src.to(torch.int32)*16+0x7600)
print('LOSSLESS_SCALE_REPACK_PASS',flush=True)
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(data,name):
    state['wm']['mod']=mods[name]
    sf=expanded if name.startswith('expanded') and data['hidden_states'].shape[0]<=32 else original
    state['wm']['s13'],state['wm']['s2']=sf
    sub.kernel_fn(data,state);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=sub.kernel_fn(data,state)
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for seed in (419,7919):
    for bs in (1,8,16,32,128,256):
        data=R.generate_input(bs,'ep8',seed)
        gb,ob=capture(data,'base')
        for name in ('prmt','expanded','expanded_m2'):
            if name.startswith('expanded') and bs>32:continue
            g,o=capture(data,name)
            ok,msg=R.check_implementation(data,o,weights)
            equal=torch.equal(o,ob)
            print('CHECK',name,bs,seed,ok,msg,'bit_identical',equal,flush=True)
            assert ok and equal,(name,bs,msg)
            bt,ct,ratios=[],[],[]
            for rep in range(24):
                if rep%2:c,b=elapsed(g),elapsed(gb)
                else:b,c=elapsed(gb),elapsed(g)
                bt.append(b);ct.append(c);ratios.append(b/c)
            print('RESULT',json.dumps(dict(variant=name,batch=bs,seed=seed,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
            del g,o
        del gb,ob,data
print('MOE_SCALE_SCREEN_PASSED',flush=True)
