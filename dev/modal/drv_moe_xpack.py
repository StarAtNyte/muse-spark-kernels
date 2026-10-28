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
mod=build('moe_xpack_v1',m.CPP,m.CUDA,['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf','wm_pack_run'],['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a'])
mod.wm_set_pdl_t(1);sub._wm_module=lambda:mod
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights);wm=state['wm'];flush=torch.zeros(256*1024*1024//4,device='cuda')
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000


n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
src=ast.get_source_segment(m.PARENT,n).replace('def kernel_fn(','def packed_forward(')
for mode,x,y,wp,sf,gs,nt in [(0,'x','inter','w13','s13','gs13','nt'),(1,'inter','y','w2','s2','gs2','nt1')]:
 old=f'    mod.wm_gemm_run({mode}, nch, {nt}, nw, ring, _WM_DBG, wm["{wp}"], wm["{sf}"], wm["{gs}"], {x}, {y}, cnt, lst, act, nact, woff, maxslots, lmax)'
 new=f'    xp={x}\n    dbg=0\n    if state["pack"][{mode}]:\n        xp=torch.empty((maxslots,64,256,8),dtype=x.dtype,device=dev)\n        mod.wm_pack_run({mode},{x},xp,lst,act,maxslots,lmax)\n        dbg=128\n    mod.wm_gemm_run({mode}, nch, {nt}, nw, ring, dbg, wm["{wp}"], wm["{sf}"], wm["{gs}"], xp, {y}, cnt, lst, act, nact, woff, maxslots, lmax)'
 assert old in src;src=src.replace(old,new)
exec(src,sub.__dict__)
def capture(data,pack):
 state['pack']=pack;sub.packed_forward(data,state);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=sub.packed_forward(data,state)
 g.replay();torch.cuda.synchronize();return g,out
for bs in (128,256):
 for seed in (419,7919,15241):
    data=R.generate_input(bs,'ep8',seed);gb,ob=capture(data,(0,0))
    for pack in ((1,1),(1,0),(0,1)):
        g,o=capture(data,pack);ok,msg=R.check_implementation(data,o,weights);equal=torch.equal(o,ob)
        print('CHECK',bs,seed,pack,ok,msg,'bit_equal',equal,flush=True);assert ok and equal,msg
        bt=[];ct=[]
        for rep in range(24):
            if rep%2:c,b=elapsed(g),elapsed(gb)
            else:b,c=elapsed(gb),elapsed(g)
            bt.append(b);ct.append(c)
        print('RESULT',json.dumps(dict(batch=bs,seed=seed,pack=pack,baseline_us=statistics.median(bt),candidate_us=statistics.median(ct),ratio=statistics.median(bt)/statistics.median(ct))),flush=True)
print('XPACK_SCREEN_PASSED',flush=True)
