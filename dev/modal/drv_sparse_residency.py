"""Test register-spill relief in the exact packed-head decode prototype."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
from torch.utils.cpp_extension import load_inline
variants={
 'loose':candidate._SP_CUDA.replace('__launch_bounds__(128,3)','__launch_bounds__(128,1)'),
 'memory_fence':candidate._SP_CUDA.replace('"r"(e));','"r"(e):"memory");').replace('"r"(0xEEEE4444u));','"r"(0xEEEE4444u):"memory");')}
modules={}
for name,src in variants.items():
    modules[name]=load_inline('muse_sparse_resource_'+name,candidate._SP_CPP,cuda_sources=src,
        functions=['sparse_decode_run'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
for layer in ('sliding','full'):
    for batch in (1,16,32):
        for seed in (539,1427):
            data=R.generate_input(batch,80000,'tp8',layer,seed,page=1,spread='agentic',
                dist='soft' if layer=='sliding' else 'coherent')
            gb,ob=capture(base,data)
            for name,mod in modules.items():
                candidate._SP_MOD=mod
                candidate._SP_SPLITS_OVERRIDE=(32 if batch==1 else 16 if batch==16 else 8) if layer=='sliding' else (256 if batch==1 else 128 if batch==16 else 64)
                gc,oc=capture(candidate,data)
                err=((ob.float()-oc.float()).norm()/ob.float().norm()).item()
                if not torch.isfinite(oc).all().item() or err>.003:raise AssertionError((layer,batch,seed,name,err))
                bt,ct,ratios=[],[],[]
                for rep in range(16):
                    if rep%2:c,b=elapsed(gc),elapsed(gb)
                    else:b,c=elapsed(gb),elapsed(gc)
                    bt.append(b);ct.append(c);ratios.append(b/c)
                print('RESULT',json.dumps(dict(variant=name,layer=layer,batch=batch,seed=seed,
                    error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),
                    candidate_us=statistics.median(ct))),flush=True)
                del gc,oc
            del gb,ob,data
