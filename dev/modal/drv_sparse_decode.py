"""Reference-first check and paired benchmark for exact sparse-head decode."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
# Fail early before compiling the baseline's large CUDA module.
for batch,ctx,layer,page,scale,dist in ((1,3000,'full',1,1.,'coherent'),
    (2,6000,'sliding',64,1/64,'edges'),(3,9000,'full',64,1/64,'soft')):
    data=R.generate_input(batch,ctx,'tp8',layer,431+batch,page=page,kv_scale=scale,dist=dist)
    out=candidate.kernel_fn(data)
    ok,message=R.check_implementation(data,R.collect(data,out))
    print('EARLY_CHECK',batch,ctx,layer,page,scale,ok,message,flush=True)
    if not ok: raise AssertionError(message)
for layer in ('sliding','full'):
    for batch in (1,16,32):
        options=((16,32,64) if batch==1 else (4,8,16)) if layer=='sliding' else ((128,256,512) if batch==1 else (32,64,128) if batch==16 else (16,32,64))
        for seed in (539,1427):
            data=R.generate_input(batch,80000,'tp8',layer,seed,page=1,spread='agentic',
                dist='soft' if layer=='sliding' else 'coherent')
            gb,ob=capture(base,data)
            for splits in options:
                candidate._SP_SPLITS_OVERRIDE=splits
                gc,oc=capture(candidate,data)
                err=((ob.float()-oc.float()).norm()/ob.float().norm()).item()
                if not torch.isfinite(oc).all().item() or err>.003:
                    raise AssertionError((layer,batch,seed,splits,err))
                ratios,bt,ct=[],[],[]
                for rep in range(16):
                    if rep%2:c,b=elapsed(gc),elapsed(gb)
                    else:b,c=elapsed(gb),elapsed(gc)
                    ratios.append(b/c);bt.append(b);ct.append(c)
                print('RESULT',json.dumps(dict(layer=layer,batch=batch,seed=seed,splits=splits,
                    error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),
                    candidate_us=statistics.median(ct))),flush=True)
                del gc,oc
            del gb,ob,data
