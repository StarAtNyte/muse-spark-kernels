"""Screen native 128-key tiles on the scored one-request verify regime."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
split_count = 64

def native(data):
    qkv,q = candidate._fd_prepare(data)
    return candidate._wg_forward(data,q,qkv,split_count)

candidate.kernel_fn = native
rows=[]
for batch in (1, 2):
    for seed in (173,2387,7919):
        data=R.generate_input(batch,80000,'tp8','full',seed,q_len=8,
            spread='agentic',dist='coherent',page=1)
        gb,ob=capture(base,data)
        for splits in (32,64,128,256):
            split_count=splits
            gc,oc=capture(candidate,data)
            error=((ob.float()-oc.float()).norm()/ob.float().norm()).item()
            if not torch.isfinite(oc).all().item() or error>.003:
                raise AssertionError((batch,seed,splits,error))
            ratios,bt,ct=[],[],[]
            for rep in range(20):
                if rep%2: c,b=elapsed(gc),elapsed(gb)
                else: b,c=elapsed(gb),elapsed(gc)
                ratios.append(b/c);bt.append(b);ct.append(c)
            row=dict(batch=batch,seed=seed,splits=splits,error=error,
                speedup=statistics.median(ratios),baseline_us=statistics.median(bt),
                candidate_us=statistics.median(ct))
            rows.append(row);print('RESULT',json.dumps(row),flush=True)
            del gc,oc
        del gb,ob,data
split_count=128
for batch,ctx,page,scale,dist in ((1,6000,64,1.,'edges'),(2,9000,64,1/64,'soft'),
                                  (1,1048576,1,1.,'coherent')):
    data=R.generate_input(batch,ctx,'tp8','full',351+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
    out=candidate.kernel_fn(data)
    ok,message=R.check_implementation(data,R.collect(data,out))
    print('CHECK',batch,ctx,page,scale,ok,message,flush=True)
    if not ok: raise AssertionError(message)
