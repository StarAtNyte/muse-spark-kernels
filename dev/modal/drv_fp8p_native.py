"""Check canonical WGMMA swizzles, then compare paired graph replays."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
# Verify before baseline compilation and benchmarking.
for batch,ctx,page,scale,dist in ((8,6000,64,1.,'edges'),(9,9000,64,1/64,'soft')):
 data=R.generate_input(batch,ctx,'tp8','full',371+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
 out=candidate.kernel_fn(data);ok,message=R.check_implementation(data,R.collect(data,out));print('EARLY_CHECK',batch,ok,message,flush=True);assert ok,message
for batch in (8,16,32):
 for seed in (113,7919,3613):
  data=R.generate_input(batch,80000,'tp8','full',seed,q_len=8,page=1,spread='agentic',dist='coherent')
  gb,ob=capture(base,data);gc,oc=capture(candidate,data)
  err=((ob.float()-oc.float()).norm()/ob.float().norm()).item();ok,msg=R.check_implementation(data,R.collect(data,oc));print('TIMED_CHECK',batch,seed,ok,msg,flush=True);assert ok,msg
  bt,ct,ratios=[],[],[]
  for rep in range(20):
   if rep%2:c,b=elapsed(gc),elapsed(gb)
   else:b,c=elapsed(gb),elapsed(gc)
   bt.append(b);ct.append(c);ratios.append(b/c)
  print('RESULT',json.dumps(dict(batch=batch,seed=seed,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
  del data,gb,gc,ob,oc
print('FP8P_NATIVE_VERIFY_PASSED',flush=True)
