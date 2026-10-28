"""Baseline-matched FP8 probabilities: reference first, then weighted-shape screen."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
for batch,ctx,layer,page,scale,dist in ((8,6000,'full',64,1.,'edges'),(9,9000,'full',64,1/64,'soft'),(3,4000,'sliding',64,1/64,'soft')):
 data=R.generate_input(batch,ctx,'tp8',layer,371+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
 out=candidate.kernel_fn(data);ok,msg=R.check_implementation(data,R.collect(data,out))
 print('CHECK',batch,layer,ok,msg,flush=True);assert ok,msg
for layer,batch,options in (('full',16,(8,16,32)),('full',32,(8,16,32))):
 data=R.generate_input(batch,80000,'tp8',layer,7919,q_len=8,page=1,spread='agentic',dist='coherent')
 gb,ob=capture(base,data)
 for bn in (64,128):
  for splits in options:
   candidate._FP8P_BN=bn;candidate._FP8P_SPLITS=splits
   gc,oc=capture(candidate,data)
   ok,msg=R.check_implementation(data,R.collect(data,oc));assert ok,msg
   error=((oc.float()-ob.float()).norm()/ob.float().norm()).item()
   bt,ct,ratios=[],[],[]
   for rep in range(16):
    if rep%2:c,b=elapsed(gc),elapsed(gb)
    else:b,c=elapsed(gb),elapsed(gc)
    bt.append(b);ct.append(c);ratios.append(b/c)
   print('RESULT',json.dumps(dict(layer=layer,batch=batch,bn=bn,splits=splits,relative_error=error,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
   del gc,oc
 del gb,ob,data
print('FP8P_SCREEN_PASSED',flush=True)
