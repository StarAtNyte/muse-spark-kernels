"""Sufficient-statistic attention merge with several row groupings."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
sys.path.insert(0,'/w/modal');from compile_split import build
candidate._WG_MOD=build('muse_fp8p_stats',candidate._WG_CPP,candidate._WG_CUDA,['wg5_run_t'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
for batch,ctx,page,scale,dist in ((8,6000,64,1.,'edges'),(9,9000,64,1/64,'soft')):
 data=R.generate_input(batch,ctx,'tp8','full',371+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
 out=candidate.kernel_fn(data);ok,msg=R.check_implementation(data,R.collect(data,out));print('EARLY_CHECK',batch,ok,msg,flush=True);assert ok,msg
for batch in (16,32):
 for seed in (173,2387):
  data=R.generate_input(batch,80000,'tp8','full',seed,q_len=8,page=1,spread='agentic',dist='coherent')
  gb,ob=capture(base,data)
  for rows in (1,2,4,8):
   candidate._STATS_ROWS=rows;gc,oc=capture(candidate,data)
   ok,msg=R.check_implementation(data,R.collect(data,oc));print('CHECK',batch,seed,rows,ok,msg,flush=True);assert ok,msg
   bt,ct,ratios=[],[],[]
   for rep in range(20):
    if rep%2:c,b=elapsed(gc),elapsed(gb)
    else:b,c=elapsed(gb),elapsed(gc)
    bt.append(b);ct.append(c);ratios.append(b/c)
   print('RESULT',json.dumps(dict(batch=batch,seed=seed,rows=rows,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
   del gc,oc
  del data,gb,ob
print('STATS_MERGE_PASSED',flush=True)
