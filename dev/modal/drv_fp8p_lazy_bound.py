exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
sys.path.insert(0,'/w/modal');from compile_split import build
mods={}
for th,cuda in candidate._LAZY_VARIANTS.items():
 mods[th]=build('muse_lazy_bound_'+str(th).replace('.','_'),candidate._WG_CPP,cuda,['wg5_run_t'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
 print('COMPILED',th,flush=True)
base._WG_MOD=mods[0.]
for th in (.25,.5,.75):
 candidate._WG_MOD=mods[th]
 for batch,ctx,page,scale,dist in ((8,6000,64,1.,'edges'),(9,9000,64,1/64,'soft')):
  data=R.generate_input(batch,ctx,'tp8','full',733+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
  out=candidate.kernel_fn(data);ok,msg=R.check_implementation(data,R.collect(data,out));print('EARLY_CHECK',th,batch,ok,msg,flush=True);assert ok,msg
for batch in (16,32):
 for seed in (331,1223):
  data=R.generate_input(batch,80000,'tp8','full',seed,q_len=8,page=1,spread='agentic',dist='coherent')
  gb,ob=capture(base,data)
  for th in (.25,.5,.75):
   candidate._WG_MOD=mods[th];gc,oc=capture(candidate,data)
   ok,msg=R.check_implementation(data,R.collect(data,oc));print('CHECK',batch,seed,th,ok,msg,flush=True);assert ok,msg
   bt,ct,ratios=[],[],[]
   for rep in range(20):
    if rep%2:c,b=elapsed(gc),elapsed(gb)
    else:b,c=elapsed(gb),elapsed(gc)
    bt.append(b);ct.append(c);ratios.append(b/c)
   print('RESULT',json.dumps(dict(batch=batch,seed=seed,threshold=th,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
   del gc,oc
  del data,gb,ob
print('FP8P_LAZY_BOUND_PASSED',flush=True)
