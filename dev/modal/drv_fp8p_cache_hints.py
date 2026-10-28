exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
sys.path.insert(0,'/w/modal');from compile_split import build
mods={}
for name,cuda in candidate._CACHE_VARIANTS.items():
 mods[name]=build('muse_fp8p_cache_'+name,candidate._WG_CPP,cuda,['wg5_run_t'],['-O3','-gencode=arch=compute_90a,code=sm_90a']);print('COMPILED',name,flush=True)
for batch in (16,32):
 for seed in (587,2417):
  data=R.generate_input(batch,80000,'tp8','full',seed,q_len=8,page=1,spread='agentic',dist='coherent')
  gb,ob=capture(base,data)
  for name,mod in mods.items():
   candidate._WG_MOD=mod;gc,oc=capture(candidate,data)
   ok,msg=R.check_implementation(data,R.collect(data,oc));print('CHECK',name,batch,seed,ok,msg,flush=True);assert ok,msg
   err=((oc.float()-ob.float()).norm()/ob.float().norm()).item();assert err<1.e-6,err
   bt,ct,ratios=[],[],[]
   for rep in range(20):
    if rep%2:c,b=elapsed(gc),elapsed(gb)
    else:b,c=elapsed(gb),elapsed(gc)
    bt.append(b);ct.append(c);ratios.append(b/c)
   print('RESULT',json.dumps(dict(variant=name,batch=batch,seed=seed,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
   del gc,oc
  del data,gb,ob
print('FP8P_CACHE_HINTS_PASSED',flush=True)
