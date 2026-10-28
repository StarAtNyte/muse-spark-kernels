"""Compare two CPU-validated scheduling variants against the current production."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
sys.path.insert(0,'/w/modal');from compile_split import build
modules={}
for name,cuda in (('vsw64',candidate._WG_CUDA),('original',candidate._FP8P_SERIAL_CUDA)):
 modules[name]=build('muse_joint_vsw_'+name,candidate._WG_CPP,cuda,['wg5_run_t'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
 print('COMPILED',name,flush=True)
 candidate._WG_MOD=modules[name]
 for batch,ctx,page,scale,dist in ((8,6000,64,1.,'edges'),(9,9000,64,1/64,'soft')):
  data=R.generate_input(batch,ctx,'tp8','full',371+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
  out=candidate.kernel_fn(data);ok,msg=R.check_implementation(data,R.collect(data,out));print('CHECK',name,batch,ok,msg,flush=True);assert ok,msg
for batch in (16,32):
 for seed in (7919,):
  data=R.generate_input(batch,80000,'tp8','full',seed,q_len=8,page=1,spread='agentic',dist='coherent')
  gb,ob=capture(base,data)
  for name,mod in modules.items():
   candidate._WG_MOD=mod;gc,oc=capture(candidate,data)
   ok,msg=R.check_implementation(data,R.collect(data,oc));print('TIMED_CHECK',name,batch,seed,ok,msg,flush=True);assert ok,msg
   err=((oc.float()-ob.float()).norm()/ob.float().norm()).item();bt,ct,ratios=[],[],[]
   for rep in range(20):
    if rep%2:c,b=elapsed(gc),elapsed(gb)
    else:b,c=elapsed(gb),elapsed(gc)
    bt.append(b);ct.append(c);ratios.append(b/c)
   print('RESULT',json.dumps(dict(variant=name,batch=batch,seed=seed,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
   del gc,oc
  del data,gb,ob
print('FP8P_JOINT_COMPARE_PASSED',flush=True)
