exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
sys.path.insert(0,'/w/modal');from compile_split import build
for name,cuda in (('readwrite',candidate._QREG_RW_CUDA),('serialize',candidate._QREG_SERIAL_CUDA)):
 candidate._WG_MOD=build('muse_qreg_hazard_'+name,candidate._WG_CPP,cuda,['wg5_run_t'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
 success=True
 for batch,ctx,page,scale,dist in ((8,6000,64,1.,'edges'),(9,9000,64,1/64,'soft')):
  data=R.generate_input(batch,ctx,'tp8','full',371+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
  out=candidate.kernel_fn(data);ok,msg=R.check_implementation(data,R.collect(data,out));print('CHECK',name,batch,ok,msg,flush=True);success&=ok
 if not success:continue
 for batch in (16,32):
  data=R.generate_input(batch,80000,'tp8','full',173,q_len=8,page=1,spread='agentic',dist='coherent')
  gc,oc=capture(candidate,data);ok,msg=R.check_implementation(data,R.collect(data,oc));print('TIMED_CHECK',name,batch,ok,msg,flush=True)
  if not ok:continue
  gb,ob=capture(base,data);bt,ct,ratios=[],[],[]
  for rep in range(20):
   if rep%2:c,b=elapsed(gc),elapsed(gb)
   else:b,c=elapsed(gb),elapsed(gc)
   bt.append(b);ct.append(c);ratios.append(b/c)
  print('RESULT',json.dumps(dict(variant=name,batch=batch,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
  del data,gc,gb,oc,ob
print('QREG_HAZARD_SCREEN_FINISHED',flush=True)
