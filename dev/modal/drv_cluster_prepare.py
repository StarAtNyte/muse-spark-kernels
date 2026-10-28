exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import triton
sys.path.insert(0,'/w/modal');from compile_split import build
mod=build('muse_cluster_prepare_fused',candidate.CPP,candidate.CUDA,['fused_attention'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
def fused(data):
 c=data['config'];qkv=data['qkv'].view(c['batch']*8,-1)
 return mod.fused_attention(qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],c['page_size'],320,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.),c['rope_theta'])
candidate.kernel_fn=fused
for batch,ctx,page,scale,dist in ((1,6000,64,1.,'edges'),(4,9000,64,1/64,'soft'),(1,80000,1,1.,'coherent'),(4,80000,1,1.,'coherent'),(16,80000,1,1.,'coherent')):
 for seed in (173,2387):
  data=R.generate_input(batch,ctx,'tp8','sliding',seed,q_len=8,page=page,kv_scale=scale,dist=dist,spread='agentic')
  gc,oc=capture(candidate,data);ok,msg=R.check_implementation(data,R.collect(data,oc));print('CHECK',batch,ctx,seed,ok,msg,flush=True);assert ok,msg
  if batch>4:continue
  gb,ob=capture(base,data);bt,ct,ratios=[],[],[]
  for rep in range(24):
   if rep%2:c,b=elapsed(gc),elapsed(gb)
   else:b,c=elapsed(gb),elapsed(gc)
   bt.append(b);ct.append(c);ratios.append(b/c)
  print('RESULT',json.dumps(dict(batch=batch,ctx=ctx,seed=seed,baseline_us=statistics.median(bt),fused_us=statistics.median(ct),speedup=statistics.median(ratios))),flush=True)
  del data,gc,gb,oc,ob
print('CLUSTER_PREPARE_FUSED_PASSED',flush=True)
