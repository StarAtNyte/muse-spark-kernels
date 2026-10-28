exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import types
sys.path.insert(0,'/w/modal');from compile_split import build
mods={}
mods['old']=build('muse_cluster_original',candidate._OLD_CPP,candidate._OLD_CUDA,['cluster_attention'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
for name,cuda in (('prep',candidate._PREP_CUDA),('stats',candidate._STATS_CUDA)):
 mods[name]=build('muse_cluster_'+name,candidate.CPP,cuda,['fused_attention'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
def old(data):
 c=data['config'];qkv,q=base._fd_prepare(data)
 return mods['old'].cluster_attention(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],qkv,c['page_size'],2048,320,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.)).view(c['batch'],8,-1)
def fused(data,name):
 c=data['config'];qkv=data['qkv'].view(c['batch']*8,-1)
 return mods[name].fused_attention(qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],c['page_size'],320,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.),c['rope_theta'])
variants={'old':types.SimpleNamespace(kernel_fn=old)}
for name in ('prep','stats'):variants[name]=types.SimpleNamespace(kernel_fn=lambda data,name=name:fused(data,name))
cases=[(b,80000,1,1.,'coherent',seed) for b in (1,4,16) for seed in (173,2387)]
cases += [(1,6000,64,1.,'edges',331),(4,9000,64,1/64,'soft',347)]
for batch,ctx,page,scale,dist,seed in cases:
 data=R.generate_input(batch,ctx,'tp8','sliding',seed,q_len=8,page=page,kv_scale=scale,dist=dist,spread='agentic')
 graphs={};outputs={};times={name:[] for name in variants}
 for name,v in variants.items():
  graphs[name],outputs[name]=capture(v,data)
  ok,msg=R.check_implementation(data,R.collect(data,outputs[name]));print('CHECK',name,batch,ctx,seed,ok,msg,flush=True);assert ok,msg
 names=list(variants)
 for rep in range(24):
  for offset in range(3):
   name=names[(rep+offset)%3];times[name].append(elapsed(graphs[name]))
 for name in names:
  ratio=statistics.median([a/b for a,b in zip(times['old'],times[name])])
  print('RESULT',json.dumps(dict(variant=name,batch=batch,ctx=ctx,page=page,seed=seed,old_us=statistics.median(times['old']),candidate_us=statistics.median(times[name]),speedup=ratio)),flush=True)
 del graphs,outputs,data
print('CLUSTER_PREPARE_COMPARE_PASSED',flush=True)
