"""Fresh-seed confirmation, validating production outputs as well as candidates."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import triton,types
sys.path.insert(0,'/w/modal');from compile_split import build
flags=['-O3','-gencode=arch=compute_90a,code=sm_90a']
mods={'separate':build('muse_c16_confirm',candidate.CPP,candidate.CUDA,['cluster_attention','cluster_capacity'],flags),
      'prepare':build('muse_c16_prepare',candidate.PREP_CPP,candidate.PREP_CUDA,['fused_attention'],flags),
      'stats':build('muse_c16_stats',candidate.PREP_CPP,candidate.STATS_CUDA,['fused_attention'],flags)}
assert mods['separate'].cluster_capacity(192,1)>0
def fused(data):
    c=data['config'];chunk=triton.cdiv(triton.cdiv(c['window']+7,16),64)*64
    if chosen=='separate':
        qkv,q=base._fd_prepare(data)
        return mods[chosen].cluster_attention(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],qkv,c['page_size'],c['window'],chunk,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.)).view(c['batch'],8,-1)
    qkv=data['qkv'].view(c['batch']*8,-1)
    return mods[chosen].fused_attention(qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],c['page_size'],chunk,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.),c['rope_theta'])
trial=types.SimpleNamespace(kernel_fn=fused)
for batch in (1,2,4,8):
    for seed in (5011,15241,49157):
        data=R.generate_input(batch,80000,'tp8','sliding',seed,q_len=8,page=1,dist='soft',spread='agentic')
        gb,ob=capture(base,data)
        assert isinstance(ob,torch.Tensor),'production path uses an unavailable mocked library'
        ok,msg=R.check_implementation(data,R.collect(data,ob))
        print('BASE_CHECK',batch,seed,ok,msg,flush=True);assert ok,msg
        for chosen in ('separate','prepare','stats'):
            g,o=capture(trial,data);ok,msg=R.check_implementation(data,R.collect(data,o))
            print('CHECK',chosen,batch,seed,ok,msg,flush=True);assert ok,msg
            bt,ct,ratios=[],[],[]
            for rep in range(24):
                if rep%2:c,b=elapsed(g),elapsed(gb)
                else:b,c=elapsed(gb),elapsed(g)
                bt.append(b);ct.append(c);ratios.append(b/c)
            print('RESULT',json.dumps(dict(variant=chosen,batch=batch,seed=seed,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
            del g,o
        del gb,ob,data
for batch,ctx,page,scale,dist in [(3,4000,1,1/64,'soft'),(4,6000,64,1.,'edges')]:
    data=R.generate_input(batch,ctx,'tp8','sliding',819+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
    for chosen in ('separate','prepare','stats'):
        o=fused(data);ok,msg=R.check_implementation(data,R.collect(data,o))
        print('EDGE_CHECK',chosen,batch,ctx,page,scale,ok,msg,flush=True);assert ok,msg
print('CLUSTER_NONPORTABLE_CONFIRM_PASSED',flush=True)
