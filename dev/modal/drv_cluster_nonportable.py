"""Verify actual H200 support, then compare 8/16-block DSM attention."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import triton,types
sys.path.insert(0,'/w/modal');from compile_split import build
flags=['-O3','-gencode=arch=compute_90a,code=sm_90a']
mods={8:build('muse_cluster_control8',candidate.CPP8,candidate.CUDA8,['cluster_attention'],flags),
      16:build('muse_cluster_nonportable16',candidate.CPP,candidate.CUDA,['cluster_attention','cluster_capacity'],flags)}
def fused(data):
    c=data['config'];qkv,q=base._fd_prepare(data)
    chunk=triton.cdiv(triton.cdiv(c['window']+7,chosen),64)*64
    return mods[chosen].cluster_attention(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],qkv,c['page_size'],c['window'],chunk,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.)).view(c['batch'],8,-1)
trial=types.SimpleNamespace(kernel_fn=fused)
for batch in (1,4,16,32):
    capacity=mods[16].cluster_capacity(192,batch)
    print('CLUSTER16_CAPACITY',batch,capacity,flush=True)
    assert capacity>0,('H200 cannot launch this cluster geometry',capacity)
    for seed in (173,2387):
        data=R.generate_input(batch,80000,'tp8','sliding',seed,q_len=8,page=1,dist='soft',spread='agentic')
        gb,ob=capture(base,data)
        for chosen in (8,16):
            g,o=capture(trial,data)
            ok,msg=R.check_implementation(data,R.collect(data,o))
            print('CHECK',batch,seed,chosen,ok,msg,flush=True);assert ok,msg
            bt,ct,ratios=[],[],[]
            for rep in range(24):
                if rep%2:c,b=elapsed(g),elapsed(gb)
                else:b,c=elapsed(gb),elapsed(g)
                bt.append(b);ct.append(c);ratios.append(b/c)
            print('RESULT',json.dumps(dict(batch=batch,seed=seed,cluster_size=chosen,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
            del g,o
        del gb,ob,data
for batch,ctx,page,scale,dist in [(3,4000,1,1/64,'soft'),(4,6000,64,1.,'edges')]:
    data=R.generate_input(batch,ctx,'tp8','sliding',819+batch,q_len=8,page=page,kv_scale=scale,dist=dist)
    chosen=16;o=fused(data);ok,msg=R.check_implementation(data,R.collect(data,o))
    print('EDGE_CHECK',batch,ctx,page,scale,ok,msg,flush=True);assert ok,msg
print('CLUSTER_NONPORTABLE_SCREEN_PASSED',flush=True)
