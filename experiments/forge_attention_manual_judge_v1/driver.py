import importlib.util, sys, torch
sys.path.insert(0,'/w/kernels/muse_attention_decode')
import reference as R
sp=importlib.util.spec_from_file_location('judge','/w/run/sub.py'); j=importlib.util.module_from_spec(sp); sp.loader.exec_module(j)
for b,l,g in ((1,1,'typical'),(1,8,'short'),(2,1,'scaled')):
    t=5000
    axes=dict(batch=b,length=l,tokens=b*l,pages=b*t,table_len=t)
    d=j.make_inputs(axes,g,123+b+l,'cuda')
    cache0=d['k_cache'].clone(); value0=d['v_cache'].clone()
    out,k,v=j.run(*(d[n] for n in ('qkv','seq_lens','cache_loc','k_cache','v_cache','page_table','k_scale','v_scale')))
    cfg=dict(batch=b,q_len=l,num_q_heads=16,num_kv_heads=2,head_dim=64,page_size=1,window=2048,rope_theta=500000.0,qk_scale_factor=5.4730064863875,sm_scale=0.125,kv_scale=1.0)
    inp=dict(d); inp.update(config=cfg,k_cache=cache0,v_cache=value0)
    ref,kn,vn=R.ref_kernel(inp)
    out_err=((out.float()-ref).square().sum()/ref.square().sum()).sqrt().item()
    slots=d['cache_loc']; km=torch.count_nonzero(k.view(-1,2,64)[slots].float()!=kn.reshape(-1,2,64).float()).item(); vm=torch.count_nonzero(v.view(-1,2,64)[slots].float()!=vn.reshape(-1,2,64).float()).item()
    print('RESULT',b,l,g,'relative_l2',out_err,'k_mismatch',km,'v_mismatch',vm,flush=True)
    assert out_err < 0.01 and km == 0 and vm == 0
print('DONE',flush=True)
