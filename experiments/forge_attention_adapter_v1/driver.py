"""Validate the exact Forge attention wrapper against the challenge reference."""
import importlib.util, sys, json, torch
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R
sp=importlib.util.spec_from_file_location('submission','/w/run/sub.py')
S=importlib.util.module_from_spec(sp);sys.modules['submission']=S;sp.loader.exec_module(S)
exec(compile('"""Promoted Muse attention, page-1 sliding path with cache writes returned."""\nimport submission as _mod\n\ndef run(qkv, seq_lens, cache_loc, k_cache, v_cache, page_table, k_scale, v_scale):\n    batch, length = qkv.shape[:2]\n    cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2,\n               head_dim=64, page_size=1, window=2048, rope_theta=500000.0,\n               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0)\n    data = dict(qkv=qkv, seq_lens=seq_lens, cache_loc=cache_loc, k_cache=k_cache,\n                v_cache=v_cache, page_table=page_table, k_scale=k_scale, v_scale=v_scale,\n                config=cfg)\n    out = _mod.kernel_fn(data)\n    return out, k_cache, v_cache\n', 'forge_attention_wrapper', 'exec'))
for B,L,scale in ((1,1,1.0),(1,8,1.0),(2,1,1/64)):
    d=R.generate_input(B, 3000, 'tp8', 'sliding', 927+B+L, q_len=L, page=1, kv_scale=scale, dist='soft')
    o,k,v=run(*(d[n] for n in ('qkv','seq_lens','cache_loc','k_cache','v_cache','page_table','k_scale','v_scale')))
    torch.cuda.synchronize()
    ok,msg=R.check_implementation(d,R.collect(d,o))
    assert ok and k.data_ptr()==d['k_cache'].data_ptr() and v.data_ptr()==d['v_cache'].data_ptr(),msg
    print('PASS',json.dumps(dict(batch=B,length=L,scale=scale,message=msg)),flush=True)
print('DONE',flush=True)
