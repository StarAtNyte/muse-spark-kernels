"""Screen m=1 dense GEMV tile choices; current official QKV/O picks use v5."""
exec(open('/w/modal/drv_router.py').read().split("print('GPU'")[0])
import triton
configs={
 'qkv_gate':((1,512,8),(1,1024,8),(2,512,8),(2,1024,8),(4,1024,8),(8,512,8),(8,1024,8)),
 'o':((4,1024,4),(4,1024,8),(8,512,4),(8,1024,4),(8,1024,8),(16,1024,8),(16,1024,4))}
for layer,cfgs in configs.items():
    for seed in (173,2387,7919):
        data=R.generate_input(1,layer,seed)
        n,k=data['config']['n'],data['x'].shape[1]
        baseline,_=capture(lambda:sub._kernel_v5(data,state))
        for bn,bk,nw in cfgs:
            output=torch.empty((1,n),dtype=torch.bfloat16,device='cuda')
            def run():
                sub._gemv[(triton.cdiv(n,bn),)](data['x'],weights[layer],output,n,k,bn,bk,num_warps=nw)
                return output
            graph,out=capture(run)
            ok,message=R.check_implementation(data,out,weights)
            if not ok:raise AssertionError((layer,bn,bk,nw,message))
            ratios,bt,ct=[],[],[]
            for rep in range(16):
                if rep%2:c,b=elapsed(graph),elapsed(baseline)
                else:b,c=elapsed(baseline),elapsed(graph)
                ratios.append(b/c);bt.append(b);ct.append(c)
            print('RESULT',json.dumps(dict(layer=layer,seed=seed,bn=bn,bk=bk,warps=nw,
                speedup=statistics.median(ratios),baseline_us=statistics.median(bt),
                candidate_us=statistics.median(ct))),flush=True)
            del graph,out,output
