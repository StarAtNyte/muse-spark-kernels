"""Exact FP32 router GEMV for one token versus the validated BF16 hi/lo path."""
exec(open('/w/modal/drv_router.py').read().split("print('GPU'")[0])
import triton
print('GPU',torch.cuda.get_device_name(),flush=True)
for seed in (173,2387,7919):
    data=R.generate_input(1,'router',seed)
    baseline,_=capture(lambda:sub._kernel_v5(data,state))
    rows=[]
    for bn,bk,nw in ((1,512,4),(1,1024,4),(1,2048,4),(2,512,4),(2,1024,4),
                     (2,2048,4),(4,1024,4),(4,2048,4),(1,1024,8),(2,1024,8)):
        output=torch.empty((1,256),dtype=torch.float32,device='cuda')
        def run():
            sub._gemv[(triton.cdiv(256,bn),)](data['x'],weights['router'],output,256,8192,bn,bk,num_warps=nw)
            return output
        graph,out=capture(run)
        ok,message=R.check_implementation(data,out,weights)
        if not ok:raise AssertionError((bn,bk,nw,message))
        ratios,bt,ct=[],[],[]
        for rep in range(16):
            if rep%2:c,b=elapsed(graph),elapsed(baseline)
            else:b,c=elapsed(baseline),elapsed(graph)
            ratios.append(b/c);bt.append(b);ct.append(c)
        print('RESULT',json.dumps(dict(seed=seed,bn=bn,bk=bk,warps=nw,
            speedup=statistics.median(ratios),baseline_us=statistics.median(bt),
            candidate_us=statistics.median(ct))),flush=True)
        del graph,out,output
