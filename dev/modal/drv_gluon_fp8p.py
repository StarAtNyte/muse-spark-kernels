"""Cold-L2 screen, independent reference first; full official coverage required."""
from pathlib import Path
for cached in Path('/usr/local/lib/python3.12/site-packages/triton').rglob('*.pyc'):
 cached.unlink(missing_ok=True)
import torch,triton
print('ENVIRONMENT',torch.__version__,triton.__version__,flush=True)
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
for bs,ctx,scale,dist in ((8,512,1.,'edges'),(9,9000,1/64,'soft'),(8,96,1.,'soft')):
 data=R.generate_input(bs,ctx,'tp8','full',371+bs,q_len=8,page=1,kv_scale=scale,dist=dist)
 out=candidate.kernel_fn(data);assert isinstance(out,torch.Tensor)
 ok,msg=R.check_implementation(data,R.collect(data,out));print('CHECK',bs,ctx,scale,ok,msg,flush=True);assert ok,msg
for bs in (16,32):
 for seed in (419,7919,15241):
    data=R.generate_input(bs,80000,'tp8','full',seed,q_len=8,page=1,spread='agentic',dist='coherent')
    gb,ob=capture(base,data);g,o=capture(candidate,data)
    ok,msg=R.check_implementation(data,R.collect(data,o));print('TIMED_CHECK',bs,seed,ok,msg,flush=True);assert ok,msg
    error=((o.float()-ob.float()).norm()/ob.float().norm()).item();bt=[];ct=[];rr=[]
    for rep in range(24):
        if rep%2:c,b=elapsed(g),elapsed(gb)
        else:b,c=elapsed(gb),elapsed(g)
        bt.append(b);ct.append(c);rr.append(b/c)
    print('RESULT',json.dumps(dict(batch=bs,seed=seed,error=error,ratio=statistics.median(rr),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
print('GLUON_FP8P_SCREEN_PASSED',flush=True)
