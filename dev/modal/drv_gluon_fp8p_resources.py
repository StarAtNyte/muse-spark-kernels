"""Cold-L2 screen, independent reference first; full official coverage required."""
from pathlib import Path
for cached in Path('/usr/local/lib/python3.12/site-packages/triton').rglob('*.pyc'):
 cached.unlink(missing_ok=True)
import torch,triton
print('ENVIRONMENT',torch.__version__,triton.__version__,flush=True)
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])

data=R.generate_input(16,80000,'tp8','full',419,q_len=8,page=1,spread='agentic',dist='coherent')
for name,module,fn_name in [('baseline',base,'_os__g_full_verify'),('fp8',candidate,'_g_full_verify')]:
    graph,out=capture(module,data)
    ok,msg=R.check_implementation(data,R.collect(data,out));print('CHECK',name,ok,msg,flush=True);assert ok,msg
    fn=getattr(module,fn_name)
    for entry in fn.device_caches.values():
        for kernel in entry[0].values():
            ptx=kernel.asm.get('ptx','')
            print('RESOURCES',name,json.dumps(dict(registers=kernel.n_regs,spills=kernel.n_spills,shared=kernel.metadata.shared,metadata=str(kernel.metadata),ptx_bytes=len(ptx),shared_b8_stores=ptx.count('st.shared.b8'),shared_b16_stores=ptx.count('st.shared.b16'),shared_b32_stores=ptx.count('st.shared.b32'),shared_b64_stores=ptx.count('st.shared.b64'),shared_b128_stores=ptx.count('st.shared.v4.b32'),wgmma_waits=ptx.count('wgmma.wait_group'),wgmma_ops=ptx.count('wgmma.mma_async'))),flush=True)
print('GLUON_RESOURCE_COMPARISON_COMPLETE',flush=True)
