"""Retain all production paths; sixteen-block clusters only for small verify."""
from pathlib import Path
import runpy,ast
m=runpy.run_path('dev/cluster_attention/nonportable.py')
s=Path('experiments/attention_fp8p_cluster_v1/submission.py').read_text()
s+='\n_CLUSTER16_CPP='+repr(m['CPP'])+'\n_CLUSTER16_CUDA='+repr(m['CUDA'])+'\n'
s+=r'''
_CLUSTER16_MOD=None
def _cluster16_module():
    global _CLUSTER16_MOD
    if _CLUSTER16_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _CLUSTER16_MOD=load_inline('muse_cluster16_small_verify',_CLUSTER16_CPP,cuda_sources=_CLUSTER16_CUDA,functions=['cluster_attention','cluster_capacity'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
    return _CLUSTER16_MOD
_before_cluster16_kernel_fn=kernel_fn
def kernel_fn(data):
    c=data['config']
    if (c['q_len']==8 and c['window']==2048 and 1<=c['batch']<=4
            and c['num_q_heads']==16 and c['num_kv_heads']==2
            and c['head_dim']==64 and c['page_size']&(c['page_size']-1)==0):
        qkv,q=_fd_prepare(data)
        chunk=triton.cdiv(triton.cdiv(c['window']+7,16),64)*64
        out=_cluster16_module().cluster_attention(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],qkv,c['page_size'],c['window'],chunk,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.))
        return out.view(c['batch'],8,-1)
    return _before_cluster16_kernel_fn(data)
'''
ast.parse(s)
Path('dev/ptx_campaign/attention_cluster16_small_v1.py').write_text(s)
