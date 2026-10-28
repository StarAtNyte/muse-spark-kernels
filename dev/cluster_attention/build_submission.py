from pathlib import Path
import runpy,ast,argparse
parser=argparse.ArgumentParser()
parser.add_argument('--parent',default='kernels/muse_attention_decode/submission.py')
parser.add_argument('--name',default='attention_cluster_sliding_v1')
args=parser.parse_args()
m=runpy.run_path('dev/cluster_attention/fused.py')
s=Path(args.parent).read_text()
s+='\n_CLUSTER_CPP = '+repr(m['CPP'])+'\n_CLUSTER_CUDA = '+repr(m['CUDA'])+'\n'
s+=r'''
_CLUSTER_MOD=None
def _cluster_module():
    global _CLUSTER_MOD
    if _CLUSTER_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _CLUSTER_MOD=load_inline('muse_cluster_sliding_v1',_CLUSTER_CPP,cuda_sources=_CLUSTER_CUDA,functions=['cluster_attention'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
    return _CLUSTER_MOD

_before_cluster_kernel_fn=kernel_fn
def kernel_fn(data):
    c=data['config']
    if (c['q_len']==8 and c['window']==2048 and 2<=c['batch']<=24
            and c['num_q_heads']==16 and c['num_kv_heads']==2
            and c['head_dim']==64 and c['page_size']&(c['page_size']-1)==0):
        qkv,q=_fd_prepare(data)
        chunk=triton.cdiv(triton.cdiv(c['window']+7,8),64)*64
        out=_cluster_module().cluster_attention(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],qkv,c['page_size'],c['window'],chunk,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.))
        return out.view(c['batch'],8,-1)
    return _before_cluster_kernel_fn(data)
'''
ast.parse(s)
Path('dev/ptx_campaign/'+args.name+'.py').write_text(s)
