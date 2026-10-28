"""Assemble standalone sparse-head decode candidate from the validated parent."""
import ast,runpy,argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--kernel',default='dev/sparse_head/decode_kernel.py')
parser.add_argument('--output',default='dev/ptx_campaign/attention_sparse_head_v1.py')
parser.add_argument('--module',default='muse_sparse_head_decode_v1')
args=parser.parse_args()
src=Path('experiments/attention_verify_small128/submission.py').read_text()
probe=runpy.run_path(args.kernel)
src+='\n\n_SP_CPP = '+repr(probe['CPP'])+'\n_SP_CUDA = '+repr(probe['CUDA'])+'\n'
src+='''
_SP_MOD = None
_SP_SPLITS_OVERRIDE = None

def _sp_module():
    global _SP_MOD
    if _SP_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _SP_MOD = load_inline('muse_sparse_head_decode_v1', _SP_CPP, cuda_sources=_SP_CUDA,
            functions=['sparse_decode_run'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'], verbose=False)
    return _SP_MOD

def _sp_forward(data, q, qkv):
    cfg = data['config']
    batch = cfg['batch']
    window = cfg['window']
    splits = _SP_SPLITS_OVERRIDE
    if splits is None:
        splits = (64 if batch == 1 else 8 if batch <= 24 else 4) if window else (256 if batch == 1 else 64 if batch <= 24 else 32)
    span = window if window else data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), 32) * 32
    partial = torch.empty((batch, 2, splits, 16, 64), dtype=torch.float32, device=qkv.device)
    lse = torch.empty((batch, 2, splits, 16), dtype=torch.float32, device=qkv.device)
    _sp_module().sparse_decode_run(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],
        data['k_scale'],data['v_scale'],partial,lse,cfg['page_size'],window,splits,chunk,
        cfg['sm_scale']*cfg['qk_scale_factor'],int(cfg['kv_scale'] != 1.0))
    result = torch.empty((batch,1,1024),dtype=torch.bfloat16,device=qkv.device)
    _merge_norm_gate[(batch,16)](partial,lse,qkv,result,16,2,64,1,qkv.stride(0),splits,16,num_warps=4)
    return result
'''
old="    if (cfg['q_len'] == 8 and not cfg['window'] and cfg['batch'] <= 4"
new="""    if (cfg['q_len'] == 1 and cfg['batch'] <= 64 and cfg['num_q_heads'] == 16
            and cfg['num_kv_heads'] == 2 and ctx_bound <= (1 << 20) + 64
            and cfg['page_size'] & (cfg['page_size'] - 1) == 0):
        qkv, q = _fd_prepare(data)
        return _sp_forward(data, q, qkv)
    if (cfg['q_len'] == 8 and not cfg['window'] and cfg['batch'] <= 4"""
assert src.count(old)==1
src=src.replace(old,new)
src=src.replace("'muse_sparse_head_decode_v1'",repr(args.module))
ast.parse(src)
Path(args.output).write_text(src)
