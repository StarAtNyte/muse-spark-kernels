"""Carry validated attention branches into the current decoder, retaining SP."""
from pathlib import Path
import ast
attention=Path('experiments/attention_fp8p_cluster_v1/submission.py').read_text()
decoder=Path('kernels/muse_decode_layer/submission.py').read_text()
names={'_WG_CPP','_WG_CUDA','_WG_MOD','_wg_module','_wg_forward','_CLUSTER_CPP','_CLUSTER_CUDA','_CLUSTER_MOD','_cluster_module'}
parts=[];found=set()
for node in ast.parse(attention).body:
 name=node.name if isinstance(node,ast.FunctionDef) else next((t.id for t in node.targets if isinstance(t,ast.Name)),None) if isinstance(node,ast.Assign) else None
 if name in names:parts.append(ast.get_source_segment(attention,node));found.add(name)
assert found==names,(found,names)
marker='''    if use_transposed:
        o = _sp_forward(data, q, qkv).view(tokens, hq*d)'''
replacement='''    use_cluster = (length == 8 and cfg['window'] == 2048 and 2 <= batch <= 24
                   and hq == 16 and hkv == 2 and d == 64
                   and cfg['page_size'] & (cfg['page_size'] - 1) == 0)
    use_wg = (length == 8 and not cfg['window'] and batch >= 8 and hkv == 2 and hq == 16 and d == 64
              and data['page_table'].shape[1] * cfg['page_size'] <= (1 << 20) + 64
              and cfg['page_size'] & (cfg['page_size'] - 1) == 0)
    if use_cluster:
        o = _cluster_module().cluster_attention(q, k_cache, v_cache, data['page_table'], data['seq_lens'],
                data['k_scale'], data['v_scale'], qkv, cfg['page_size'], cfg['window'], 320,
                cfg['sm_scale'] * cfg['qk_scale_factor'], int(scaled)).view(tokens, hq*d)
    elif use_wg:
        o = _wg_forward(data, q, qkv, 16 if batch <= 24 else 32).view(tokens, hq*d)
    elif use_transposed:
        o = _sp_forward(data, q, qkv).view(tokens, hq*d)'''
assert decoder.count(marker)==1
s=decoder.replace(marker,replacement)+'\n\n# Officially validated standalone attention branches.\n'+'\n\n'.join(parts)+'\n'
s=s.replace('muse_wg5_fp8p_joint_serial','muse_decoder_wg5_fp8p_joint_serial').replace('muse_cluster_sliding_v1','muse_decoder_cluster_sliding_v1')
ast.parse(s);Path('dev/ptx_campaign/decoder_fp8p_cluster_v1.py').write_text(s)
