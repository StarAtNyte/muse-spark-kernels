import ast
from pathlib import Path

base = Path('experiments/scored_decoder_stack_verify_bucket_v2/submission.py').read_text()
moe = Path('experiments/scored_moe_hybrid_m32_m128_v2/submission.py').read_text()
tree = ast.parse(moe)
source_nodes = {}
import_nodes = []
first_om = next(n for n in tree.body if isinstance(n, ast.ImportFrom) and any(a.asname == '_om_TypedDict' for a in n.names))
weights_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_om_Weights')
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        source_nodes[n.name] = n
    elif isinstance(n, (ast.Assign, ast.AnnAssign)):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
        for target in targets:
            if isinstance(target, ast.Name): source_nodes[target.id] = n
    if isinstance(n, (ast.Import, ast.ImportFrom)) and first_om.lineno <= n.lineno < weights_cls.lineno:
        if not (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith('sglang.')):
            import_nodes.append(n)

roots = {'_om__route', '_om__w2', '_om__tile_major', '_om__permute_bits', '_om__gu_rows'}
seen = set()
todo = list(roots)
while todo:
    name = todo.pop()
    if name in seen: continue
    seen.add(name)
    node = source_nodes.get(name)
    if node is not None:
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        todo.extend(names & source_nodes.keys() - seen)
impl_nodes = sorted((source_nodes[n] for n in seen if n in source_nodes), key=lambda n:n.lineno)
def node_source(node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return '\n'.join(moe.splitlines()[start-1:node.end_lineno])
impl = '\n'.join(node_source(n) for n in import_nodes + impl_nodes)
impl += r'''
_om_TOP_K=8
_om_POST_NORM_EPS=1e-8
_om_NUM_WARPS=4

def _om_prepare_m32(weights):
    L = 32
    gu = _om__gu_rows(weights['w13'].shape[1] // 2).to(weights['w13'].device)
    w13g, w13g_sf = _om__tile_major(
        _om_torch.stack([_om__permute_bits(weights['w13'][e][gu]) for e in range(L)]),
        weights['w13_sf'][:L][:, gu])
    w2, w2_sf = _om__tile_major(
        _om_torch.stack([_om__permute_bits(weights['w2'][e]) for e in range(L)]),
        weights['w2_sf'][:L])
    return {'w13g':w13g,'w13g_sf':w13g_sf,'w13_gs':weights['w13_gs'][:L].float().contiguous(),
            'w2':w2,'w2_sf':w2_sf,'w2_gs':weights['w2_gs'][:L].float().contiguous(),
            'norm_w':(weights['post_norm_w'].float()+1.0).to(_om_torch.bfloat16),
            'local':(0,L),'hopper':_om_torch.cuda.get_device_capability()[0]==9}

def _om_m32_kernel_fn(data,state):
    x=data['hidden_states'].contiguous(); M,H=x.shape
    I=data['config']['d_expert']; L=data['config']['local_count']; ls=data['config']['local_start']
    assert M==32 and (ls,L)==state['local']
    P=M*_om_TOP_K; ids=data['topk_ids'].contiguous(); tw=data['topk_weights'].float().contiguous()
    bm=8; p_block=_om_triton.next_power_of_2(P); w_max=_om_triton.next_power_of_2(_om_triton.cdiv(P,bm)+L)
    local=_om_torch.empty((M,_om_TOP_K),dtype=_om_torch.int32,device=x.device)
    sorted_pairs=_om_torch.empty((P,),dtype=_om_torch.int32,device=x.device)
    work=_om_torch.empty((w_max,3),dtype=_om_torch.int32,device=x.device)
    nwork=_om_torch.empty((1,),dtype=_om_torch.int32,device=x.device)
    _om__route[1,](ids,local,sorted_pairs,work,nwork,P,ls,LOCAL=L,_om_BM=bm,P_BLOCK=p_block,W_MAX=w_max,num_warps=16 if p_block>=1024 else 4)
    inter=_om_torch.empty((P,I),dtype=_om_torch.bfloat16,device=x.device)
    _om__w2[2*I//_om_BN,w_max](x,state['w13g'],state['w13g_sf'],state['w13_gs'],sorted_pairs,work,nwork,inter,K=H,N=2*I,_om_BN=_om_BN,_om_BK=_om_BK,_om_BM=bm,XS=True,SMB=False,HOP=False,CHK=True,PF=False,GU=True,num_warps=_om_NUM_WARPS)
    y=_om_torch.empty((P,H),dtype=_om_torch.bfloat16,device=x.device)
    _om__w2[H//_om_BN,w_max](inter,state['w2'],state['w2_sf'],state['w2_gs'],sorted_pairs,work,nwork,y,K=I,N=H,_om_BN=_om_BN,_om_BK=_om_BK,_om_BM=bm,XS=True,SMB=False,HOP=False,CHK=True,PF=False,num_warps=_om_NUM_WARPS)
    out=_om_torch.empty((M,H),dtype=_om_torch.bfloat16,device=x.device)
    _post_expert_norm_weighted_sum_kernel[(M,)](y,state['norm_w'],tw,out,local,TOPK=_om_TOP_K,H=H,eps=_om_POST_NORM_EPS,APPLY_NORM=True,HAS_WEIGHT=True,WEIGHT_FIRST=True,HAS_IDS=True,BLOCK_H=_om_triton.next_power_of_2(H),num_warps=8,num_stages=2)
    return out
'''

marker = '\n_before_stack_bucket_prepare=prepare\ndef prepare(weights):'
assert base.count(marker)==1
base = base.replace(marker, '\n'+impl+marker)
old_prepare = "    state['_bucket_caps']=(sms*4,sms*2,sms*2)\n    return state"
new_prepare = "    state['_bucket_caps']=(sms*4,sms*2,sms*2)\n    om_weights=dict(weights);om_weights['post_norm_w']=weights['post_expert_norm']\n    state['operatorstack_m32']=_om_prepare_m32(om_weights)\n    return state"
assert base.count(old_prepare)==1
base=base.replace(old_prepare,new_prepare)
old_tail='''_before_stack_bucket_experts_wm=_experts_wm
def _experts_wm(x,topk_weights,local_mod_ids,state):
    if x.shape[0]==256:
        data={'hidden_states':x,'topk_ids':local_mod_ids.to(torch.int32),'topk_weights':topk_weights}
        return _decoder_bucket_forward(data,state)
    return _before_stack_bucket_experts_wm(x,topk_weights,local_mod_ids,state)'''
new_tail='''_before_stack_bucket_experts_wm=_experts_wm
def _experts_wm(x,topk_weights,local_mod_ids,state):
    if x.shape[0]==32:
        data={'hidden_states':x.contiguous(),'topk_weights':topk_weights.contiguous(),
              'topk_ids':local_mod_ids.to(torch.int32).contiguous(),
              'config':{'d_expert':D_EXPERT,'local_start':0,'local_count':LOCAL_EXPERTS}}
        return _om_m32_kernel_fn(data,state['operatorstack_m32'])
    if x.shape[0]==256:
        data={'hidden_states':x,'topk_ids':local_mod_ids.to(torch.int32),'topk_weights':topk_weights}
        return _decoder_bucket_forward(data,state)
    return _before_stack_bucket_experts_wm(x,topk_weights,local_mod_ids,state)'''
assert base.count(old_tail)==1
base=base.replace(old_tail,new_tail)
out=Path('dev/decoder_moe_hybrid/submission.py')
out.write_text(base)
print('extracted MoE dependency source bytes:',len(impl.encode()))
print(out,len(base),'bytes')
