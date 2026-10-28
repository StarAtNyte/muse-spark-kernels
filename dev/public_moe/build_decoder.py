"""Integrate independently validated M32/M128 MoE winners into frozen decoder."""
from pathlib import Path
import ast,base64,zlib,runpy,argparse
parser=argparse.ArgumentParser()
parser.add_argument('--base',default='experiments/decoder_stack_v3/submission.py')
parser.add_argument('--out',default='dev/public_moe/decoder_moe_stack_v1.py')
parser.add_argument('--m32-only',action='store_true',help='dispatch public MoE only for 32-row inputs')
args=parser.parse_args()
# Reuse the exact namespace transform that passed the standalone hybrid check.
component=runpy.run_path('dev/public_moe/build_hybrid.py')['namespaced']
base=Path(args.base).read_text();tree=ast.parse(base);lines=base.splitlines(keepends=True);changes=[]
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_DENSE_CUDA','_WM_CUDA') for t in n.targets):
  name=n.targets[0].id;text=ast.literal_eval(n.value)
  encoded=base64.b64encode(zlib.compress(text.encode(),9)).decode()
  assert zlib.decompress(base64.b64decode(encoded)).decode()==text
  changes.append((n.lineno-1,n.end_lineno,name+'=_stack_zsrc('+repr(encoded)+')\n'))
for a,b,replacement in sorted(changes,reverse=True):lines[a:b]=[replacement]
base=''.join(lines)
license=Path('dev/public_moe/LICENSE.operatorstack').read_text()
notice=Path('dev/public_moe/NOTICE.operatorstack').read_text()
source=base+'\n# MoE component: team bigboateng, MIT, namespace-only adaptation.\n# operatorstack/open-kernel-challenge-solutions @ 6d9b77656b6c295e421be9e1424b0f8f99299b0a\n'+''.join('# '+l+'\n' for l in (license+'\n'+notice).splitlines())+component+'''

_before_public_moe_prepare=prepare
def prepare(weights):
    state=_before_public_moe_prepare(weights)
    moe_weights=dict(weights)
    moe_weights['post_norm_w']=weights['post_expert_norm']
    state['stack_public_moe']=_om_prepare(moe_weights)
    state['stack_public_moe']['norm_w']=state['post_expert_w']
    return state
_before_public_moe_experts_wm=_experts_wm
def _experts_wm(x,topk_weights,local_mod_ids,state):
    if x.shape[0] in (32,128):
        m=x.shape[0]
        cfg={'bs':m,'d_hidden':4096,'d_expert':4096,'n_experts':256,
             'local_start':0,'local_count':32,'top_k':8,'sharding':'ep8'}
        data={'hidden_states':x,'topk_ids':local_mod_ids.to(torch.int32),
              'topk_weights':topk_weights,'config':cfg}
        return _om_kernel_fn(data,state['stack_public_moe'])
    return _before_public_moe_experts_wm(x,topk_weights,local_mod_ids,state)
'''
if args.m32_only:
    old='if x.shape[0] in (32,128):'
    assert source.count(old)==1
    source=source.replace(old,'if x.shape[0] == 32:')
ast.parse(source);assert len(source.encode())<1_000_000,len(source.encode())
p=Path(args.out);p.write_text(source);print(p,len(source.encode()))
