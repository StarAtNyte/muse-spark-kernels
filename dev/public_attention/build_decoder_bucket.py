"""Add the validated M256 count-tiered MoE path to the attention decoder stack."""
import ast,argparse
parser=argparse.ArgumentParser()
parser.add_argument("--base",default="dev/public_attention/decoder_stack_v1.py")
parser.add_argument("--out",default="dev/public_attention/decoder_stack_bucket_v1.py")
args=parser.parse_args()
from pathlib import Path
base=Path(args.base).read_text()
moe=Path('experiments/moe_bucket_row_m256_v1/submission.py').read_text();mt=ast.parse(moe)
def value(name):
 return next(ast.literal_eval(n.value) for n in mt.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
changes=[]
for n in ast.parse(base).body:
 if isinstance(n,ast.Assign):
  for target in n.targets:
   if isinstance(target,ast.Name) and target.id in ('_WM_CUDA','_WM_CPP'):
    changes.append((n.lineno-1,n.end_lineno,target.id+'='+repr(value(target.id))+'\n'))
 if isinstance(n,ast.FunctionDef) and n.name=='_wm_module':
  for call in ast.walk(n):
   if isinstance(call,ast.Call):
    for kw in call.keywords:
     if kw.arg=='functions' and isinstance(kw.value,ast.List):kw.value.elts.append(ast.Constant(value='wm_bucket_run'))
  changes.append((n.lineno-1,n.end_lineno,ast.unparse(n).replace('muse_wm_moe','muse_decoder_wm_bucket')+'\n'))
lines=base.splitlines(keepends=True)
for a,b,text in sorted(changes,reverse=True):lines[a:b]=[text]
source=''.join(lines)
node=next(n for n in mt.body if isinstance(n,ast.FunctionDef) and n.name=='_bucket_forward')
forward=ast.get_source_segment(moe,node).replace('def _bucket_forward(','def _decoder_bucket_forward(').replace('_WM_NOLS',"wm['nols']").replace("state['norm_w']","state['post_expert_w']")
source+='\n'+forward+'''
_before_stack_bucket_prepare=prepare
def prepare(weights):
    state=_before_stack_bucket_prepare(weights)
    device=weights['w13'].device
    state['_bucket_streams']=[torch.cuda.Stream(device) for _ in range(3)]
    sms=torch.cuda.get_device_properties(device).multi_processor_count
    state['_bucket_caps']=(sms*4,sms*2,sms*2)
    return state
_before_stack_bucket_experts_wm=_experts_wm
def _experts_wm(x,topk_weights,local_mod_ids,state):
    if x.shape[0]==256:
        data={'hidden_states':x,'topk_ids':local_mod_ids.to(torch.int32),'topk_weights':topk_weights}
        return _decoder_bucket_forward(data,state)
    return _before_stack_bucket_experts_wm(x,topk_weights,local_mod_ids,state)
'''
ast.parse(source);assert len(source.encode())<1_000_000,len(source.encode())
p=Path(args.out);p.write_text(source);print(p,len(source.encode()))
