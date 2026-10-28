"""Stack our native/cluster attention and attributed Gluon attention into decoder.

Keeps best validated projection, FP32 router and normalization overlap. The
attention dispatch remains experimental until complete official checks pass.
"""
from pathlib import Path
import ast
import base64,zlib
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--attention",default="dev/public_attention/hybrid_full_v1.py")
parser.add_argument("--out",default="dev/public_attention/decoder_stack_v1.py")
args=parser.parse_args()

upstream=Path(args.attention).read_text()
tree=ast.parse(upstream);bound=set()
for n in tree.body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):bound.add(n.name)
    elif isinstance(n,(ast.Import,ast.ImportFrom)):
        for a in n.names:bound.add(a.asname or a.name.split('.')[0])
    elif isinstance(n,(ast.Assign,ast.AnnAssign)):
        for target in (n.targets if isinstance(n,ast.Assign) else [n.target]):
            bound.update(x.id for x in ast.walk(target) if isinstance(x,ast.Name))
rename={name:'_ha_'+name for name in bound}
class Namespace(ast.NodeTransformer):
    def visit_Name(self,n):n.id=rename.get(n.id,n.id);return n
    def visit_arg(self,n):n.arg=rename.get(n.arg,n.arg);return self.generic_visit(n)
    def visit_FunctionDef(self,n):n.name=rename.get(n.name,n.name);return self.generic_visit(n)
    def visit_ClassDef(self,n):n.name=rename.get(n.name,n.name);return self.generic_visit(n)
    def visit_Global(self,n):n.names=[rename.get(x,x) for x in n.names];return n
    def visit_Import(self,n):
        for a in n.names:a.asname=rename.get(a.asname or a.name.split('.')[0],a.asname)
        return n
    def visit_ImportFrom(self,n):
        for a in n.names:a.asname=rename.get(a.asname or a.name,a.asname)
        return n
component=ast.unparse(ast.fix_missing_locations(Namespace().visit(tree)))
base=Path('experiments/decoder_public_projection_router_v2/submission.py').read_text()
start=base.index('    q = torch.empty((tokens, hq, d)',base.index('def kernel_fn(data, state):'))
end=base.index('    attn_output = _dgemm(o,',start)
base=base[:start]+'''    attn_data=dict(data)
    attn_data['qkv']=qkv.view(batch,length,-1)
    o=_ha_kernel_fn(attn_data).view(tokens,hq*d)
'''+base[end:]
# Legacy attention helpers no longer have any caller. Remove their CUDA strings
# and helpers so the single-file submission remains below the 1 MiB limit.
t=ast.parse(base);lines=base.splitlines(keepends=True);cuts=[]
for n in t.body:
    names=[]
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):names=[n.name]
    elif isinstance(n,ast.Assign):names=[x.id for x in n.targets if isinstance(x,ast.Name)]
    if names and all(x.startswith(('_FD_','_fd_','_SP_','_sp_','_SLIDE_','_slide_')) or x in ('_prepare_qkv','_sliding_partials','_merge_norm_gate') for x in names):
        cuts.append((min([n.lineno]+[d.lineno for d in getattr(n,"decorator_list",[])])-1,n.end_lineno))
for a,b in reversed(cuts):del lines[a:b]
base=''.join(lines)
license=Path('dev/public_attention/LICENSE.operatorstack').read_text()
source=base+'\n# Attention component: our FP8/cluster paths plus MIT bigboateng Gluon paths.\n# operatorstack/open-kernel-challenge-solutions @ 6d9b77656b6c295e421be9e1424b0f8f99299b0a\n'+''.join('# '+l+'\n' for l in license.splitlines())+component+'\n'
# Losslessly compress the fixed native CUDA source, as the projection backend
# already does for its templates. No input data or executable Python is encoded.
parsed=ast.parse(source)
node=next(n for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ha__FD_CUDA' for t in n.targets))
cuda=ast.literal_eval(node.value)
encoded=base64.b64encode(zlib.compress(cuda.encode(),9)).decode()
lines=source.splitlines(keepends=True)
lines[node.lineno-1:node.end_lineno]=['_ha__FD_CUDA=_stack_zsrc('+repr(encoded)+')\n']
source=''.join(lines)
helper="import base64 as _stack_base64\nimport zlib as _stack_zlib\ndef _stack_zsrc(data):\n    return _stack_zlib.decompress(_stack_base64.b64decode(data)).decode()\n"
first,rest=source.split('\n',1);source=first+'\n'+helper+rest
assert zlib.decompress(base64.b64decode(encoded)).decode()==cuda
ast.parse(source)
assert len(source.encode())<1_000_000,len(source.encode())
out=Path(args.out);out.write_text(source)
print(out,len(source.encode()))
