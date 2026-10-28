"""Namespace the MIT-licensed public projection implementation into our decoder."""
from pathlib import Path
import ast

upstream=Path('dev/public_projection/nathen_v18.py').read_text()
tree=ast.parse(upstream)
bound=set()
for n in tree.body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):bound.add(n.name)
    elif isinstance(n,(ast.Import,ast.ImportFrom)):
        for a in n.names:bound.add(a.asname or a.name.split('.')[0])
    elif isinstance(n,ast.Assign):
        for t in n.targets:
            bound.update(x.id for x in ast.walk(t) if isinstance(x,ast.Name))
rename={name:'_np_'+name for name in bound}
class Namespace(ast.NodeTransformer):
    def visit_Name(self,n):
        n.id=rename.get(n.id,n.id);return n
    def visit_arg(self,n):
        n.arg=rename.get(n.arg,n.arg);return self.generic_visit(n)
    def visit_FunctionDef(self,n):
        n.name=rename.get(n.name,n.name);return self.generic_visit(n)
    def visit_ClassDef(self,n):
        n.name=rename.get(n.name,n.name);return self.generic_visit(n)
    def visit_Global(self,n):
        n.names=[rename.get(x,x) for x in n.names];return n
    def visit_Import(self,n):
        for a in n.names:a.asname=rename[a.asname or a.name.split('.')[0]]
        return n
    def visit_ImportFrom(self,n):
        for a in n.names:a.asname=rename[a.asname or a.name]
        return n
public=ast.unparse(ast.fix_missing_locations(Namespace().visit(tree)))
base=Path('experiments/decoder_pre_norm_overlap_v1/submission.py').read_text()
license=Path('dev/public_projection/LICENSE.nathen').read_text()
notice='\n'.join('# '+l for l in license.splitlines())
source=base+'\n# Projection component from Nathen Fernandes, upstream commit\n# 86aff75e6aca72511bf9b41b6dc190bf1e763373. Namespace-only adaptation.\n'+notice+'\n'+public+r'''

# The public backend owns every dense projection, so do not build unused
# legacy BF16 tile buffers or its CUDA extension during decoder preparation.
def _dense_prepare(weights,device):
    return {}
_before_public_projection_prepare=prepare
def prepare(weights):
    state=_before_public_projection_prepare(weights)
    names=('qkv_gate','o','pooler','unpooler','router')
    state['public_projection']=_np_prepare({name:weights[name] for name in names})
    return state
_before_public_projection_dgemm=_dgemm
def _dgemm(x,layer,n,state):
    x=x.contiguous()
    data={'x':x,'config':{'m':x.shape[0],'n':n,'k':x.shape[1],'layer':layer}}
    return _np_kernel_fn(data,state['public_projection'])
'''
ast.parse(source)
assert len(source.encode())<1024*1024,len(source.encode())
Path('dev/ptx_campaign/decoder_public_projection_v1.py').write_text(source)
print('Combined source bytes:',len(source.encode()))
# Preserve copyright in extracted inspection copies too.
for p in Path('dev/public_projection/expanded').glob('*.cu'):
    text=p.read_text()
    if 'Copyright (c) 2026 Nathen Fernandes' not in text:p.write_text('/*\n'+license+'*/\n'+text)
