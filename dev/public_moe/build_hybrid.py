"""Shape-selected MoE hybrid, with complete official validation required."""
from pathlib import Path
import ast
public=Path('dev/public_moe/operatorstack_v1.py').read_text()
tree=ast.parse(public)
bound=set()
for node in tree.body:
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)):bound.add(node.name)
    elif isinstance(node,(ast.Import,ast.ImportFrom)):
        for alias in node.names:bound.add(alias.asname or alias.name.split('.')[0])
    elif isinstance(node,(ast.Assign,ast.AnnAssign)):
        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
        for target in targets:bound.update(n.id for n in ast.walk(target) if isinstance(n,ast.Name))
rename={name:'_om_'+name for name in bound}
class Namespace(ast.NodeTransformer):
    def visit_Name(self,node):node.id=rename.get(node.id,node.id);return node
    def visit_arg(self,node):node.arg=rename.get(node.arg,node.arg);return self.generic_visit(node)
    def visit_FunctionDef(self,node):node.name=rename.get(node.name,node.name);return self.generic_visit(node)
    def visit_ClassDef(self,node):node.name=rename.get(node.name,node.name);return self.generic_visit(node)
    def visit_Global(self,node):node.names=[rename.get(n,n) for n in node.names];return node
    def visit_keyword(self,node):
        if node.arg is not None:node.arg=rename.get(node.arg,node.arg)
        return self.generic_visit(node)
    def visit_Import(self,node):
        for alias in node.names:alias.asname=rename.get(alias.asname or alias.name.split('.')[0],alias.asname)
        return node
    def visit_ImportFrom(self,node):
        for alias in node.names:alias.asname=rename.get(alias.asname or alias.name,alias.asname)
        return node
namespaced=ast.unparse(ast.fix_missing_locations(Namespace().visit(tree)))
base=Path('experiments/moe_bucket_row_m256_v1/submission.py').read_text()
license=Path('dev/public_moe/LICENSE.operatorstack').read_text()
notice=Path('dev/public_moe/NOTICE.operatorstack').read_text()
source=base+'\n# Component: MIT team bigboateng, operatorstack/open-kernel-challenge-solutions\n# Commit 6d9b77656b6c295e421be9e1424b0f8f99299b0a. Namespace-only adaptation.\n'+''.join('# '+l+'\n' for l in (license+'\n'+notice).splitlines())+'\n'+namespaced+"""

_native_bucket_prepare=prepare
def prepare(weights):
    state=_native_bucket_prepare(weights)
    state['operatorstack']=_om_prepare(weights)
    return state
_native_bucket_kernel_fn=kernel_fn
def kernel_fn(data,state):
    c=data['config'];m=data['hidden_states'].shape[0]
    if (m in (32,128) and c['local_start']==0 and c['local_count']==32
            and c['d_hidden']==4096 and c['d_expert']==4096):
        return _om_kernel_fn(data,state['operatorstack'])
    return _native_bucket_kernel_fn(data,state)
"""
ast.parse(source);assert len(source.encode())<1_000_000
out=Path('dev/public_moe/hybrid_m32_m128_v2.py');out.write_text(source);print(out,len(source.encode()))
