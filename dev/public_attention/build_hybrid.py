"""Combine attributed public attention with frozen native/cluster attention.

Dispatch uses shapes only. Official validation is required before promotion.
"""
from pathlib import Path
import ast

public=Path('dev/public_attention/operatorstack_v1.py').read_text()
tree=ast.parse(public)
bound=set()
for node in tree.body:
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)):bound.add(node.name)
    elif isinstance(node,(ast.Import,ast.ImportFrom)):
        for alias in node.names:bound.add(alias.asname or alias.name.split('.')[0])
    elif isinstance(node,(ast.Assign,ast.AnnAssign)):
        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
        for target in targets:bound.update(n.id for n in ast.walk(target) if isinstance(n,ast.Name))
rename={name:'_os_'+name for name in bound}
class Namespace(ast.NodeTransformer):
    def visit_Name(self,node):node.id=rename.get(node.id,node.id);return node
    def visit_arg(self,node):node.arg=rename.get(node.arg,node.arg);return self.generic_visit(node)
    def visit_FunctionDef(self,node):node.name=rename.get(node.name,node.name);return self.generic_visit(node)
    def visit_ClassDef(self,node):node.name=rename.get(node.name,node.name);return self.generic_visit(node)
    def visit_Global(self,node):node.names=[rename.get(n,n) for n in node.names];return node
    def visit_Import(self,node):
        for alias in node.names:alias.asname=rename.get(alias.asname or alias.name.split('.')[0],alias.asname)
        return node
    def visit_ImportFrom(self,node):
        for alias in node.names:alias.asname=rename.get(alias.asname or alias.name,alias.asname)
        return node
namespaced=ast.unparse(ast.fix_missing_locations(Namespace().visit(tree)))
base=Path('experiments/attention_fp8p_cluster_v1/submission.py').read_text()
license=Path('dev/public_attention/LICENSE.operatorstack').read_text()
source=base+'\n# MIT component: team bigboateng, operatorstack/open-kernel-challenge-solutions\n# Commit 6d9b77656b6c295e421be9e1424b0f8f99299b0a; namespace-only adaptation.\n'+''.join('# '+line+'\n' for line in license.splitlines())+'\n'+namespaced+'''

_native_cluster_attention=kernel_fn
def kernel_fn(data):
    cfg=data['config']
    if (cfg['q_len']==8 and (cfg['window'] or cfg['batch']>4) and cfg['page_size']==1
            and cfg['num_q_heads']==16 and cfg['num_kv_heads']==2 and cfg['head_dim']==64):
        return _os_kernel_fn(data)
    return _native_cluster_attention(data)
'''
ast.parse(source)
assert len(source.encode())<1024*1024
out=Path('dev/public_attention/hybrid_verify_v2.py');out.write_text(source)
print(out,len(source.encode()))
