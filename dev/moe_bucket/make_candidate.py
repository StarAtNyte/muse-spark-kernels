"""Build complete official candidates; never submit the diagnostic CUDA bundle."""
import argparse
import ast
import runpy
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--variant',choices=['row','native','native_ca'],required=True)
p.add_argument('--min-batch',type=int,default=256)
a=p.parse_args()
s=Path('dev/ptx_campaign/moe_bucket_v1.py').read_text()
if a.variant!='row':
    # Reuse the exact screened transformation on the full CUDA, retaining all
    # original dispatch instantiations for fallback shapes.
    t=ast.parse(s)
    node=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_WM_CUDA' for x in n.targets))
    cu=ast.literal_eval(node.value)
    script=Path('dev/moe_bucket/build_native.py').read_text()
    start=script.index("s=s.replace('__launch_bounds__")
    end=script.index("Path('dev/moe_bucket/native.cu')")
    ns={'s':cu}
    exec(script[start:end],ns)
    out=ns['ca' if a.variant=='native_ca' else 's']
    lines=s.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=['_WM_CUDA = '+repr(out)+'\n']
    s=''.join(lines)
s=s.replace('128<=m<=256',f'{a.min_batch}<=m<=256')
ast.parse(s)
target=Path(f'dev/ptx_campaign/moe_bucket_{a.variant}_m{a.min_batch}.py')
target.write_text(s)
print(target)
