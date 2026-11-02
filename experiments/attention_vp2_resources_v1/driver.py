"""Attention paired screen: scored source (/w/kernels) vs candidate (/w/run/sub.py) on the 12 scored shapes;
each output reference-checked; alternating cold-L2 graph replays."""
import importlib.util, json, statistics, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
import ast, base64, zlib
from pathlib import Path
source = Path('/w/run/sub.py').read_text()
tree = ast.parse(source)
enc = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode' and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str) and len(n.args[0].value) > 1000]
assert len(enc) == 1
inner = zlib.decompress(base64.b64decode(enc[0])).decode()
a = '__global__ void __launch_bounds__(256, 4)\nvp2_kernel'
b = '__global__ void __launch_bounds__(256, 3)\nvp2_kernel'
assert inner.count(a) == 1
base_src = source.replace(enc[0], base64.b64encode(zlib.compress(inner.replace(a,b).encode(),9)).decode(), 1)
Path('/w/run/base_lb3.py').write_text(base_src)
B = load('base', '/w/run/base_lb3.py')
C = load('cand', '/w/run/sub.py')

# The existing driver constructs B=LB3 and C=LB4 from the submitted LB4 file.
base3=Path('/w/run/base_lb3.py').read_text()
tree=ast.parse(base3)
enc=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and len(n.args[0].value)>1000]
assert len(enc)==1
inner=zlib.decompress(base64.b64decode(enc[0])).decode()
a='__global__ void __launch_bounds__(256, 3)\nvp2_kernel'; b='__global__ void __launch_bounds__(256, 2)\nvp2_kernel'
assert inner.count(a)==1
Path('/w/run/base_lb2.py').write_text(base3.replace(enc[0],base64.b64encode(zlib.compress(inner.replace(a,b).encode(),9)).decode(),1))
A=load('base2','/w/run/base_lb2.py')
import subprocess, shutil
tool=shutil.which('cuobjdump')
if not tool: raise RuntimeError('cuobjdump unavailable')
for label,mod in [('lb2',A),('lb3',B),('lb4',C)]:
    mod._kz_attn._zh_join()
    ext=mod._kz_attn._ZH['vfg']
    p=subprocess.run([tool,'--dump-resource-usage',ext.__file__],text=True,capture_output=True,check=True)
    lines=p.stdout.splitlines()
    hits=[i for i,x in enumerate(lines) if 'vp2_kernel' in x]
    if not hits: raise RuntimeError('VP2 resource entry not found')
    for i in hits: print('RESOURCE',label,repr(lines[i:i+4]),flush=True)
print('DONE',flush=True)
