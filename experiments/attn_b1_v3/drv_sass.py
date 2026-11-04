"""Build the b1 module with ptxas -v and dump per-kernel register / spill / local-memory facts."""
import importlib.util, os, subprocess, sys, glob, re
from unittest.mock import MagicMock
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
from torch.utils.cpp_extension import load_inline
flags = os.environ.get('FLAGS', '').split()
m = load_inline(name='b1sass' + str(abs(hash(tuple(flags))) % 1000), cpp_sources=C.B1_CPP_SRC, cuda_sources=C.B1_CUDA_SRC, functions=['run_b1'],
                extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-Xptxas=-v', *flags], verbose=True)
so = m.__file__
out = subprocess.run(['cuobjdump', '-sass', so], capture_output=True, text=True).stdout
for fn in re.split(r'\n\s*Function : ', out)[1:]:
    name = fn.split('\n', 1)[0]
    body = fn
    print('FN', name[:90], 'instr', len(re.findall(r'/\*[0-9a-f]{4,6}\*/ ', body)), 'LDL', body.count('LDL'), 'STL', body.count('STL'),
          'STG', body.count('STG'), 'LDG', body.count('LDG'), 'MEMBAR', body.count('MEMBAR'), 'CALL', body.count('CALL'), 'BRA', body.count('BRA'))
if os.environ.get('DUMP'):
    for fn in re.split(r'\n\s*Function : ', out)[1:]:
        if os.environ['DUMP'] in fn.split('\n', 1)[0]:
            lines = [l for l in fn.split('\n') if re.search(r'STG|LDG|MEMBAR|CCTL|ERRBAR|FENCE', l)]
            print('\n'.join(lines[:80]))
print('DONE')
