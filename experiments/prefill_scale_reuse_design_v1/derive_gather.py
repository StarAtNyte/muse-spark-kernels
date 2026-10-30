"""Map the abstract K64 permutation onto four gather threads, 16 BF16s each."""
import importlib.util
from pathlib import Path
p=Path(__file__).with_name('derive_permutation.py')
s=importlib.util.spec_from_file_location('perm',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
for dst in range(4):
    src=[m.perm[dst*16+i] for i in range(16)]
    by_thread={k//16 for k in src}
    assert by_thread==set(range(4))
    assert all(sum(k//16==t for k in src)==4 for t in range(4))
    print(f'destination thread {dst}: source lanes {[k//16 for k in src]}, offsets {[k%16 for k in src]}')
# Model gather's original pair-load distribution; the output needs values from all four threads.
print('PASS: each destination thread needs four values from each of four source threads')
