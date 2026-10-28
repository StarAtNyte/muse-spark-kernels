"""Keep Torch bindings in host C++ rather than reparsing them with nvcc.

Diagnostic helper only. Device source and launch flags are unchanged.
"""
from torch.utils.cpp_extension import load_inline


def build(name, cpp, cuda, functions, flags):
    marker = '#include <torch/extension.h>'
    split = cuda.find(marker)
    if split <= 0:
        raise ValueError('expected a trailing Torch binding block')
    device, host = cuda[:split], cuda[split:]
    if 'torch::' in device or '__global__' in host or '<<<' in host:
        raise ValueError('source does not cleanly separate kernels and host bindings')
    return load_inline(name, cpp + '\n' + host, cuda_sources=device,
        functions=functions, extra_cuda_cflags=flags, verbose=False)
