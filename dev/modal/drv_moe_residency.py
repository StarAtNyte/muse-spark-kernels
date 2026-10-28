"""Compare explicit activation-cache PTX with the validated native MoE path.

Modal only: SGLang imports are stubbed because these tests exercise the native
path. The complete standalone submission must still pass the official worker.
"""
import copy
import importlib.util
import inspect
import json
import os
import statistics
import sys
import time
from unittest.mock import MagicMock

import torch
from torch.utils.cpp_extension import load_inline


class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = MagicMock()
        module.__path__ = []
        module.__name__ = spec.name
        module.__spec__ = spec
        return module

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R

spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
# Run the file's own equivalent Triton reduction for large batches as well,
# because the SGLang package is absent from this diagnostic image.
exec(inspect.getsource(sub._finish_experts).replace('if tokens >= 32:', 'if False:'), sub.__dict__)
# Generate the same raw format for this GPU's local bank only.
R.N_EXPERTS = 32
weights = R.generate_weights(93819)
R.N_EXPERTS = 256
sub._prepare_marlin = lambda w: {'norm_w': (1 + w['post_norm_w'].float()).bfloat16()}
state = sub.prepare(weights)
base_mod = state['wm']['mod']
print('READY', torch.cuda.get_device_name(), flush=True)

HELPER = r'''
__device__ __forceinline__ uint4 ptx_activation_load(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
'''
source = sub._WM_CUDA
source = source.replace('typedef __nv_bfloat162 bf162;', 'typedef __nv_bfloat162 bf162;\n' + HELPER)
for old in ('__ldcg(xv)', '__ldcg(xv + 1)'):
    source = source.replace(old, old.replace('__ldcg', 'ptx_activation_load'))

variants = [('minb2_cg', sub._WM_CUDA, ['-DWM_MINB=2']), ('minb3_cg', sub._WM_CUDA, ['-DWM_MINB=3']), ('minb1_cg', sub._WM_CUDA, ['-DWM_MINB=1'])]
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')


def capture(data, module):
    state['wm']['mod'] = module
    result = sub.kernel_fn(data, state)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = sub.kernel_fn(data, state)
    graph.replay()
    torch.cuda.synchronize()
    return graph, result


def elapsed(graph):
    flush.add_(1)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


for name, cuda, flags in variants:
    if name == 'l1_ca':
        continue  # Completed in moe_cache_v2.log: 0.997685x, reject.
    start = time.time()
    module = load_inline('moe_ptx_' + name, sub._WM_CPP, cuda_sources=cuda,
                         functions=['wm_route', 'wm_gemm_run', 'wm_set_pdl_t',
                                    'wm_set_persist_t', 'wm_finish', 'wm_pf'],
                         extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'] + flags,
                         verbose=False)
    module.wm_set_pdl_t(1)
    print('COMPILED', name, time.time() - start, flush=True)
    gains = []
    for batch in (8, 16, 32, 128):
        for seed in (419, 2387):
            data = R.generate_input(batch, 'ep8', seed + batch)
            base_graph, base = capture(data, base_mod)
            graph, result = capture(data, module)
            # Both outputs are independently allocated and retained by their graphs.
            error = ((result.float() - base.float()).norm() /
                     base.float().norm().clamp_min(1e-20)).item()
            if not torch.isfinite(result).all().item() or error > 0.001:
                raise AssertionError((name, batch, seed, error))
            ratios, base_times, times = [], [], []
            for repeat in range(16):
                if repeat % 2:
                    candidate_us, base_us = elapsed(graph), elapsed(base_graph)
                else:
                    base_us, candidate_us = elapsed(base_graph), elapsed(graph)
                ratios.append(base_us / candidate_us)
                base_times.append(base_us)
                times.append(candidate_us)
            speedup = statistics.median(ratios)
            gains.append(speedup)
            print('RESULT', json.dumps(dict(variant=name, batch=batch, seed=seed,
                  speedup=speedup, baseline_us=statistics.median(base_times),
                  candidate_us=statistics.median(times), relative_error=error)), flush=True)
    print('AGGREGATE', name, statistics.geometric_mean(gains), flush=True)
    for batch, dist in ((4, 'normal'), (33, 'normal'), (128, 'skewed')):
        data = R.generate_input(batch, 'ep8', 777 + batch, dist=dist)
        _, result = capture(data, module)
        ok, message = R.check_implementation(data, result, weights)
        print('CHECK', name, batch, dist, ok, message, flush=True)
        if not ok:
            raise AssertionError((name, batch, dist, message))
