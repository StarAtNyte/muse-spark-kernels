"""CuTe DSL Hopper TMA/WGMMA split-K vs current projection dispatch.

Uses NVIDIA's BSD-licensed example without modifying it. The standalone
submission must vendor any selected implementation with its license preserved.
"""
import importlib.util
import json
import os
import statistics
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import triton
import triton.language as tl

sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
sys.path.insert(0, '/w/cute_examples/cute/hopper/kernel/dense_gemm')
from dense_gemm import HopperWgmmaGemmKernel

tuned = bool(os.environ.get('CUTE_TUNED'))
if tuned:
    # Extend the example's conservative N-tile whitelist to native Hopper
    # WGMMA widths; retain its BSD license in the derived source.
    path = '/w/cute_examples/cute/hopper/kernel/dense_gemm/dense_gemm.py'
    source = open(path).read().replace('not in [64, 128, 256]', 'not in [16, 32, 64, 128, 256]')
    if os.environ.get('CUTE_AB_STAGE'):
        ab_stage = int(os.environ['CUTE_AB_STAGE'])
        source = source.replace('return ab_stage, epi_stage', f'return {ab_stage}, epi_stage')
    open('/w/run/cute_gemm_tuned.py', 'w').write(source)
    example_spec = importlib.util.spec_from_file_location('cute_gemm_tuned', '/w/run/cute_gemm_tuned.py')
    example = importlib.util.module_from_spec(example_spec)
    sys.modules['cute_gemm_tuned'] = example
    example_spec.loader.exec_module(example)
    HopperWgmmaGemmKernel = example.HopperWgmmaGemmKernel

spec = importlib.util.spec_from_file_location('parent', '/w/run/sub.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
bench_stream = torch.cuda.Stream()
bench_stream.wait_stream(torch.cuda.current_stream())
torch.cuda.set_stream(bench_stream)
weights = R.generate_weights(8339)
state = parent._prepare_v5(weights)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@triton.jit
def reduce_split(P, OUT, SIZE: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0., tl.float32)
    for s in range(SPLITS):
        acc += tl.load(P + s * SIZE + i, i < SIZE, 0.)
    tl.store(OUT + i, acc, i < SIZE)


def capture(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # CuTe receives an explicit stream handle. Capture that same stream;
    # PyTorch's implicit capture stream would omit the GEMM from the graph.
    with torch.cuda.graph(g, stream=torch.cuda.current_stream()):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def elapsed(g):
    flush.add_(1)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000


print('GPU', torch.cuda.get_device_name(), 'CuTe', cutlass.__file__, flush=True)
for layer in os.environ.get('CUTE_LAYERS', 'qkv_gate,pooler,unpooler').split(','):
    weight = weights[layer]
    n, k = weight.shape
    for batch in map(int, os.environ.get('CUTE_BATCHES', '16,32,128,256').split(',')):
        data = R.generate_input(batch, layer, 3719 + batch)
        x = data['x']
        gb = capture(lambda: parent._kernel_v5(data, state))
        configs = (((64, 16, 4), (64, 16, 8), (64, 32, 4), (64, 64, 1),
                    (64, 64, 2), (64, 128, 1)) if tuned else
                   ((64, 64, 2), (64, 64, 4), (64, 64, 8),
                    (128, 64, 4), (64, 128, 4)))
        if os.environ.get('CUTE_AB_STAGE'):
            configs = [(64, tn, sp) for tn in (16, 32) for sp in (4, 8, 16)]
        for tm, tn, splits in configs:
            try:
                a = weight.view(n, splits, k // splits).permute(0, 2, 1)
                b = x.view(batch, splits, k // splits).permute(0, 2, 1)
                partial = torch.empty((splits, batch, n), dtype=torch.float32, device='cuda')
                output = torch.empty((batch, n), dtype=torch.bfloat16, device='cuda')
                c = partial.permute(2, 1, 0)
                ma, mb, mc = [from_dlpack(t, assumed_align=16) for t in (a, b, c)]
                gemm = HopperWgmmaGemmKernel(cutlass.Float32, (tm, tn), (1, 1))
                if tuned:
                    gemm.occupancy = 2
                compiled = cute.compile(gemm, ma, mb, mc, stream)

                def fn():
                    compiled(ma, mb, mc, stream)
                    reduce_split[(triton.cdiv(batch*n, 256),)](
                        partial, output, batch*n, splits, 256)

                gc = capture(fn)
                # Prove replay recomputes from fresh activations, rather than
                # timing only the reduction over an earlier GEMM result.
                x.copy_(R.generate_input(batch, layer, 9481 + batch)['x'])
                gc.replay()
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(data, output, weights)
                if not ok:
                    raise ValueError(msg)
                ratios, bt, ct = [], [], []
                for rep in range(12):
                    if rep % 2:
                        t, baseline = elapsed(gc), elapsed(gb)
                    else:
                        baseline, t = elapsed(gb), elapsed(gc)
                    ratios.append(baseline / t)
                    bt.append(baseline)
                    ct.append(t)
                print('RESULT', json.dumps(dict(layer=layer, batch=batch, tile=[tm, tn],
                      splits=splits, baseline_us=statistics.median(bt),
                      candidate_us=statistics.median(ct), paired_speedup=statistics.median(ratios))), flush=True)
            except Exception as exc:
                print('FAIL', layer, batch, tm, tn, splits, type(exc).__name__, str(exc)[:700], flush=True)
