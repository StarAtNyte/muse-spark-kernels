"""Tune WGMMA staging and split-KV with paired cold-cache graph timing."""
exec(open('/w/modal/drv_attention_merge.py').read().split('for batch in (3, 16, 32, 33):')[0])
from torch.utils.cpp_extension import load_inline
modules = {}
for stages, residency in ((2, 2), (2, 3), (3, 1), (3, 2)):
    src = candidate._WG_CUDA.replace('#define WS 3', f'#define WS {stages}')
    src = src.replace('__launch_bounds__(256, 2)', f'__launch_bounds__(256, {residency})')
    src = src.replace('  dim3 grid(splits, 1, B);', '  cudaFuncSetAttribute(wg5_partials, cudaFuncAttributePreferredSharedMemoryCarveout, 100);\n  dim3 grid(splits, 1, B);')
    modules[stages, residency] = load_inline(f'verify_res_{stages}_{residency}', candidate._WG_CPP,
        cuda_sources=src, functions=['wg5_run_t'],
        extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-Xptxas=-v'], verbose=False)
for batch in (16, 32):
    for seed in (313, 9587):
        data = R.generate_input(batch, 80000, 'tp8', 'full', seed,
            q_len=8, spread='agentic', dist='coherent', page=1)
        gb, ob = capture(base, data)
        for (stages, residency), module in modules.items():
            candidate._WG_MOD = module
            for splits in (16, 32, 64):
                candidate._WG_SPLITS = splits
                gc, oc = capture(candidate, data)
                error = ((ob.float()-oc.float()).norm()/ob.float().norm()).item()
                if not torch.isfinite(oc).all().item() or error > .003:
                    raise AssertionError((batch, seed, stages, splits, error))
                bt, ct, ratios = [], [], []
                for rep in range(16):
                    if rep % 2: c,b = elapsed(gc),elapsed(gb)
                    else: b,c = elapsed(gb),elapsed(gc)
                    bt.append(b); ct.append(c); ratios.append(b/c)
                print('RESULT', json.dumps(dict(batch=batch,seed=seed,stages=stages,
                    splits=splits,residency=residency,error=error,speedup=statistics.median(ratios),
                    baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
                del gc, oc
        del gb, ob, data
