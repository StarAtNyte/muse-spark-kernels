"""Distribute softmax exponentials across SFU and FP32 FMA pipelines."""
exec(open('/w/modal/drv_attention_merge.py').read().split('for batch in (3, 16, 32, 33):')[0])
from torch.utils.cpp_extension import load_inline
helper = r'''
__device__ __forceinline__ float poly_ex2(float x) {
  // Degree-5 Taylor on [-0.5,0.5], FP32 throughout. Preserve extremes
  // with the original SFU path, including masked negative infinities.
  if (!(x > -125.f && x < 125.f)) return ex2f(x);
  const int n = __float2int_rn(x);
  const float f = x - (float)n;
  float p = 0.001333355814642844f;
  p = fmaf(p, f, 0.009618129107628477f);
  p = fmaf(p, f, 0.05550410866482158f);
  p = fmaf(p, f, 0.2402265069591007f);
  p = fmaf(p, f, 0.6931471805599453f);
  p = fmaf(p, f, 1.f);
  return p * __int_as_float((n + 127) << 23);
}
'''
modules = {}
for mode in ('half', 'all'):
    src = candidate._WG_CUDA
    marker = '__device__ __forceinline__ void wg_pv72('
    src = src.replace(marker, helper + '\n' + marker)
    start = src.index('    uint32_t pa[8][2];')
    end = src.index('    wg_fence();', start)
    block = src[start:end]
    fn = 'poly_ex2' if mode == 'all' else '((j & 1) ? poly_ex2 : ex2f)'
    block = block.replace('ex2f(fmaf(', fn + '(fmaf(')
    src = src[:start] + block + src[end:]
    modules[mode] = load_inline('verify_exp_' + mode, candidate._WG_CPP,
        cuda_sources=src, functions=['wg5_run_t'],
        extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a'], verbose=False)
for batch in (8, 16, 32):
    for seed in (313, 9587):
        data = R.generate_input(batch, 80000, 'tp8', 'full', seed,
            q_len=8, spread='agentic', dist='coherent', page=1)
        gb, ob = capture(base, data)
        for mode, module in modules.items():
            candidate._WG_MOD = module
            gc, oc = capture(candidate, data)
            error = ((ob.float()-oc.float()).norm()/ob.float().norm()).item()
            if not torch.isfinite(oc).all().item() or error > .001:
                raise AssertionError((batch,seed,mode,error))
            bt,ct,ratios=[],[],[]
            for rep in range(24):
                if rep%2: c,b=elapsed(gc),elapsed(gb)
                else: b,c=elapsed(gb),elapsed(gc)
                bt.append(b);ct.append(c);ratios.append(b/c)
            print('RESULT',json.dumps(dict(batch=batch,seed=seed,mode=mode,error=error,
                speedup=statistics.median(ratios),baseline_us=statistics.median(bt),
                candidate_us=statistics.median(ct))),flush=True)
            del gc,oc
        del gb,ob,data
