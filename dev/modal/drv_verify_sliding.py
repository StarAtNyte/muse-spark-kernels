"""Measure short-window native WGMMA. Local baseline is Triton, not FA3."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import inspect
import re
fallback = inspect.getsource(base._v7_kernel_fn)
fallback = fallback.replace('def kernel_fn(data):', 'def _v7_kernel_fn(data):')
fallback = re.sub(r'    use_sliding = .*', "    use_sliding = bool(cfg['window'])", fallback)
exec(fallback, base.__dict__)
print('BASELINE native Triton fallback; not official FA3 for batches >4', flush=True)
for batch in (1, 16, 32):
    for seed in (313, 9587):
        data = R.generate_input(batch, 80000, 'tp8', 'sliding', seed,
            q_len=8, spread='agentic', dist='soft', page=1)
        gb, ob = capture(base, data)
        for splits in (1, 2, 4, 8, 16):
            candidate._WG_SLIDE_SPLITS = splits
            gc, oc = capture(candidate, data)
            error = ((ob.float()-oc.float()).norm()/ob.float().norm()).item()
            if not torch.isfinite(oc).all().item() or error > .003:
                raise AssertionError((batch, seed, splits, error))
            bt, ct, ratios = [], [], []
            for rep in range(16):
                if rep % 2: c,b = elapsed(gc),elapsed(gb)
                else: b,c = elapsed(gb),elapsed(gc)
                bt.append(b); ct.append(c); ratios.append(b/c)
            print('RESULT', json.dumps(dict(batch=batch,seed=seed,splits=splits,
                error=error,speedup_vs_triton=statistics.median(ratios),
                baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
            del gc,oc
        del gb,ob,data
for batch, page, scale, dist in ((3, 64, 1., 'edges'), (4, 1, 1/64, 'soft')):
    data = R.generate_input(batch, 6000, 'tp8', 'sliding', 371 + batch,
        q_len=8, spread='mixed', dist=dist, page=page, kv_scale=scale)
    output = candidate.kernel_fn(data)
    ok, message = R.check_implementation(data, R.collect(data, output))
    print('CHECK', batch, page, scale, ok, message, flush=True)
    if not ok: raise AssertionError(message)
