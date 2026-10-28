"""Compare row ownership and independent gate/down tiles in the native MoE."""
exec(open('/w/modal/drv_moe_ptx.py').read().split('for name, cuda, flags in variants:')[0])
original = sub._WM_CFG
configs = [(4, 4, 1), (2, 4, 1), (2, 8, 1), (4, 4, 4), (2, 4, 4),
           (4, 4, 5), (4, 4, 6), (2, 4, 4, 4), (4, 4, 4, 2),
           (4, 4, 1, 2), (2, 4, 1, 4)]
for batch in (16, 32, 128):
    for seed in (419, 2387):
        data = R.generate_input(batch, 'ep8', seed + batch)
        sub._WM_CFG = original
        base_graph, base = capture(data, base_mod)
        for cfg in configs:
            sub._WM_CFG = [(10**9, cfg)]
            try:
                graph, result = capture(data, base_mod)
                error = ((result.float() - base.float()).norm() /
                         base.float().norm().clamp_min(1e-20)).item()
                if not torch.isfinite(result).all().item() or error > 0.001:
                    raise AssertionError((batch, seed, cfg, error))
                ratios, bt, ct = [], [], []
                for rep in range(16):
                    if rep % 2:
                        c, b = elapsed(graph), elapsed(base_graph)
                    else:
                        b, c = elapsed(base_graph), elapsed(graph)
                    ratios.append(b/c)
                    bt.append(b)
                    ct.append(c)
                print('RESULT', json.dumps(dict(batch=batch, seed=seed, cfg=cfg,
                      relative_error=error, speedup=statistics.median(ratios),
                      baseline_us=statistics.median(bt), candidate_us=statistics.median(ct))), flush=True)
            except (RuntimeError, AssertionError) as exc:
                print('FAIL', batch, seed, cfg, str(exc)[:200], flush=True)
