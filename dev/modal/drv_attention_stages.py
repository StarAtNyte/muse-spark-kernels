"""Test reduced shared-memory residency and split counts against official best."""
exec(open('/w/modal/drv_attention_merge.py').read().split('for batch in (3, 16, 32, 33):')[0])

for batch in (16, 32):
    for seed in (539, 1427):
        data = R.generate_input(batch, 80000, 'tp8', 'full', seed,
                                spread='agentic', dist='coherent', page=1)
        gb, ob = capture(base, data)
        for stages, splits in ((1, 32), (1, 64), (1, 128), (2, 32), (2, 64), (3, 64)):
            candidate._FD_POLICY[('full', False)] = [(2, 64, stages, splits, 3)]
            gc, oc = capture(candidate, data)
            error = ((ob.float() - oc.float()).norm() / ob.float().norm()).item()
            if not torch.isfinite(oc).all().item() or error > 0.001:
                raise AssertionError((batch, seed, stages, splits, error))
            ratios, bt, ct = [], [], []
            for rep in range(16):
                if rep % 2:
                    c, b = elapsed(gc), elapsed(gb)
                else:
                    b, c = elapsed(gb), elapsed(gc)
                ratios.append(b / c)
                bt.append(b)
                ct.append(c)
            print('RESULT', json.dumps(dict(batch=batch, seed=seed, stages=stages, splits=splits,
                  relative_error=error, paired_speedup=statistics.median(ratios),
                  baseline_us=statistics.median(bt), candidate_us=statistics.median(ct))), flush=True)
            del gc, oc
        del data, gb, ob
