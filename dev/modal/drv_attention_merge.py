"""Paired graph comparison and correctness for the native attention merge."""
exec(open('/w/modal/drv_attention_profile.py').read().split("spec = importlib.util")[0])
import json
import statistics


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('/w/kernels/muse_attention_decode/submission.py', 'base')
candidate = load('/w/run/sub.py', 'candidate')
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')


def capture(module, data):
    module.kernel_fn(data)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module.kernel_fn(data)
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def elapsed(graph):
    flush.add_(1)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


for batch in (3, 16, 32, 33):
    for seed in (539, 1427):
        data = R.generate_input(batch, 80000, 'tp8', 'full', seed,
                                spread='agentic', dist='coherent', page=1)
        gb, ob = capture(base, data)
        gc, oc = capture(candidate, data)
        error = ((ob.float() - oc.float()).norm() / ob.float().norm()).item()
        if not torch.isfinite(oc).all().item() or error > 0.001:
            raise AssertionError((batch, seed, error))
        ratios, bt, ct = [], [], []
        for rep in range(24):
            if rep % 2:
                c, b = elapsed(gc), elapsed(gb)
            else:
                b, c = elapsed(gb), elapsed(gc)
            ratios.append(b / c)
            bt.append(b)
            ct.append(c)
        print('RESULT', json.dumps(dict(batch=batch, seed=seed, relative_error=error,
              paired_speedup=statistics.median(ratios), baseline_us=statistics.median(bt),
              candidate_us=statistics.median(ct))), flush=True)
        del data, gb, gc, ob, oc

for batch, ctx, page, scale, dist in ((4, 6000, 64, 1., 'edges'),
                                    (2, 9000, 64, 1/64, 'soft'),
                                    (5, 80000, 1, 1., 'coherent')):
    data = R.generate_input(batch, ctx, 'tp8', 'full', 391 + batch,
                            page=page, kv_scale=scale, dist=dist)
    output = candidate.kernel_fn(data)
    ok, message = R.check_implementation(data, R.collect(data, output))
    print('CHECK', batch, ctx, page, scale, ok, message, flush=True)
    if not ok:
        raise AssertionError(message)
