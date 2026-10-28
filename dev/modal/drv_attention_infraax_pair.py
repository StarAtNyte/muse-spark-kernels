"""Paired CUDA-graph screen of infraax's full-verify attention vs our public kernel."""
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
flush = torch.empty(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)


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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


for batch in (16, 32):
    for seed in (539, 1427, 7919):
        data = R.generate_input(batch, 80000, 'tp8', 'full', seed,
                                q_len=8, spread='agentic', dist='coherent', page=1)
        gb, ob = capture(base, data)
        gc, oc = capture(candidate, data)
        ok, message = R.check_implementation(data, R.collect(data, oc))
        error = ((ob.float() - oc.float()).norm() / ob.float().norm()).item()
        if not ok or not torch.isfinite(oc).all().item():
            raise AssertionError((batch, seed, ok, message, error))
        ratios, bt, ct = [], [], []
        for rep in range(20):
            if rep % 2:
                c, b = elapsed(gc), elapsed(gb)
            else:
                b, c = elapsed(gb), elapsed(gc)
            ratios.append(b / c)
            bt.append(b)
            ct.append(c)
        print('RESULT', json.dumps(dict(batch=batch, seed=seed, official_check=ok,
              relative_error=error, paired_speedup=statistics.median(ratios),
              baseline_us=statistics.median(bt), candidate_us=statistics.median(ct))), flush=True)
        del data, gb, gc, ob, oc
