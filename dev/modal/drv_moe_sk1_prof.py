"""sk1 cycle profile: per CTA wait vs compute cycles for gate/up and down stages, plus register/spill usage."""
import importlib.util, json, subprocess, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
ext = S._sk1_ext(4, 1)
print(subprocess.run(f"cuobjdump --dump-resource-usage {ext.__file__} | grep -i -E 'REG|STACK|LOCAL' | head", shell=True, capture_output=True, text=True).stdout, flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
G = st['grid']
prof = torch.zeros(G * 10, dtype=torch.int64, device='cuda')
for seed in (7919, 419):
    d = R.generate_input(1, 'ep8', seed)
    ids = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    out = torch.empty((1, 4096), dtype=torch.bfloat16, device='cuda')
    for rep in range(4):
        flush.add_(1); torch.cuda.synchronize()
        ext.sk1_run(st['sk1_w'] + [prof], d['hidden_states'], ids, tw, out, st['sk1_ctr'], st['sk1_ybuf'], 0, 32, G, 0)
        torch.cuda.synchronize()
    print('CHECK', R.check_implementation(d, out, w), flush=True)
    pr = prof.view(G, 10).double().cpu()
    ts = pr[:, 6:10]; t0 = ts[:, 0].min()
    rel = (ts - t0) / 1000.0
    print('STAMPS', json.dumps({n: [round(float(rel[:, i].min()), 2), round(float(rel[:, i].median()), 2), round(float(rel[:, i].max()), 2)]
          for i, n in enumerate(['start', 'loop', 'barrier_done', 'end'])}), flush=True)
    p = pr[:, :6] / 1980.0  # us at 1.98 GHz
    med = p.median(0).values.tolist(); mx = p.max(0).values.tolist()
    print('PROF', json.dumps(dict(seed=seed, us_median=dict(zip(['waitA', 'compA', 'waitB', 'compB', 'total', 'stages'], [round(v, 2) for v in med])),
          us_max_total=round(mx[4], 2), stages_med=float(prof.view(G, 10)[:, 5].median()))), flush=True)
print('DONE', flush=True)
