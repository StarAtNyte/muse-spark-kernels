"""Does the WGS scale multiply (bf16 HMUL2, FP16/tensor 'shared' pipe) cost tensor-core issue? Ablation: decoded
weights fed to wgmma without the scale HMUL2s (outputs wrong by design). Paired cold-L2 graph medians."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
OLD = '''          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);'''
assert S.CUDA_SRC.count(OLD) == 1
NEW = '''          af[ci & 1][jj][0] = r0 ^ (sgp & p.zero); af[ci & 1][jj][1] = r1;
          af[ci & 1][jj][2] = r2; af[ci & 1][jj][3] = r3;'''
S.ALT_SRCS['nohmul'] = S.CUDA_SRC.replace(OLD, NEW)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
sfx = st.get('sfx', ())
main = tuple(S._main_defines(sfx)) + sfx
exts = {'base': S._ext(main), 'nohmul': S._ext(main + ('@nohmul',))}
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(ext, d):
    x = d['hidden_states'].contiguous(); ids = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    out = torch.empty_like(x)
    fn = lambda: ext.okc_run(st['w'], st['scratch'], x, ids, tw, out, 0, 32, st['grid'])
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for bs in (32, 128, 256):
    for seed in (419, 7919):
        d = R.generate_input(bs, 'ep8', seed)
        graphs = {k: cap(e, d) for k, e in exts.items()}
        ts = {k: [] for k in graphs}
        for rep in range(30):
            for k in (('base', 'nohmul') if rep % 2 == 0 else ('nohmul', 'base')): ts[k].append(t1(graphs[k]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(bs=bs, seed=seed, base=round(med['base'], 1), nohmul=round(med['nohmul'], 1),
              gain=round(med['base'] / med['nohmul'], 4))), flush=True)
print('DONE', flush=True)
