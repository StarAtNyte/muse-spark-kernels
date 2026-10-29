"""WGS consumer wait breakdown: cycles each consumer warpgroup spends waiting for weight stages (BW_F) vs x stages
(BX_F, i.e. gathered activations / phase-A results), per phase, vs its total phase time. Alt source, debug only."""
import importlib.util, json, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
src = S.CUDA_SRC.replace('#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif')
W1 = '    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);'
W2 = '    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);'
ST = '  auto stage = [&](auto first) {'
END = '''  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);'''
for s_ in (W1, W2, ST, END): assert src.count(s_) == 1, s_
src = src.replace(W1, '    { const long long _t = clock64();\n' + W1 + '\n      t_w += clock64() - _t; }')
src = src.replace(W2, '    { const long long _t = clock64();\n' + W2 + '\n      t_x += clock64() - _t; }')
src = src.replace(ST, '  long long t_w = 0, t_x = 0; const long long t_0 = clock64();\n' + ST)
src = src.replace(END, END + '''
  if ((threadIdx.x & 127) == 0 && p.dbg) {
    unsigned long long* q = p.dbg + gridDim.x * 8 + (blockIdx.x * 16 + (PB ? 8 : 0) + wg * 2);
    q[0] = (unsigned long long)t_w; q[1] = (unsigned long long)t_x;
    p.dbg[gridDim.x * 8 + gridDim.x * 16 + blockIdx.x * 8 + (PB ? 4 : 0) + wg] = (unsigned long long)(clock64() - t_0);
  }''')
S.ALT_SRCS['waits'] = src
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
sfx = st.get('sfx', ())
ext = S._ext(tuple(S._main_defines(sfx)) + sfx + ('@waits',))
G = st['grid']
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
for bs in (32, 128, 256):
    for seed in (419, 7919):
        d = R.generate_input(bs, 'ep8', seed)
        x = d['hidden_states'].contiguous(); ids = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
        out = torch.empty_like(x)
        dbg = torch.zeros(G * 8 + G * 16 + G * 8, dtype=torch.int64, device='cuda')
        sc = list(st['scratch'])[:3] + [dbg]
        for _ in range(4):
            flush.add_(1); dbg.zero_(); torch.cuda.synchronize()
            ext.okc_run(st['w'], sc, x, ids, tw, out, 0, 32, G); torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, out, w)
        wt = dbg[G * 8:G * 8 + G * 16].view(G, 2, 4, 2).double().cpu() / 1980.0   # us: [cta][phase][wg][w|x]
        tot = dbg[G * 24:G * 32].view(G, 2, 4).double().cpu() / 1980.0
        res = {}
        for ph, name in ((0, 'A'), (1, 'B')):
            res[name] = dict(total=round(float(tot[:, ph].mean()), 1), wait_w=round(float(wt[:, ph, :, 0].mean()), 1),
                             wait_x=round(float(wt[:, ph, :, 1].mean()), 1))
        print('ROW', json.dumps(dict(bs=bs, seed=seed, ok=ok, **res)), flush=True)
print('DONE', flush=True)
