"""WGS (MAIN build) per-CTA timeline with stall decomposition, judge seeds, cold L2 (flush.sum()).
Driver-side patches only (the submission is unchanged). Per CTA (ext slots, clock64 cycles unless noted):
 8/9 W-producer empty wait A/B, 10/11 st1 throttle A/B, 12/13 x-producer empty wait A/B,
 14/15 consumer weights-full wait A/B, 16/17 consumer x-full wait A/B, 18 xready spin, 19 doneAs spin, 20 doneAs spins,
 21 first B x issue (gtimer), 22 first B wait start (gtimer), 23 max single doneAs spin, 24 W first B issue (gtimer),
 25 W last issue (gtimer), 1/2 stages A/B, 3/4 first/last B job."""
import importlib.util, json, sys, os
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
E = 'okc_dbgx[blockIdx.x * 32 + %d]'
def rep1(s, a, b):
    assert s.count(a) == 1, a[:70]
    return s.replace(a, b)
def dbg_patch(s):
    s = rep1(s, '#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif')
    s = rep1(s, '#define DBG(k) do {', '__device__ unsigned long long* okc_dbgx;\n#define DBG(k) do {')
    s = rep1(s, '  DBG(0);\n', '  DBG(0);\n  if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) { okc_dbgx = p.dbg + gridDim.x * 8; unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); ' + E % 0 + ' = sm_; }\n')
    s = rep1(s, '__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {\n  (void)tag;\n  mbar_wait(bar, parity);\n}',
             '__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {\n#if OKC_DEBUG\n  const long long c0 = clock64();\n  mbar_wait(bar, parity);\n  if (okc_dbgx && (tag < 6 || threadIdx.x == 0)) ' + E % 8 + ' += 0; if (okc_dbgx && (tag < 6 || threadIdx.x == 0)) okc_dbgx[blockIdx.x * 32 + 8 + tag] += clock64() - c0;\n#else\n  (void)tag;\n  mbar_wait(bar, parity);\n#endif\n}')
    s = rep1(s, '1000 + gs', '0 + pb'); s = rep1(s, '5000 + gs', '2 + pb'); s = rep1(s, '2000000 + gs', '4 + pb')
    s = rep1(s, '3000000 + seq', '6 + (PB ? 1 : 0)'); s = rep1(s, '4000000 + seq', '8 + (PB ? 1 : 0)')
    s = rep1(s, 'if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }',
             'if (!pb && !(OKC_WGSX & 8)) { const long long c0 = clock64(); while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); if (OKC_DEBUG && okc_dbgx) ' + E % 18 + ' += clock64() - c0; }')
    s = rep1(s, 'if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }',
             'if (pb && !(OKC_WGSX & 8)) { if (OKC_DEBUG && okc_dbgx && ' + E % 22 + ' == 0) ' + E % 22 + ' = gtimer(); const long long c0 = clock64(); bool sp_ = false; while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) { sp_ = true; __nanosleep(32); } if (OKC_DEBUG && okc_dbgx) { const long long dc = clock64() - c0; ' + E % 19 + ' += dc; ' + E % 20 + ' += sp_; if (dc > (long long)' + E % 23 + ') ' + E % 23 + ' = dc; if (' + E % 21 + ' == 0) ' + E % 21 + ' = gtimer(); } }')
    s = rep1(s, '      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }',
             '      if (OKC_DEBUG && okc_dbgx) { if (pb && ' + E % 24 + ' == 0) ' + E % 24 + ' = gtimer(); ' + E % 25 + ' = gtimer(); }\n      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }')
    s = rep1(s, '  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);\n', '  if ((threadIdx.x & 31) == 0) { wgs_ranges(p, S); if (OKC_DEBUG && okc_dbgx) { ' + E % 1 + ' = S.wgr[1] - S.wgr[0]; ' + E % 2 + ' = S.wgr[3] - S.wgr[2]; ' + E % 3 + ' = S.wgr[2] >> 8; ' + E % 4 + ' = (S.wgr[3] - 1) >> 8; } }\n')
    return s
S.CUDA_SRC = dbg_patch(S.CUDA_SRC)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
G = st['grid']
flush = torch.empty(512 * 1024 * 1024 // 4, device='cuda')
NAMES = {0: 'start', 1: 'routed', 7: 'gathered', 2: 'A_done', 3: 'B_done', 4: 'barrier', 5: 'final'}
ext = S._ext(tuple(S._main_defines(st['sfx'])) + tuple(st['sfxm']) + ('-DOKC_DEBUG=1',))
CASES = [(16, 5041358389864036635), (32, 5041490560287313858), (128, 5041754901133868307), (256, 5041887071557145533)]
CK = 1.98e3  # cycles per us
for bs, seed in CASES:
    d = R.generate_input(bs, 'ep8', seed)
    x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    ids = idt.long(); distinct = int(ids[(ids >= 0) & (ids < 32)].unique().numel())
    out = torch.empty((bs, 4096), dtype=torch.bfloat16, device='cuda')
    dbg = torch.zeros(G * 8 + G * 32, dtype=torch.int64, device='cuda')
    sc = list(st['scratch'])[:3] + [dbg]
    agg = []
    for rep in range(10):
        flush.sum(); dbg.zero_(); torch.cuda.synchronize()
        ext.okc_run(st['w'], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
        if rep < 4: continue
        tt = dbg[:G * 8].view(G, 8).cpu().double(); ex = dbg[G * 8:].view(G, 32).cpu().double()
        t00 = tt[:, 0][tt[:, 0] > 0].min()
        res = {}
        for k, n in NAMES.items():
            col = tt[:, k][tt[:, k] > 0]
            if col.numel():
                c = ((col - t00) / 1000).sort().values
                res[n] = [round(float(c[0]), 2), round(float(c[c.numel() // 2]), 2), round(float(c[-1]), 2)]
        cyc = {nm: round(float(ex[:, k].mean() / CK), 2) for k, nm in ((8, 'WE_A'), (9, 'WE_B'), (10, 'ST1_A'), (11, 'ST1_B'), (12, 'XE_A'), (13, 'XE_B'),
                (14, 'CWF_A'), (15, 'CWF_B'), (16, 'CXF_A'), (17, 'CXF_B'), (18, 'xrdy'), (19, 'doneAs'), (23, 'maxDA'))}
        cyc['nspins'] = round(float(ex[:, 20].mean()), 2)
        A = (tt[:, 2] - t00) / 1000; B = (tt[:, 3] - t00) / 1000
        bfirst = (ex[:, 21] - t00) / 1000; bws = (ex[:, 22] - t00) / 1000; wfb = (ex[:, 24] - t00) / 1000
        nA = ex[:, 1]; nB = ex[:, 2]
        res.update(dict(stall_mean_us=cyc, Bwaitstart_minus_Adone=round(float((bws - A).median()), 2), firstBx_minus_Adone=round(float((bfirst - A).median()), 2),
                        Wfirst_B_minus_Adone=round(float((wfb - A).median()), 2),
                        stagesA=[int(nA.min()), int(nA.max())], stagesB=[int(nB.min()), int(nB.max())]))
        tot = float(nA.sum() + nB.sum()) * 35840
        res['GBs_A'] = round(float(nA.sum()) * 35840 / float(A.max() - res['gathered'][0]) / 1e3, 2)
        res['GBs_B'] = round(float(nB.sum()) * 35840 / float(B.max() - A.median()) / 1e3, 2)
        print('TL', json.dumps(dict(bs=bs, seed=seed, rep=rep, distinct=distinct, **res)), flush=True)
    rows = []
    for c in range(G):
        rows.append(dict(c=c, sm=int(ex[c, 0]), nA=int(ex[c, 1]), nB=int(ex[c, 2]), jB=[int(ex[c, 3]), int(ex[c, 4])], A=round(float(A[c]), 2), B=round(float(B[c]), 2),
                         bws=round(float(bws[c]), 2), bfx=round(float(bfirst[c]), 2), cxfB=round(float(ex[c, 17] / CK), 2), cwfB=round(float(ex[c, 15] / CK), 2),
                         cxfA=round(float(ex[c, 16] / CK), 2), cwfA=round(float(ex[c, 14] / CK), 2), da=round(float(ex[c, 19] / CK), 2), weB=round(float(ex[c, 9] / CK), 2)))
    print('CTA', bs, json.dumps(rows), flush=True)
print('DONE', flush=True)
