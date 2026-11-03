"""Per-CTA timeline of the TINY (bs<=4) and bs8 (t2_wsi) builds of a submission, OKC_DEBUG globaltimer stamps, cold L2.
Driver-side patches only (the submission file is unchanged): OKC_DEBUG made overridable, SM id stamped, A-unit landing
stamps for the first 16 units. Prints per-CTA rows as JSON for offline analysis."""
import importlib.util, json, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

def dbg_patch(s):
    s = s.replace('#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif', 1)
    s = s.replace('  DBG(0);\n', '  DBG(0);\n  if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 0] = sm_; }\n', 1)
    s = s.replace('if (OKC_DEBUG && !PB && i < 4 && tid == 0 && p.dbg) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 24 + i] = gtimer();',
                  'if (OKC_DEBUG && !PB && i < 16 && tid == 0 && p.dbg) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 1 + i] = gtimer();')
    E = 'p.dbg[gridDim.x * 8 + blockIdx.x * 32 + %d]'
    a = "    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = sc_load(slot, warp, lane);\n"
    if s.count(a) == 1:
        s = s.replace(a, "    if (OKC_DEBUG && PB && i < 4 && tid == 0 && p.dbg) " + (E % 24).replace('24]', '24 + i]') + " = gtimer();\n" + a)
    a = "    // WSB: the compute warps keep the base loop's CTA barrier"
    if s.count(a) == 1:
        s = s.replace(a, "    if (OKC_DEBUG && PB && i < 4 && tid == 0 && p.dbg) " + (E % 28).replace('28]', '28 + i]') + " = gtimer();\n" + a)
    a = "      if (!OKC_DF && q == nA - 1) {"
    if s.count(a) == 1:
        s = s.replace(a, "      if (OKC_DEBUG && lane == 0 && p.dbg && q == nA - 1) " + E % 17 + " = gtimer();\n" + a)
    a = "  __threadfence();\n  WS_JOIN();  // every epilogue store"
    if s.count(a) == 1:
        s = s.replace(a, "  if (OKC_DEBUG && (threadIdx.x & 31) == 0 && p.dbg) " + E % 18 + " = gtimer();\n  __threadfence();\n  if (OKC_DEBUG && (threadIdx.x & 31) == 0 && p.dbg) " + E % 19 + " = gtimer();\n  WS_JOIN();  // every epilogue store")
    return s
for k in list(S.ALT_SRCS):
    S.ALT_SRCS[k] = dbg_patch(S.ALT_SRCS[k])
S.CUDA_SRC = dbg_patch(S.CUDA_SRC)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
G = st['grid']
flush = torch.empty(512 * 1024 * 1024 // 4, device='cuda')
NAMES = {0: 'start', 1: 'routed', 7: 'Bwait1', 2: 'A_done', 3: 'B_done', 4: 'barrier', 5: 'final'}
def defs_for(bs):
    if bs in st.get('ps', {}): return tuple(st['ps'][bs])
    if bs <= S.TINY_MAX_M: return tuple(S.TINY_DEFINES) + st['sfxs']
    return tuple(S.TINY2_DEFINES) + st['sfxs']
for bs, seed in ((1, 7919), (1, 419), (8, 5041622730710591082)):
    ext = S._ext(defs_for(bs) + ('-DOKC_DEBUG=1',))
    d = R.generate_input(bs, 'ep8', seed)
    x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    ids = idt.long(); distinct = int(ids[(ids >= 0) & (ids < 32)].unique().numel())
    out = torch.empty((bs, 4096), dtype=torch.bfloat16, device='cuda')
    dbg = torch.zeros(G * 8 + G * 32, dtype=torch.int64, device='cuda')
    sc = list(st['scratch'])[:3] + [dbg]
    for rep in range(8):
        flush.sum(); dbg.zero_(); torch.cuda.synchronize()
        ext.okc_run(st['w'], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
        if rep >= 5:
            tt = dbg[:G * 8].view(G, 8).cpu().double(); ex = dbg[G * 8:].view(G, 32).cpu()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            res = {}
            for k, n in NAMES.items():
                col = tt[:, k][tt[:, k] > 0]
                if col.numel():
                    c = ((col - t00) / 1000).sort().values
                    res[n] = [round(float(c[0]), 2), round(float(c[c.numel() // 2]), 2), round(float(c[-1]), 2)]
            print('TL', json.dumps(dict(bs=bs, seed=seed, rep=rep, distinct=distinct, **res)), flush=True)
    rel = ((tt - t00) / 1000).clamp_min(-1).tolist()
    land = ex[:, 1:17].double()
    land = torch.where(land > 0, (land - t00) / 1000, torch.full_like(land, -1)).tolist()
    print('CTA', json.dumps(dict(bs=bs, seed=seed, smid=ex[:, 0].tolist(), t=[[round(v, 2) for v in r] for r in rel],
                                land=[[round(v, 2) for v in r] for r in land],
                                bwait=ex[:, 20:23].tolist(), ext=[[round(float(v - t00) / 1000, 2) if v > 0 else -1 for v in r] for r in ex[:, 17:32].tolist()])), flush=True)
print('PATCHCOUNT', {k: S.ALT_SRCS[k].count('p.dbg[gridDim.x * 8 + blockIdx.x * 32 +') for k in ('var', 't2_wsi')}, flush=True)
print('DONE', flush=True)
