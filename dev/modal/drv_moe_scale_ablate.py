"""Upper bound for compressing the MoE block scales: the WGS producer copies fewer bytes per 36 KB stage (dropping
1 KB = 6-bit scales, 2 KB = 4-bit, 4 KB = no scale bytes). Compute is unchanged; outputs are wrong by design.
Paired cold-L2 graph medians on reference inputs for the WGS sizes."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
OLD = '''      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }'''
assert S.CUDA_SRC.count(OLD) == 1, S.CUDA_SRC.count(OLD)
NEW = OLD.replace('"r"(UNIT_BYTES) : "memory");', '"r"(UNIT_BYTES - OKC_ABL) : "memory");').replace(
    '"l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar))', '"l"(src), "r"(UNIT_BYTES - OKC_ABL), "r"(smem_u32(bar))')
S.ALT_SRCS['abl'] = S.CUDA_SRC.replace(OLD, NEW).replace('#define OKC_DEBUG 0', '#define OKC_DEBUG 0\n#ifndef OKC_ABL\n#define OKC_ABL 0\n#endif', 1)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
sfx = st.get('sfx', ())
main = tuple(S._main_defines(sfx)) + sfx
V = {'base': None, 'drop1k': 1024, 'drop2k': 2048, 'drop4k': 4096}
exts = {k: (S._ext(main) if v is None else S._ext(main + ('@abl', f'-DOKC_ABL={v}'))) for k, v in V.items()}
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
        names = list(graphs); ts = {k: [] for k in names}
        for rep in range(30):
            for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(bs=bs, seed=seed, **{k: round(v, 1) for k, v in med.items()},
              **{f'gain_{k}': round(med['base'] / v, 4) for k, v in med.items() if k != 'base'})), flush=True)
print('DONE', flush=True)
