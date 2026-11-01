"""Why does the OKC_DEBUG=1 WGS build (bs=32) hit an illegal memory access? One state, sync after every step."""
import importlib.util, os, sys, traceback
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
_fix = lambda t: t.replace('#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif')
S.CUDA_SRC = _fix(S.CUDA_SRC)
for _k in list(getattr(S, 'ALT_SRCS', {})): S.ALT_SRCS[_k] = _fix(S.ALT_SRCS[_k])
w = R.generate_weights(R.WEIGHTS_SEED); st = S.prepare(w)
G = st['grid']; sfx = st.get('sfx', ()); dev = 'cuda'
print('GRID', G, 'scratch sizes', [(tuple(t.shape), str(t.dtype)) for t in st['scratch']], flush=True)
mode = os.environ.get('MODE', 'debug')
defs = tuple(S._main_defines(sfx))
extra = ('-DOKC_DEBUG=1',) if mode == 'debug' else ()
ext = S._ext(defs + sfx + extra)
d = R.generate_input(32, 'ep8', 419)
x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
out = torch.empty_like(x)
big = int(os.environ.get('DBGN', '5280'))
dbg = torch.zeros(big, dtype=torch.int64, device=dev)
for tag, sc in (('with_dbg_scratch', list(st['scratch'])[:3] + [dbg]), ('original_scratch', list(st['scratch']))):
    try:
        ext.okc_run(st['w'], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
        print('OK', tag, 'stamps>0:', int((dbg[:G * 8] > 0).sum()), flush=True)
    except Exception as e:
        print('FAIL', tag, repr(e)[:200], flush=True); traceback.print_exc(); break
print('DONE', flush=True)
