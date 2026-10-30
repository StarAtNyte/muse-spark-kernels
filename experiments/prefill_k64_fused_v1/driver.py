"""Compare the persistent MoE path against its unpermuted parent on H200."""
import importlib.util, json, sys
import torch
sys.path.insert(0, '/w/kernels/muse_prefill_layer')
import reference as R

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

B = load('baseline', '/w/kernels/muse_prefill_layer/submission.py')
C = load('candidate', '/w/run/sub.py')
torch.manual_seed(921)
w = R.generate_weights(R.WEIGHTS_SEED)
print('PREPARING BASELINE', flush=True)
sb = B.prepare(w)
print('PREPARING CANDIDATE', flush=True)
sc = C.prepare(w)
for T in (32, 96, 753):
    p = torch.randn((T, 4096), device='cuda', dtype=torch.bfloat16)
    lid = torch.full((T, 8), -1, device='cuda', dtype=torch.int32)
    pos = torch.zeros_like(lid)
    lid[:, 0] = 0
    pos[:, 0] = torch.arange(T, device='cuda', dtype=torch.int32)
    cnt = torch.zeros(32, device='cuda', dtype=torch.int32)
    cnt[0] = T
    ob = B._moe_forward(sb, p, lid, pos, cnt, T)
    oc = C._moe_forward(sc, p, lid, pos, cnt, T)
    torch.cuda.synchronize()
    a = ob['y'][ob['rrow'][:, 0].long()].float()
    b = oc['y'][oc['rrow'][:, 0].long()].float()
    err = ((a - b).norm() / a.norm()).item()
    print('RESULT', json.dumps({'tokens': T, 'relative_l2': err, 'finite': bool(torch.isfinite(b).all())}), flush=True)
    if not torch.isfinite(b).all() or err > 0.01:
        raise AssertionError((T, err))
print('PASS: persistent MoE screening', flush=True)
