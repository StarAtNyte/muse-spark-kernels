"""Verify the Forge O32 wrapper against the complete promoted projection stack."""
import importlib.util, sys, json, torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
sp=importlib.util.spec_from_file_location('submission','/w/run/sub.py')
S=importlib.util.module_from_spec(sp);sys.modules['submission']=S;sp.loader.exec_module(S)
exec(compile('"""The canonical O projection\'s M=32 b2o route, with untimed immutable-weight preparation."""\nimport weakref\nimport submission as _mod\n\n_cache = {}\n\ndef run(x, weight):\n    key = id(weight)\n    version = weight._version\n    entry = _cache.get(key)\n    if entry is None or entry[0]() is not weight or entry[1] != version:\n        state = _mod.b2o_prepare_olayer_candidates({\'o\': weight}, [(32, 128, 8, 1, 1, 1)])\n        entry = (weakref.ref(weight), version, state)\n        _cache[key] = entry\n    return _mod.b2o_run_olayer_candidate(x, entry[2], 0)\n', 'forge_projection_wrapper', 'exec'))
torch.manual_seed(991)
w = torch.randn((8192,1024),device='cuda',dtype=torch.bfloat16).contiguous()
x = torch.zeros((32,1024),device='cuda',dtype=torch.bfloat16);x[:,0]=1
before=run(x,w);torch.cuda.synchronize()
assert torch.equal(before[:,0],w[0,0].expand(32))
w[0,0].add_(1)
after=run(x,w);torch.cuda.synchronize()
assert torch.equal(after[:,0],w[0,0].expand(32))
assert not torch.equal(before[:,0],after[:,0])
w2=w.clone();w2[0,0].add_(1)
third=run(x,w2);torch.cuda.synchronize()
assert torch.equal(third[:,0],w2[0,0].expand(32))
print('PASS: in-place version change and replacement weight invalidate preparation',flush=True)
