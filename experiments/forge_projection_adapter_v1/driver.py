"""Verify the Forge O32 wrapper against the complete promoted projection stack."""
import importlib.util, sys, json, torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
sp=importlib.util.spec_from_file_location('submission','/w/run/sub.py')
S=importlib.util.module_from_spec(sp);sys.modules['submission']=S;sp.loader.exec_module(S)
exec(compile('"""The canonical O projection\'s M=32 b2o route, with untimed immutable-weight preparation."""\nimport weakref\nimport submission as _mod\n\n_cache = {}\n\ndef run(x, weight):\n    key = id(weight)\n    version = weight._version\n    entry = _cache.get(key)\n    if entry is None or entry[0]() is not weight or entry[1] != version:\n        state = _mod.b2o_prepare_olayer_candidates({\'o\': weight}, [(32, 128, 8, 1, 1, 1)])\n        entry = (weakref.ref(weight), version, state)\n        _cache[key] = entry\n    return _mod.b2o_run_olayer_candidate(x, entry[2], 0)\n', 'forge_projection_wrapper', 'exec'))
weights=R.generate_weights(R.WEIGHTS_SEED)
state=S.prepare(weights)
print('PREPARED',flush=True)
for seed in (419, 7919):
    d=R.generate_input(32,'o',seed)
    a=S.kernel_fn(d,state); b=run(d['x'],weights['o']);torch.cuda.synchronize()
    for name,o in (('full',a),('wrapper',b)):
        ok,msg=R.check_implementation(d,o,weights)
        if not ok: raise AssertionError((name,msg))
    err=((a.float()-b.float()).norm()/a.float().norm()).item()
    print('RESULT',json.dumps(dict(seed=seed,relative_l2=err,identical=bool(torch.equal(a,b)))),flush=True)
    if err>0.001:raise AssertionError(err)
print('DONE',flush=True)
