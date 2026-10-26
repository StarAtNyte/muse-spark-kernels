import os, sys
os.environ.setdefault('WM_SMALL', '1')
import importlib.util
src = open('test_wm.py').read().split("TESTS = (")[0]
exec(compile(src, 'test_wm_head', 'exec'))
bad = 0
for bs in tuple(int(x) for x in os.environ.get('BSS', '32').split(',')):
    nbad = 0
    for seed in range(int(os.environ.get('N', '60'))):
        dist = 'skewed' if seed % 3 == 0 else 'normal'
        data = R.generate_input(bs, "ep8", seed=1000 + seed, dist=dist, device=dev)
        out, inter, y = run(data)
        ok, msg = R.check_implementation(data, out, weights)
        if not ok:
            nbad += 1
            print('BAD', bs, seed, dist, msg)
    print('bs', bs, 'bad', nbad, flush=True)
