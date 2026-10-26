import os, sys
os.environ['WM_SMALL'] = '1'
src = open('test_wm.py').read().split("TESTS = (")[0]
src = src.replace("W13, S13 = wm_cuda.repack", "W13, S13 = None, None  # skip\n_ = None # wm_cuda.repack").replace("W2, S2 = wm_cuda.repack(weights[\"w2\"], weights[\"w2_sf\"])", "W2, S2 = None, None")
exec(compile(src, 'test_head', 'exec'))
import random
bad = 0
for trial in range(3000):
    M = random.choice([1, 2, 3, 4, 8, 9, 16, 17, 32, 33, 128, 256])
    nch = 1 if M <= 8 else 2 if M <= 16 else 4
    lmax = (M + 7) // 8 * 8
    grp = nch * 8
    kind = random.choice(['uniform', 'hot', 'one', 'none'])
    if kind == 'uniform':
        ids = torch.stack([torch.randperm(256, device=dev)[:8] for _ in range(M)]).to(torch.int32)
    elif kind == 'hot':
        ids = torch.stack([torch.cat([torch.tensor([random.randrange(32)], device=dev), torch.randperm(256, device=dev)[:7]]) for _ in range(M)])
        # dedupe within a token not required for routing counts; keep as is
        ids = ids.to(torch.int32)
    elif kind == 'one':
        e = random.randrange(32)
        ids = torch.stack([torch.cat([torch.tensor([e], device=dev), (torch.randperm(223, device=dev)[:7] + 33)]) for _ in range(M)]).to(torch.int32)
    else:
        ids = (torch.randint(32, 256, (M, 8), device=dev)).to(torch.int32)
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(512, dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(M * 8, dtype=torch.int32, device=dev)
    mod.wm_route(ids.contiguous(), NOLS, 0, lmax, cnt, lst, act, nact, lids, woff, grp)
    torch.cuda.synchronize()
    items = act.view(-1, 4)[:128].cpu().tolist()
    c = torch.bincount(ids.flatten().long()[ids.flatten() < 32], minlength=32).cpu().tolist() if (ids < 32).any() else [0] * 32
    exp = []
    for e in range(32):
        for j in range((c[e] + grp - 1) // grp):
            exp.append([e, j * grp, min(c[e] - j * grp, grp), 1])
    got = [it for it in items if it[2] != 0]
    ok = got == exp and all(it == [0, 0, 0, 0] for it in items[len(exp):])
    # list contents: each (e) list holds exactly its pair ids
    if ok:
        lst_h = lst.cpu().view(32, lmax)
        ids_h = ids.cpu().flatten()
        for e in range(32):
            want = sorted((ids_h == e).nonzero().flatten().tolist())
            have = sorted(lst_h[e, :c[e]].tolist())
            if want != have:
                ok = False
    if not ok:
        bad += 1
        print('BAD', M, kind, exp[:4], got[:4])
print('route trials bad:', bad)
