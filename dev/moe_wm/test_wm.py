"""Local correctness test of the weight-major kernels on a small GPU (32 experts only)."""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../kernels/muse_moe_w4a16"))
sys.path.insert(0, HERE)

import reference as R  # noqa: E402
from nvfp4 import random_weights  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402
import wm_cuda  # noqa: E402

dev = "cuda"
NOLS = torch.empty(0, dtype=torch.int32, device='cuda')
gen = torch.Generator(device=dev).manual_seed(5252)
w13, w13_sf, gs13 = random_weights(8192, 4096, generator=gen, device=dev, leading=(32,))
w2, w2_sf, gs2 = random_weights(4096, 4096, generator=gen, device=dev, leading=(32,))
weights = {
    "w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(),
    "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
    "post_norm_w": (0.1 * torch.randn((4096,), generator=gen, device=dev)).to(torch.bfloat16),
}

import ctypes, subprocess
t0 = time.time()
H = os.path.abspath(os.path.join(HERE, "../../.cuda_home/home"))
ARCH = os.environ.get("WM_ARCH", "sm_86")
open(os.path.join(HERE, "wm_dev.cu"), "w").write(wm_cuda.CUDA_SRC)
subprocess.check_call(["clang++", "-x", "cuda", f"--cuda-path={H}", f"--cuda-gpu-arch={ARCH}", "-O3", "-std=c++17", "-fPIC",
                       "-shared", "-Wno-unknown-cuda-version", "-Xcuda-ptxas", "-v", "-o", os.path.join(HERE, "wm_dev.so"), os.path.join(HERE, "wm_dev.cu"),
                       f"-L{H}/lib64", "-lcudart", f"-Wl,-rpath,{H}/lib64"])
lib = ctypes.CDLL(os.path.join(HERE, "wm_dev.so"))
lib.wm_set_pdl(0)
lib.wm_set_persist(int(os.environ.get('WM_PERSIST','0')))
print("compiled", time.time() - t0)


def P(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class Mod:
    @staticmethod
    def wm_route(ids, ls, ls_val, lmax, cnt, lst, act, nact, lids, woff, grp):
        lib.wm_route_c(P(ids), ctypes.c_int(ids.numel()), P(ls) if ls.numel() else ctypes.c_void_p(0), ctypes.c_int(ls_val), ctypes.c_int(lmax), P(cnt), P(lst), P(act), P(nact), P(lids), P(woff), ctypes.c_int(grp), ctypes.c_int(act.numel() // 4), stream())

    @staticmethod
    def wm_finish(y, tw, lids, nw, out, eps):
        lib.wm_finish_c(P(y), P(tw), P(lids), P(nw), P(out), ctypes.c_int(out.shape[0]), ctypes.c_float(eps), stream())

    @staticmethod
    def wm_gemm_run(mode, nch, nt, nw, ring, W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax):
        r = lib.wm_gemm_c(ctypes.c_int(mode), ctypes.c_int(nch), ctypes.c_int(nt), ctypes.c_int(nw), ctypes.c_int(ring), ctypes.c_int(0), P(W), P(S), P(GS), P(X), P(OUT), P(cnt), P(lst), P(act), P(nact), P(woff), ctypes.c_int(maxslots), ctypes.c_int(lmax), stream())
        assert r == 0


mod = Mod()

t0 = time.time()
W13, S13 = wm_cuda.repack(weights["w13"], weights["w13_sf"])
W2, S2 = wm_cuda.repack(weights["w2"], weights["w2_sf"])
torch.cuda.synchronize()
print("repacked", time.time() - t0)


import os as _os


def CFG(M):
    v = _os.environ.get('WM_CFG')
    if v:
        a, b, c = map(int, v.split(','))
        return a, b, c
    return (2, 4, 0) if M <= 16 else (int(_os.environ.get("WM_BIGNT", "4")), 4, int(_os.environ.get("WM_BIGRING", "1")))


def run(data):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    nch = 1 if M <= 8 else 2 if M <= 16 else (8 if (_os.environ.get('WM_NCH8') and M >= 512) else 4)
    lmax = (M + 7) // 8 * 8
    NT, NW, RING = CFG(M)
    ids = data["topk_ids"].contiguous()
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(4 * (32 + (M * 8) // (nch * 8) + 8), dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(M * 8, dtype=torch.int32, device=dev)
    mod.wm_route(ids, NOLS, 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
    maxslots = min(M * 8, 32 + (M * 8) // (nch * 8))
    inter = torch.zeros((M * 8, 4096), dtype=torch.bfloat16, device=dev)
    y = torch.zeros((M * 8, 4096), dtype=torch.bfloat16, device=dev)
    mod.wm_gemm_run(0, nch, NT, NW, RING, W13, S13, weights["w13_gs"], x, inter, cnt, lst, act, nact, woff, maxslots, lmax)
    mod.wm_gemm_run(1, nch, NT, NW, RING, W2, S2, weights["w2_gs"], inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
    if os.environ.get('WM_FIN'):
        outf = torch.empty((M, 4096), dtype=torch.bfloat16, device=dev)
        nw1 = (1.0 + weights['post_norm_w'].float()).to(torch.bfloat16)
        mod.wm_finish(y, data['topk_weights'].contiguous(), lids, nw1, outf, R.POST_NORM_EPS)
        return outf, inter, y
    # torch finish
    yf = y.view(M, 8, 4096).float() * (1.0 + weights["post_norm_w"].float())
    yf = yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + R.POST_NORM_EPS)
    live = (lids.view(M, 8) >= 0).float().unsqueeze(-1)
    out = (yf * data["topk_weights"].unsqueeze(-1) * live).sum(1)
    return out.to(torch.bfloat16), inter, y


TESTS = ((1000, 'normal'), (2048, 'skewed'), (1, 'normal'), (8, 'normal'), (16, 'normal'), (32, 'normal'), (4, 'normal'), (32, 'skewed'), (33, 'normal'), (128, 'skewed'), (257, 'normal'))
if _os.environ.get('WM_SMALL'):
    TESTS = ((1, 'normal'), (8, 'normal'), (16, 'skewed'))
for bs, dist in TESTS:
    data = R.generate_input(bs, "ep8", seed=7 + bs, dist=dist, device=dev)
    out, inter, y = run(data)
    ok, msg = R.check_implementation(data, out, weights)
    print(bs, dist, ok, msg)


def bench(bs, dist="normal", iters=20):
    data = R.generate_input(bs, "ep8", seed=11 + bs, dist=dist, device=dev)
    x = data["hidden_states"].contiguous()
    M = bs
    nch = 1 if M <= 8 else 2 if M <= 16 else 4
    lmax = (M + 7) // 8 * 8
    NT, NW, RING = CFG(M)
    ids = data["topk_ids"].contiguous()
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
    act = torch.empty(512, dtype=torch.int32, device=dev)
    woff = torch.empty(128, dtype=torch.int32, device=dev)
    nact = torch.empty(1, dtype=torch.int32, device=dev)
    lids = torch.empty(M * 8, dtype=torch.int32, device=dev)
    inter = torch.zeros((M * 8, 4096), dtype=torch.bfloat16, device=dev)
    y = torch.zeros((M * 8, 4096), dtype=torch.bfloat16, device=dev)
    maxslots = min(M * 8, 32 + (M * 8) // (nch * 8))
    mod.wm_route(ids, NOLS, 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
    na = int(nact.item())
    def f0(): mod.wm_gemm_run(0, nch, NT, NW, RING, W13, S13, weights["w13_gs"], x, inter, cnt, lst, act, nact, woff, maxslots, lmax)
    def f1(): mod.wm_gemm_run(1, nch, NT, NW, RING, W2, S2, weights["w2_gs"], inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
    res = []
    for f, nb in ((f0, 16.78e6), (f1, 8.39e6)):
        f(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): f()
        e.record(); torch.cuda.synchronize()
        us = s.elapsed_time(e) / iters * 1000
        bytes_ = na * nb * (1 + 1 / 8)  # weights + scales (1/8)
        res.append((us, bytes_ / us / 1e3))
    print(f"bs={bs} active={na} gateup {res[0][0]:.1f}us {res[0][1]:.0f}GB/s  down {res[1][0]:.1f}us {res[1][1]:.0f}GB/s")


if os.environ.get("WM_BENCH"):
    a = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    b = torch.empty_like(a)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5): b.copy_(a)
    e.record(); torch.cuda.synchronize()
    print("copy BW GB/s (r+w)", 2 * 256e6 * 5 / (s.elapsed_time(e) * 1e-3) / 1e9)
    for bs in (1, 8, 16, 32):
        bench(bs)
