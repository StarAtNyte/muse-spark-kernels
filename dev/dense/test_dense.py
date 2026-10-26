import ctypes, os, subprocess, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dense_cuda as D

H = os.path.abspath(os.path.join(HERE, "../../.cuda_home/home"))
ARCH = os.environ.get("WM_ARCH", "sm_86")
open(os.path.join(HERE, "dense_dev.cu"), "w").write(D.CUDA_SRC)
subprocess.check_call(["clang++", "-x", "cuda", f"--cuda-path={H}", f"--cuda-gpu-arch={ARCH}", "-O3", "-std=c++17", "-fPIC", "-shared",
                       "-Wno-unknown-cuda-version", "-o", os.path.join(HERE, "dense_dev.so"), os.path.join(HERE, "dense_dev.cu"),
                       f"-L{H}/lib64", "-lcudart", f"-Wl,-rpath,{H}/lib64"])
lib = ctypes.CDLL(os.path.join(HERE, "dense_dev.so"))
lib.dense_set_pdl(0)
lib.dense_set_pf(0)
P = lambda t: ctypes.c_void_p(t.data_ptr())
stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
dev = "cuda"


def run(W, X, cfg, ksplit, pair=0, nout=None, fp32=False, fix=False):
    ntb, nch, nw, s = cfg
    M, K = X.shape
    N = W.shape[0] * 16 * 0 if False else W.shape[0] * 16
    nout = nout or N
    part = torch.empty((ksplit, M, N), dtype=torch.float32, device=dev)
    out = torch.empty((M, nout), dtype=torch.float32 if fp32 else torch.bfloat16, device=dev)
    direct = (ksplit == 1 and not pair and not fp32)
    om = 1 if direct else (2 if (fix and ksplit > 1) else 0)
    cnt = torch.zeros(4096, dtype=torch.int32, device=dev)
    r = lib.dense_run(ctypes.c_int(ntb), ctypes.c_int(nch), ctypes.c_int(nw), ctypes.c_int(s), ctypes.c_int(om), P(W), P(X), P(out) if direct else P(part), ctypes.c_int(M),
                      ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(ksplit), P(cnt), P(out), ctypes.c_int(pair), ctypes.c_int(nout), ctypes.c_int(int(fp32)), stream())
    assert r == 0, "cfg"
    if om == 0: lib.dense_reduce(P(part), P(out), ctypes.c_int(M), ctypes.c_int(nout), ctypes.c_int(N), ctypes.c_int(ksplit), ctypes.c_int(pair),
                     ctypes.c_int(int(fp32)), stream())
    return out


torch.manual_seed(0)
for (N, K) in ((2304, 8192), (1024, 1024 * 4)):
    w = (torch.randn(N, K, device=dev) * 0.05).to(torch.bfloat16)
    Wt = D.dense_repack(w)
    for M in (1, 8, 16, 32, 128):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        ref = x.float() @ w.float().T
        for cfg, ks in (((1, 1, 4, 4), 1), ((1, 1, 4, 4), 2), ((1, 2, 4, 4), 1), ((1, 4, 4, 4), 1), ((2, 1, 4, 4), 1), ((1, 1, 8, 4), 1)):
            try:
                out = run(Wt, x, cfg, ks, fix=bool(os.environ.get('FIX')))
            except AssertionError:
                continue
            err = (out.float() - ref).norm() / ref.norm()
            print(N, K, M, cfg, ks, f"relerr {err:.2e}", "OK" if err < 5e-3 else "BAD")

if os.environ.get("BENCH"):
    N, K = 4096, 8192
    w = (torch.randn(N, K, device=dev) * 0.05).to(torch.bfloat16)
    Wt = D.dense_repack(w)
    for M in (1, 16, 32):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        for cfg, ks in (((1, 1, 4, 4), 1), ((1, 1, 4, 4), 2), ((1, 1, 8, 4), 1), ((2, 1, 4, 4), 1), ((1, 2, 4, 4), 1), ((1, 4, 4, 4), 1)):
            if M == 1 and cfg[1] > 1: continue
            out = run(Wt, x, cfg, ks)
            torch.cuda.synchronize()
            s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s_.record()
            for _ in range(10): run(Wt, x, cfg, ks)
            e_.record(); torch.cuda.synchronize()
            us = s_.elapsed_time(e_) / 10 * 1000
            print(f"M={M} cfg={cfg} ks={ks}: {us:.0f}us {N*K*2/us/1e3:.0f} GB/s")


if os.environ.get("ROUTER"):
    router = torch.randn(256, 8192, device=dev) * 0.05
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    Wt = D.dense_repack(torch.cat([hi, lo], 0).contiguous())
    for M in (1, 8, 32, 128):
        x = torch.randn(M, 8192, device=dev).to(torch.bfloat16)
        ref = x.float() @ router.T
        for cfg, ks, fix in (((1, 1, 4, 4), 4, False), ((1, 1, 4, 4), 4, True), ((1, 2, 4, 4), 8, True), ((2, 4, 4, 4), 2, True), ((1, 4, 4, 4), 1, False)):
            try:
                out = run(Wt, x, cfg, ks, pair=1, nout=256, fp32=True, fix=fix)
            except AssertionError:
                continue
            err = (out - ref).norm() / ref.norm()
            print("router", M, cfg, ks, fix, f"relerr {err:.2e}", "OK" if err < 1e-4 else "BAD")
