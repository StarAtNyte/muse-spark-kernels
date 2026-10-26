import ctypes, os, subprocess, sys, math
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
lib.dense_set_pdl(0); lib.dense_set_pf(0)
P = lambda t: ctypes.c_void_p(t.data_ptr())
stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
dev = "cuda"
torch.manual_seed(0)
for (N, K) in ((2304, 8192), (1024, 4096), (512, 1024)):
    w = (torch.randn(N, K, device=dev) / math.sqrt(K)).to(torch.bfloat16)
    pk = D.dense_crepack(w)
    tiles, emax, eoff, esc = pk
    print(N, K, "emax", emax, "escapes", esc.shape[0], "ratio", tiles.numel() / (w.numel() * 2))
    for M in (1, 8, 16, 32, 128):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        ref = x.double() @ w.double().T
        for cfg, ks, fix in (((1, 1, 4, 4), 1, 0), ((1, 1, 4, 4), 2, 0), ((1, 2, 4, 4), 2, 1), ((1, 4, 4, 4), 1, 0), ((2, 1, 4, 4), 1, 0)):
            ntb, nch, nw, s = cfg
            direct = ks == 1
            om = 1 if direct else (2 if fix else 0)
            part = torch.empty((ks, M, N), dtype=torch.float32, device=dev)
            out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
            cnt = torch.zeros(4096, dtype=torch.int32, device=dev)
            r = lib.dense_crun(ctypes.c_int(ntb), ctypes.c_int(nch), ctypes.c_int(nw), ctypes.c_int(s), ctypes.c_int(om), P(tiles), P(x),
                               P(out) if direct else P(part), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(ks), P(cnt), P(out),
                               ctypes.c_int(N), ctypes.c_int(emax), P(eoff), P(esc), stream())
            if r:
                continue
            if om == 0:
                lib.dense_reduce(P(part), P(out), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(ks), ctypes.c_int(0), ctypes.c_int(0), stream())
            torch.cuda.synchronize()
            err = ((out.double() - ref).norm() / ref.norm()).item()
            print(N, K, M, cfg, ks, f"relerr {err:.2e}", "OK" if err < 5e-3 else "BAD")
