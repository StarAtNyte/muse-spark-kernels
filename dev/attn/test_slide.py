import ctypes, os, subprocess, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../kernels/muse_attention_decode"))
import reference as R
import attn_cuda as A

H = os.path.abspath(os.path.join(HERE, "../../.cuda_home/home"))
open(os.path.join(HERE, "slide_dev.cu"), "w").write(A.CUDA_SRC)
subprocess.check_call(["clang++", "-x", "cuda", f"--cuda-path={H}", "--cuda-gpu-arch=" + os.environ.get("WM_ARCH", "sm_86"), "-O3", "-std=c++17", "-fPIC", "-shared",
                       "-Wno-unknown-cuda-version", "-Xcuda-ptxas", "-v", "-o", os.path.join(HERE, "slide_dev.so"), os.path.join(HERE, "slide_dev.cu"),
                       f"-L{H}/lib64", "-lcudart", f"-Wl,-rpath,{H}/lib64"])
lib = ctypes.CDLL(os.path.join(HERE, "slide_dev.so"))
P = lambda t: ctypes.c_void_p(t.data_ptr())
stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
dev = "cuda"


def run_slide(data, stages=3):
    cfg = data["config"]
    batch = cfg["batch"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"].view(batch, -1)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=dev)
    scaled = cfg["kv_scale"] != 1.0
    r = lib.slide_run(ctypes.c_int(stages), P(qkv), P(data["positions"]), P(data["cache_loc"]), P(data["k_cache"]), P(data["v_cache"]),
                      P(data["page_table"]), P(data["seq_lens"]), P(data["k_scale"]), P(data["v_scale"]), P(out), ctypes.c_int(batch),
                      ctypes.c_int(hq), ctypes.c_int(hkv), ctypes.c_int(data["page_table"].shape[1]), ctypes.c_int(cfg["page_size"]),
                      ctypes.c_int(cfg["window"]), ctypes.c_float(cfg["rope_theta"]), ctypes.c_float(cfg["sm_scale"] * cfg["qk_scale_factor"]),
                      ctypes.c_int(int(scaled)), ctypes.c_int(qkv.stride(0)), stream())
    assert r == 0
    return out


if __name__ == "__main__":
    cases = [dict(batch=1, ctx=10, layer="sliding", page=64), dict(batch=3, ctx=2047, layer="sliding", page=64),
             dict(batch=2, ctx=5000, layer="sliding", page=1), dict(batch=4, ctx=6000, layer="sliding", dist="edges", page=1),
             dict(batch=3, ctx=4000, layer="sliding", page=1, kv_scale=0.015625), dict(batch=4, ctx=6000, layer="sliding", dist="soft", page=1),
             dict(batch=2, ctx=2048, layer="sliding", page=1), dict(batch=5, ctx=20000, layer="sliding", page=64, spread="mixed")]
    for i, c in enumerate(cases):
        data = R.generate_input(heads="tp8", seed=300 + i, q_len=1, **c)
        for st in (2,):
            d2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
            out = run_slide(d2, st)
            raw = R.collect(d2, out)
            ok, msg = R.check_implementation(d2, raw)
            print(c, "stages", st, ok, msg[:120])
