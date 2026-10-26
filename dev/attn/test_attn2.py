import ctypes, os, subprocess, sys, time
import torch, triton
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../kernels/muse_attention_decode"))
import reference as R
import attn_cuda as A
import attn_triton as T

H = os.path.abspath(os.path.join(HERE, "../../.cuda_home/home"))
ARCH = os.environ.get("WM_ARCH", "sm_86")
open(os.path.join(HERE, "attn_dev.cu"), "w").write(A.CUDA_SRC)
subprocess.check_call(["clang++", "-x", "cuda", f"--cuda-path={H}", f"--cuda-gpu-arch={ARCH}", "-O3", "-std=c++17", "-fPIC", "-shared",
                       "-Wno-unknown-cuda-version", "-Xcuda-ptxas", "-v", "-o", os.path.join(HERE, "attn_dev.so"), os.path.join(HERE, "attn_dev.cu"),
                       f"-L{H}/lib64", "-lcudart", f"-Wl,-rpath,{H}/lib64"])
lib = ctypes.CDLL(os.path.join(HERE, "attn_dev.so"))
P = lambda t: ctypes.c_void_p(t.data_ptr())
stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
dev = "cuda"


def run(data, splits=None, block=16):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    q_in, k_in, v_in, gate = R.split_qkv(data)
    pos = data["seq_lens"].long()[:, None] + torch.arange(length, device=dev)
    qn, kn = R._rms(q_in), R._rms(k_in)
    if cfg["rope_theta"]:
        qn, kn = R._rope(qn, pos, cfg["rope_theta"]), R._rope(kn, pos, cfg["rope_theta"])
    q = qn.to(torch.float8_e4m3fn).reshape(tokens, hq, d).contiguous()
    scaled = cfg["kv_scale"] != 1.0
    k8, v8 = R._quant(kn, data["k_scale"]), R._quant(v_in, data["v_scale"])
    data["k_cache"].view(-1, hkv, d)[data["cache_loc"]] = k8.reshape(tokens, hkv, d)
    data["v_cache"].view(-1, hkv, d)[data["cache_loc"]] = v8.reshape(tokens, hkv, d)
    window = cfg["window"]
    span = window + length - 1 if window else data["page_table"].shape[1] * cfg["page_size"]
    splits = splits or int(os.environ.get("FD_SPL", "0")) or (4 if window else 16)
    chunk = triton.cdiv(triton.cdiv(span, splits), 16) * 16
    bm = max(16, triton.next_power_of_2(length * (hq // hkv)))
    mt = bm // 16
    partial = torch.empty((batch, hkv, splits, bm, d), device=dev, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, bm), device=dev, dtype=torch.float32)
    r = lib.fd_run(ctypes.c_int(mt), ctypes.c_int(int(os.environ.get("FD_ST", "4"))), ctypes.c_int(0), P(q), P(data["k_cache"]), P(data["v_cache"]), P(data["page_table"]), P(data["seq_lens"]),
                   P(data["k_scale"]), P(data["v_scale"]), P(partial), P(lse), ctypes.c_int(batch), ctypes.c_int(length),
                   ctypes.c_int(hq), ctypes.c_int(hkv), ctypes.c_int(data["page_table"].shape[1]), ctypes.c_int(cfg["page_size"]),
                   ctypes.c_int(window), ctypes.c_int(splits), ctypes.c_int(chunk),
                   ctypes.c_float(cfg["sm_scale"] * cfg["qk_scale_factor"]), ctypes.c_int(int(scaled)), stream())
    if r != 0:
        return None
    result = torch.empty((batch, length, hq * d), device=dev, dtype=torch.bfloat16)
    T._merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, bm, num_warps=4)
    return result


if __name__ == "__main__":
    cases = [dict(batch=1, ctx=10, layer="sliding", q_len=1, page=64), dict(batch=8, ctx=8192, layer="full", q_len=1, spread="mixed", heads="dp", page=64),
             dict(batch=4, ctx=6000, layer="sliding", q_len=1, dist="edges", page=1), dict(batch=4, ctx=6000, layer="full", q_len=8, dist="edges", page=64),
             dict(batch=4, ctx=80000, layer="full", q_len=1, spread="mixed", dist="soft", page=64), dict(batch=2, ctx=20000, layer="full", q_len=8, heads="dp", dist="soft", page=64),
             dict(batch=4, ctx=80000, layer="full", q_len=1, spread="mixed", dist="coherent", page=1), dict(batch=3, ctx=2047, layer="sliding", q_len=8, page=64)]
    for i, c in enumerate(cases):
        c = dict(c); heads = c.pop("heads", "tp8"); data = R.generate_input(heads=heads, seed=100 + i, **c)
        out = run(data)
        if out is None:
            print(c, 'SKIP'); continue
        raw = R.collect(data, out)
        ok, msg = R.check_implementation(data, raw)
        print(c, ok, msg)
