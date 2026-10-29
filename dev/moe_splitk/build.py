"""Build dev/moe_splitk/submission.py: the scored MoE submission plus the single-token split-K kernel (sk1.cu).
usage: python dev/moe_splitk/build.py [base_submission] [out]"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '../..'))
base = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/kernels/muse_moe_w4a16/submission.py'
out = sys.argv[2] if len(sys.argv) > 2 else f'{HERE}/submission.py'
src = open(base).read()
cu = open(f'{HERE}/sk1.cu').read()
assert "'''" not in cu

ADD = r'''

# ------------------------------------------------------------------ single-token split-K experts (sk1)
# M == 1: the down projection's K is split into 16-wide chunks so a CTA turns its gate/up rows straight into partial
# down outputs; no CTA waits for another CTA's gate/up results (one grid barrier before the post-expert norm).
SK1 = True
SK1_NS = 4
SK1_SRC = r"""__SK1_CU__"""
_SK1_EXT = {}


def _sk1_ext(ns=None, prof=0):
    ns = SK1_NS if ns is None else ns
    if (ns, prof) not in _SK1_EXT:
        _SK1_EXT[(ns, prof)] = load_inline(
            name=f"muse_sk1_ns{ns}_p{prof}", cpp_sources="void sk1_run(std::vector<torch::Tensor> w, torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, torch::Tensor ctr, torch::Tensor ybuf, int64_t lo, int64_t nloc, int64_t grid, int64_t pdl);", cuda_sources=SK1_SRC, functions=["sk1_run"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", f"-DSK_NS={ns}", f"-DSK_PROF={prof}",
                               "--use_fast_math"], verbose=False)
    return _SK1_EXT[(ns, prof)]


def _sk1_pack_down(codes, sf):
    """codes uint8 [4096 rows, 2048] and e4m3 scales [4096, 256] of one expert's down projection -> chunk-major
    copy [256 chunks of 16 K][32 KB A-fragment words (rb, g, t) | 4 KB scales (rb, g, h)]; a bit permutation only."""
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K)
    n = nib.view(R // 16, 2, 8, K // 16, 4, 4).permute(3, 0, 2, 4, 1, 5).to(torch.int64)  # c, rb, g, t, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            word = _scatter_code_bits(word, n[..., h, j0 + up], Rg, up)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32).contiguous()
    wb = word.view(torch.uint8).view(K // 16, -1)
    s = sf.contiguous().view(torch.uint8).view(R // 16, 2, 8, K // 16).permute(3, 0, 2, 1).contiguous().view(K // 16, -1)
    return torch.cat([wb, s], 1).contiguous().view(-1)


_sk1_prev_prepare = prepare


def prepare(weights):
    state = _sk1_prev_prepare(weights)
    if SK1:
        lo, cnt = state["lo"], state["cnt"]
        dev = weights["w2"].device
        wBc = torch.cat([_sk1_pack_down(weights["w2"][e], weights["w2_sf"][e]) for e in range(lo, lo + cnt)])
        state["sk1_w"] = [state["w"][10], wBc, state["w"][2], state["w"][5], state["w"][6], state["w"][7]]
        state["sk1_ctr"] = torch.zeros(4, dtype=torch.int32, device=dev)
        state["sk1_ybuf"] = torch.zeros(2 * 8 * HIDDEN, dtype=torch.float32, device=dev)
        _sk1_ext()
        torch.cuda.synchronize()
    return state


_sk1_prev_kernel_fn = kernel_fn


def kernel_fn(data, state):
    x = data["hidden_states"]
    if SK1 and x.shape[0] == 1 and "sk1_w" in state:
        cfg = data["config"]
        ids = data["topk_ids"].to(torch.int32).contiguous()
        tw = data["topk_weights"].to(torch.float32).contiguous()
        out = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device=x.device)
        _sk1_ext().sk1_run(state["sk1_w"], x.contiguous(), ids, tw, out, state["sk1_ctr"], state["sk1_ybuf"],
                           cfg["local_start"], cfg["local_count"], state["grid"], 0)
        return out
    return _sk1_prev_kernel_fn(data, state)
'''.replace('__SK1_CU__', cu)
open(out, 'w').write(src.rstrip('\n') + '\n' + ADD)
print('wrote', out, len(src) + len(ADD))
