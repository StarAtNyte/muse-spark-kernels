
_DMOD = None
_DENSE_EXTRA = []
_DENSE_PDL = True
_DENSE_PF = False
# (layer, m upper bound) -> (ntb, nch, nw, s, ksplit)
_DENSE_CFG = __CFG__


def _dense_module():
    global _DMOD
    if _DMOD is None:
        from torch.utils.cpp_extension import load_inline
        _DMOD = load_inline("muse_dense_bf16", _DENSE_CPP, cuda_sources=_DENSE_CUDA, functions=["dense_gemm", "dense_cgemm", "dense_set_pdl_t", "dense_set_pf_t", "dense_set_dbg_t", "dense2_gemm", "dense2c_gemm", "probe_t"],
                            extra_cuda_cflags=["-O3", "--use_fast_math"] + _DENSE_EXTRA, verbose=False)
        _DMOD.dense_set_pdl_t(int(_DENSE_PDL and torch.cuda.get_device_capability()[0] >= 9))
        _DMOD.dense_set_pf_t(int(_DENSE_PF and torch.cuda.get_device_capability()[0] >= 9))
    return _DMOD


def prepare(weights):
    state = _old_prepare(weights)
    state['dense_mod'] = _dense_module()
    hi = weights['router'].to(torch.bfloat16)
    lo = (weights['router'] - hi.float()).to(torch.bfloat16)
    state['d_router'] = _dense_repack(torch.cat([hi, lo], 0).contiguous())
    # self-resetting reduction counters, one set per layer (the router runs on its own stream in the decoder)
    state['d_cnt'] = {name: torch.zeros(4096, dtype=torch.int32, device=hi.device)
                      for name in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')}
    for name in ('qkv_gate', 'o', 'pooler', 'unpooler'):
        state['d_' + name] = _dense_repack(weights[name].contiguous())
        state['c_' + name] = _dense_crepack(weights[name].contiguous())
    torch.cuda.synchronize()
    if _DENSE_LAB:
        _dense_lab(state)
    return state


def _dense_crun(x, state, layer, n, cfg):
    ntb, nch, nw, s, ksplit = cfg[:5]
    fix = cfg[5] if len(cfg) > 5 else 0
    m, k = x.shape
    tiles, emax, eoff, esc = state['c_' + layer]
    nt = n // 16
    part = torch.empty((ksplit, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    state['dense_mod'].dense_cgemm(ntb, nch, nw, s, tiles[:nt], x, part, out, ksplit, n, state['d_cnt'][layer], fix, emax, eoff[:nt + 1], esc)
    return out


def _dense_cfg(layer, m):
    table = _DENSE_CFG.get(layer)
    if table is None:
        return None
    for bound, cfg in table:
        if m <= bound:
            return cfg
    return table[-1][1]


def _dense_run(x, state, layer, n, cfg):
    ntb, nch, nw, s, ksplit = cfg[:5]
    fix = cfg[5] if len(cfg) > 5 else 0
    m, k = x.shape
    router = layer == 'router'
    wt = state['d_' + layer]
    ntot = 512 if router else n
    wt = wt[: ntot // 16]
    part = torch.empty((ksplit, m, ntot), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    state['dense_mod'].dense_gemm(ntb, nch, nw, s, wt, x, part, out, ksplit, 1 if router else 0, n, state['d_cnt'][layer], fix)
    return out


def kernel_fn(data, state):
    x = data['x']
    layer = data['config']['layer']
    n = data['config']['n']
    cfg = _dense_cfg(layer, x.shape[0])
    if cfg is None:
        return _old_kernel_fn(data, state)
    return _dense_run(x.contiguous(), state, layer, n, cfg)

_DENSE_LAB = False
