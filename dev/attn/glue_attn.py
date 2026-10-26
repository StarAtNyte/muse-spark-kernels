
_FD_MOD = None
_FD_FLAGS = ["-DFD_HW_CVT", "-DFD_NATIVE"]
_FD_LAB = False
_FD_PDL = __PDL__
_PDL_MAXB = __PDLB__
# policy: which shapes use the CUDA split-KV kernel, with (stages, splits_fn)
_FD_POLICY = __POLICY__


def _fd_module():
    global _FD_MOD
    if _FD_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _FD_MOD = load_inline("muse_fd_attn", _FD_CPP, cuda_sources=_FD_CUDA, functions=["fd_run_t", "slide_run_t", "pf_t", "fd2_set_dbg_t", "fd_set_pdl_t"],
                              extra_cuda_cflags=["-O3"] + _FD_FLAGS, verbose=False)
        _FD_MOD.fd_set_pdl_t(int(_FD_PDL and torch.cuda.get_device_capability()[0] >= 9))
    return _FD_MOD


def _fd_forward(data, q, qkv, stages, splits, block=16, qk8=0, variant=1):
    cfg = data['config']
    use_pdl = _FD_PDL and cfg['batch'] <= _PDL_MAXB
    _fd_module().fd_set_pdl_t(int(use_pdl and torch.cuda.get_device_capability()[0] >= 9))
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    scaled = cfg['kv_scale'] != 1.0
    window = cfg['window']
    span = window + length - 1 if window else data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), block) * block
    if variant in (2, 3, 4, 5, 6) and chunk > 4096:
        raise ValueError('chunk too large for fd2')
    bm = max(16, triton.next_power_of_2(length * (hq // hkv)))
    partial = torch.empty((batch, hkv, splits, bm, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, bm), device=qkv.device, dtype=torch.float32)
    _fd_module().fd_run_t(bm // 16, stages + (700 if variant == 6 else 500 if variant == 5 else 400 if variant == 4 else 100 if bm == 64 else 200 if variant == 2 else 300 if variant == 3 else 0), qk8, q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'],
                          data['k_scale'], data['v_scale'], partial, lse, length, hq, hkv, cfg['page_size'], window,
                          splits, chunk, cfg['sm_scale'] * cfg['qk_scale_factor'], int(scaled))
    result = torch.empty((batch, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, bm, num_warps=4, launch_pdl=use_pdl)
    return result


def _slide_forward(data, stages, qkv=None):
    cfg = data['config']
    batch = cfg['batch']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    qkv = (data['qkv'] if qkv is None else qkv).view(batch, -1)
    out = torch.empty((batch, 1, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _fd_module().slide_run_t(stages, qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'],
                             data['page_table'], data['seq_lens'], data['k_scale'], data['v_scale'], out, hq, hkv,
                             cfg['page_size'], cfg['window'], float(cfg['rope_theta']),
                             cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0))
    return out


def _fd_prepare(data):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    qkv = data['qkv'].view(tokens, -1)
    q = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=qkv.device)
    scaled = cfg['kv_scale'] != 1.0
    _prepare_qkv[(tokens, triton.cdiv(hq + hkv, 4))](
        qkv, data['positions'], data['cache_loc'], data['k_scale'], data['v_scale'],
        q, data['k_cache'], data['v_cache'], hq, hkv, d, qkv.stride(0),
        cfg['rope_theta'], scaled, 4, num_warps=4, enable_fp_fusion=False)
    return qkv, q


def _fd_fits(cfg, table_n, splits, variant):
    if variant in (3, 4, 5, 6) and cfg['num_kv_heads'] != 2:
        return False
    if variant in (4, 5) and cfg['q_len'] != 8:
        return False
    if variant not in (2, 3, 4, 5, 6):
        return True
    span = cfg['window'] + cfg['q_len'] - 1 if cfg['window'] else table_n * cfg['page_size']
    return -(-(-(-span // splits)) // 16) * 16 <= 4096


def _fd_choice(cfg, context_bound):
    """(stages, splits) when the CUDA kernel should run this shape, else None."""
    key = ('sliding' if cfg['window'] else 'full', cfg['q_len'] > 1)
    for entry in _FD_POLICY.get(key, ()):
        lo_b, hi_b, stages, splits = entry[:4]
        variant = entry[4] if len(entry) > 4 else 1
        if lo_b <= cfg['batch'] <= hi_b and _fd_fits(cfg, context_bound // cfg['page_size'], splits, variant):
            return stages, splits, variant
    return None


def kernel_fn(data):
    cfg = data['config']
    ctx_bound = data['page_table'].shape[1] * cfg['page_size']
    choice = _fd_choice(cfg, ctx_bound)
    if _FD_LAB and not torch.cuda.is_current_stream_capturing() and not getattr(kernel_fn, '_labbed', False):
        kernel_fn._labbed = True
        _fd_lab()
    if choice is None or ctx_bound > (1 << 20) + 64:
        return _v7_kernel_fn(data)
    stages, splits, variant = choice
    qkv, q = _fd_prepare(data)
    return _fd_forward(data, q, qkv, stages, splits, variant=variant)
