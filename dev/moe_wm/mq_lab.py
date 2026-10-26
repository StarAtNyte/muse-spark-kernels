def _mq_lab(state, weights):
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    global _MQ_S, _MQ_SX
    for m in (1, 8, 16, 32, 128, 256):
        data = _lab_inputs(m, 'cuda', 100 + m)
        ref = _marlin_kernel_fn(data, state).float()
        row = ['m=%d' % m, 'marlin=%.1f' % _lab_time(lambda: _marlin_kernel_fn(data, state), flush)]
        for s in (2, 3, 4):
            _MQ_S, _MQ_SX = s, s
            try:
                o = kernel_fn(data, state).float()
                err = ((o - ref).norm() / ref.norm()).item()
                row.append('s%d=%.1f(%.0e)' % (s, _lab_time(lambda: kernel_fn(data, state), flush), err))
            except Exception as ex:
                row.append('s%d=ERR' % s)
        _MQ_S, _MQ_SX = 4, 3
        print('LABM', ' '.join(row), flush=True)
