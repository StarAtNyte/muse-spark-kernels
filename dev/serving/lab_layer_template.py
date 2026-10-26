

# ---------------------------------------------------------------------------------------------------------------------
# lab: run the patched SGLang expert layer (real classes, patched sources exec'd under their canonical module names)
# ---------------------------------------------------------------------------------------------------------------------
_WMX_SRC = r'''__WMX__'''
_QMOE_SRC = r'''__QMOE__'''
_EXPERTS_SRC = r'''__EXPERTS__'''


def _lab_inputs(m, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, 4096), generator=g, device=device).to(torch.bfloat16)
    scores = torch.rand((m, 256), generator=g, device=device)
    z = torch.randn((256,), generator=g, device=device)
    u = torch.rand((256,), generator=g, device=device).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return x, ids.to(torch.int32), (routed.values / routed.values.sum(-1, keepdim=True)).float()


def _lab_time(fn, flush, reps=12):
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def _srv_lab(state, weights):
    import importlib
    import sys
    import types

    etp = importlib.import_module("sglang.srt.layers.muse_spark_expert_tp")
    etp.get_parallel = lambda: types.SimpleNamespace(moe_tp_size=1)
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
    from sglang.srt.layers.muse_spark_layers import MuseSpark1xRMSNormWithInputScale
    from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts as StockExperts

    dev = weights["w13"].device
    qc = ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16, exclude_modules=[])

    def make_layer(cls):
        layer = cls(num_experts=256, num_local_experts=32, local_expert_offset=0, hidden_size=4096, intermediate_size=4096,
                    top_k=8, quant_config=qc, params_dtype=torch.bfloat16)
        layer.gate_up_proj.data = weights["w13"][:32].clone()
        layer.gate_up_proj_weight_scale.data = weights["w13_sf"][:32].clone()
        layer.gate_up_proj_weight_scale_2.data = weights["w13_gs"][:32].view(32, 1).expand(32, 2).contiguous()
        layer.down_proj.data = weights["w2"][:32].clone()
        layer.down_proj_weight_scale.data = weights["w2_sf"][:32].clone()
        layer.down_proj_weight_scale_2.data = weights["w2_gs"][:32].contiguous()
        layer.gate_up_proj_input_scale.data = torch.ones((32, 2), dtype=torch.float32, device=dev)
        layer.down_proj_input_scale.data = torch.ones((32,), dtype=torch.float32, device=dev)
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    norm = MuseSpark1xRMSNormWithInputScale(4096, eps=1e-8, zero_centered_gamma=True)
    norm.weight.data = weights["post_norm_w"].float().clone().to(dev)
    norm.weight.data = norm.weight.data.to(torch.bfloat16)
    stock = make_layer(StockExperts)
    print("LABS stock layer ok", flush=True)

    def load(name, src, parent):
        mod = types.ModuleType(name)
        mod.__package__ = name.rpartition(".")[0]
        mod.__file__ = name
        sys.modules[name] = mod
        exec(compile(src, name, "exec"), mod.__dict__)
        setattr(importlib.import_module(parent), name.rpartition(".")[2], mod)
        return mod

    load("sglang.srt.layers.muse_spark_wm_experts", _WMX_SRC, "sglang.srt.layers")
    load("sglang.srt.layers.quantization.muse_spark_nvfp4_moe", _QMOE_SRC, "sglang.srt.layers.quantization")
    patched_mod = load("sglang.srt.layers.muse_spark_fp4_moe", _EXPERTS_SRC, "sglang.srt.layers")
    patched = make_layer(patched_mod.MuseSpark1xFp4Experts)
    other = make_layer(patched_mod.MuseSpark1xFp4Experts)
    print("LABS patched layers ok; wm experts =", patched._wm_experts is not None, flush=True)

    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    for m in (1, 8, 16, 32, 128, 256, 300, 400, 1000):
        x, ids, tw = _lab_inputs(m, dev, 100 + m)
        ref = stock._forward_marlin(x, tw, ids, norm).float()
        out = patched._forward_marlin(x, tw, ids, norm).float()
        err = ((out - ref).norm() / ref.norm()).item()
        # the other layer in between must not disturb the shared scratch
        other._forward_marlin(x, tw, ids, norm)
        out2 = patched._forward_marlin(x, tw, ids, norm).float()
        err2 = ((out2 - ref).norm() / ref.norm()).item()
        t_stock = _lab_time(lambda: stock._forward_marlin(x, tw, ids, norm), flush)
        t_patch = _lab_time(lambda: patched._forward_marlin(x, tw, ids, norm), flush)
        print("LABS m=%d err=%.2e err_after_other_layer=%.2e finite=%d stock=%.1f patched=%.1f" % (
            m, err, err2, int(torch.isfinite(out).all()), t_stock, t_patch), flush=True)
