"""Short H200 feasibility/timing probe for Humming's native NVFP4 W4A16 indexed MoE."""
import os
import sys
import time

import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("ninja-build", "libnuma1")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "safetensors", "jinja2", "tqdm", "tabulate", "cuda-bindings")
    .add_local_dir("/tmp/muse-humming", remote_path="/w/humming", copy=True)
    .add_local_dir(f"{ROOT}/kernels/muse_moe_w4a16", remote_path="/w/moe", copy=True)
    .run_commands("pip install -e /w/humming --no-deps")
)

app = modal.App("muse-humming-moe-probe", image=image)


@app.function(gpu="H200", timeout=480)
def run_probe():
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/w/moe")
    import reference as R
    from humming import dtypes
    from humming.config import GemmType, WeightScale2Type, WeightScaleType
    from humming.forward import humming_forward
    from humming.schema import HummingWeightSchema
    from humming.testing.data import generate_moe_tensors
    from humming.transform import prepare_and_transform_humming_tensors

    torch.cuda.init()
    device = torch.device("cuda")
    print("GPU", torch.cuda.get_device_name(), flush=True)
    weights = R.generate_weights(5252)
    start, count = 0, 32
    schema = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e4m3,
        weight_scale_group_size=16,
        weight_scale_type=WeightScaleType.GROUP,
        weight_scale_2_type=WeightScale2Type.TENSOR,
    )

    def transform(wkey, skey, gkey, shape_n, shape_k):
        t = {
            "weight": weights[wkey][start : start + count].contiguous().view(torch.int32),
            "weight_scale": weights[skey][start : start + count].contiguous(),
            "weight_scale_2": weights[gkey][start : start + count].contiguous(),
        }
        cfg, t = prepare_and_transform_humming_tensors(
            t, shape_n, shape_k, schema, num_experts=count, torch_dtype=torch.bfloat16, device=device
        )
        return cfg, t

    cfg13, wt13 = transform("w13", "w13_sf", "w13_gs", 8192, 4096)
    cfg2, wt2 = transform("w2", "w2_sf", "w2_gs", 4096, 4096)
    print("REPACKED", flush=True)
    flush = torch.empty((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

    for m in (16, 256):
        data = R.generate_input(m, "ep8", seed=17319 + m, device="cuda")
        x = data["hidden_states"].contiguous()
        gids = data["topk_ids"].long()
        live = gids < count
        ids = gids.masked_fill(~live, 0).to(torch.int32).contiguous()
        scores = (data["topk_weights"] * live).contiguous()
        _, _, sorted_ids, expert_ids, n_padded = generate_moe_tensors(
            ids, count, GemmType.INDEXED, block_size_config=16
        )
        c13 = cfg13.to_str()
        c2 = cfg2.to_str()
        cc = {"gemm_type": GemmType.INDEXED.value}

        def call():
            h = humming_forward(
                cfg13, x, wt13["weight"], weight_scale=wt13["weight_scale"],
                weight_scale_2=wt13.get("weight_scale_2"), outputs=torch.zeros((m * 8, 8192), device=device, dtype=torch.bfloat16),
                sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=n_padded,
                locks=torch.zeros((2048,), dtype=torch.int32, device=device), top_k=8, compute_config=cc,
            )
            inter = F.silu(h[:, :4096].float()).to(torch.bfloat16) * h[:, 4096:]
            y = humming_forward(
                cfg2, inter, wt2["weight"], weight_scale=wt2["weight_scale"],
                weight_scale_2=wt2.get("weight_scale_2"), outputs=torch.zeros((m * 8, 4096), device=device, dtype=torch.bfloat16),
                sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=n_padded,
                locks=torch.zeros((2048,), dtype=torch.int32, device=device), top_k=1, compute_config=cc,
            )
            y = y.view(m, 8, 4096).float()
            norm_w = (weights["post_norm_w"].float() + 1).view(1, 1, -1)
            y = y * norm_w
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-8)
            return (y * scores.unsqueeze(-1)).sum(1).to(torch.bfloat16)

        out = call()
        ok, msg = R.check_implementation(data, out, weights)
        print("CHECK", m, ok, msg, flush=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(12):
            flush.zero_()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); graph.replay(); b.record(); b.synchronize()
            samples.append(a.elapsed_time(b) * 1000)
        print("TIME_US", m, sorted(samples)[len(samples)//2], sorted(samples), flush=True)
    return "probe complete"


@app.local_entrypoint()
def main():
    print(run_probe.remote())
