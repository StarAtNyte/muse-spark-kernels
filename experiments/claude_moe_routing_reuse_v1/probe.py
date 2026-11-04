"""Routing occupancy probe for scored_decoder_fp32_ybuf_v4's MoE fat path (T = 128 / 256).

Runs inside the judge image on one H200 (see run_modal.py). Routing comes from the challenge's own code:
  (A) kernels/muse_moe_w4a16/reference.generate_input  (the MoE board's routing, bs 128 / 256, ep8)
  (B) kernels/muse_decode_layer/reference: generate_input -> ref_front -> router top-8 (the decoder board's
      routing: batch 16 / 32, q_len 8, ctx 80000, spread agentic, dist soft, page 1).
Job accounting mirrors route() in the parent: cnt > SK_T(8): ceil(cnt / FT_T(64)) fat jobs; 1..8: one skinny job.
A fat unit = (job, ub); fat_issue(p, jE[j], ub, k, ...) loads the weight tile from the expert id only, so every job of
an expert issues its own load of identical bytes (no cross-job sharing in software).
"""
import importlib.util, json, math, sys, time

T0 = time.time()
BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 190.0
FT_T, SK_T, NSM = 64, 8, 132
UB_A, UB_B = 32, 16          # RB_A/16 = 512/16, RB_B/16 = 256/16 (phase A: gate+up rows, phase B: down rows)
BLOCK = 16
# bytes of NVFP4 weights+block scales of one expert: w13 [8192, 4096], w2 [4096, 4096]
EXPERT_BYTES = (8192 * 4096 // 2 + 8192 * 4096 // BLOCK) + (4096 * 4096 // 2 + 4096 * 4096 // BLOCK)


def stats(cnt):
    """cnt: 32 per-local-expert route counts."""
    routes = sum(cnt)
    distinct = sum(1 for c in cnt if c > 0)
    fat = [c for c in cnt if c > SK_T]
    skinny = [c for c in cnt if 0 < c <= SK_T]
    jobs_per = [math.ceil(c / FT_T) for c in fat]
    nfj = sum(jobs_per)
    hot = [c for c in cnt if c > FT_T]
    extra = sum(math.ceil(c / FT_T) - 1 for c in hot)
    loads = nfj + len(skinny)                       # one full-expert weight stream per job
    # wave test (grid = 132 CTAs, unit u -> CTA u % 132, wave u // 132, fat queue = LPT by ceil(jC/8) then job index)
    return {
        "routes": routes, "distinct": distinct, "max_cnt": max(cnt),
        "routes_gt64": sum(hot), "frac_routes_gt64": round(sum(hot) / routes, 4) if routes else 0,
        "n_experts_gt64": len(hot), "n_experts_9_64": sum(1 for c in cnt if SK_T < c <= FT_T),
        "n_skinny": len(skinny), "fat_jobs": nfj, "skinny_jobs": len(skinny), "jobs_total": loads,
        "extra_jobs_dup_weights": extra,
        "frac_weight_loads_redundant": round(extra / loads, 4) if loads else 0,
        "fat_units_A": nfj * UB_A, "fat_units_B": nfj * UB_B,
        "weight_GB_unique": round(distinct * EXPERT_BYTES / 1e9, 4),
        "weight_GB_if_every_job_streams": round(loads * EXPERT_BYTES / 1e9, 4),
        "floor_us_unique_4.8TBs": round(distinct * EXPERT_BYTES / 4.8e12 * 1e6, 1),
        "hypothesis_formula_w_times_1_minus_1_over_r": None,
    }


def hyp_formula(cnt):
    """FINDINGS formula w*(1-1/r): w = route fraction on cnt>64 experts, r = mean ceil(cnt/64) over those."""
    hot = [c for c in cnt if c > FT_T]
    routes = sum(cnt)
    if not hot or not routes:
        return 0.0
    w = sum(hot) / routes
    r = sum(math.ceil(c / FT_T) for c in hot) / len(hot)
    return round(w * (1 - 1 / r), 4)


def counts_from_ids(torch, ids, local_start):
    local = (ids.long() - int(local_start)) % 256
    m = local < 32
    return torch.bincount(local[m], minlength=32).tolist()


def load_ref(name, d):
    for m in ("checks", "nvfp4"):
        sys.modules.pop(m, None)
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, d + "/reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.remove(d)
    return mod


def emit(tag, meta, cnt):
    s = stats(cnt)
    s["hypothesis_formula_w_times_1_minus_1_over_r"] = hyp_formula(cnt)
    print("RUN " + json.dumps({"src": tag, **meta, "cnt": sorted(cnt, reverse=True), **s}), flush=True)


def main():
    import torch
    print("GPU", torch.cuda.get_device_name(0), "torch", torch.__version__, "EXPERT_BYTES", EXPERT_BYTES, flush=True)
    M = load_ref("moe_ref", "/w/kernels/muse_moe_w4a16")
    for bs in (128, 256):
        for dist in ("normal", "skewed"):
            for seed in range(16):
                d = M.generate_input(bs, "ep8", seed, dist)
                emit("moe_w4a16.generate_input", {"bs": bs, "dist": dist, "seed": seed, "local_start": 0},
                     counts_from_ids(torch, d["topk_ids"], 0))
    print("MOE_DONE", round(time.time() - T0, 1), flush=True)

    D = load_ref("dec_ref", "/w/kernels/muse_decode_layer")
    w = D.generate_weights(D.WEIGHTS_SEED)
    print("WEIGHTS_DONE", round(time.time() - T0, 1), flush=True)
    routers = {"default(WEIGHTS_SEED)": (w["router"], w["router_bias"])}
    for rs in (1, 2, 3):
        r = D._router(torch.Generator(device="cuda").manual_seed(rs), "cuda", w["pre_ffn_norm"])
        routers[f"router_seed={rs}"] = (r["router"], r["router_bias"])
    cases = [(16, "sliding"), (32, "sliding"), (16, "full"), (32, "full")]
    est = 0.0
    for seed in range(4):
        for batch, layer in cases:
            if time.time() - T0 + 1.5 * est > BUDGET_S:
                print("BUDGET_STOP", seed, batch, layer, round(time.time() - T0, 1), flush=True)
                return
            t1 = time.time()
            with torch.no_grad():
                d = D.generate_input(batch=batch, ctx=80000, layer=layer, seed=seed, q_len=8, spread="agentic",
                                     dist="soft", page=1, local_start=0)
                fr = D.ref_front(d, w)
                for rname, (rw, rb) in routers.items():
                    biased = D.route(fr["h"], {"router": rw, "router_bias": rb})[1]
                    ids = biased.topk(8, dim=-1).indices
                    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
                    ls = 32 * int(pairs.argmax())
                    emit("decoder.ref_front+router", {"batch": batch, "T": batch * 8, "layer": layer, "seed": seed,
                         "router": rname, "local_start": ls}, counts_from_ids(torch, ids, ls))
            est = max(est, time.time() - t1)
            print("CASE_DONE", batch, layer, seed, round(time.time() - t1, 1), flush=True)
    print("DONE", round(time.time() - T0, 1), flush=True)


if __name__ == "__main__":
    main()
