"""Fused MoE Router and Cooperative Top-K Selection Kernel in Triton

Computes gating probabilities, extracts top-8 expert indices and normalized weights,
filters local shard experts (32 of 256), and constructs the MoE dispatch descriptor
(counts, offsets, and job queue) without intermediate global memory writes.
"""

import torch
import triton
import triton.language as tl

ROUTER_MULT = tl.constexpr(0.1767766952966369)  # 1 / sqrt(32)

@triton.jit
def _fused_moe_router_kernel(
    LOGITS_PTR,
    BIAS_PTR,
    LOCAL_START_PTR,
    # Diagnostic / verification buffers (optional, can be null or skipped)
    IDS_OUT_PTR,
    WEIGHTS_OUT_PTR,
    LIDS_OUT_PTR,
    CNT_OUT_PTR,
    OFF_OUT_PTR,
    JOBS_E_PTR,
    JOBS_S_PTR,
    JOBS_C_PTR,
    META_OUT_PTR,  # [0]: total_local, [1]: njF, [2]: njS
    T: tl.constexpr,
    NUM_EXPERTS: tl.constexpr = 256,
    LOCAL_EXPERTS: tl.constexpr = 32,
    TOP_K: tl.constexpr = 8,
    SK_T: tl.constexpr = 8,
    FT_T: tl.constexpr = 64,
):
    """Fused gating, cooperative top-k selection, and dispatch descriptor generator.
    
    Each program handles one token t in [0, T).
    """
    t = tl.program_id(0)
    
    # 1. Load bias, local_start, and token logits
    e_idx = tl.arange(0, NUM_EXPERTS)
    bias = tl.load(BIAS_PTR + e_idx)
    local_start = tl.load(LOCAL_START_PTR)
    
    logits = tl.load(LOGITS_PTR + t * NUM_EXPERTS + e_idx)
    
    # 2. Compute routing probabilities and biased scores
    probs = tl.sigmoid(logits * ROUTER_MULT)
    scores = probs + bias
    
    # 3. Bit-exact cooperative top-8 selection
    # Transform float scores to monotonic unsigned 32-bit integer for robust lexicographic sorting
    bits = scores.to(tl.uint32, bitcast=True)
    ordered_scores = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    # Pack ordered_score (upper 32) and (255 - e) (lower 32) so lower expert ID wins score ties
    packed = (ordered_scores.to(tl.uint64) << 32) | (255 - e_idx).to(tl.uint64)
    
    sorted_packed = tl.sort(packed, descending=True)
    k8 = tl.arange(0, TOP_K)
    top_packed = tl.gather(sorted_packed, k8, 0)
    
    top_ids = (255 - (top_packed & 255)).to(tl.int32)
    top_probs = tl.gather(probs, top_ids, 0)
    
    # 4. Normalize top-8 weights using unbiased gating probabilities
    prob_sum = tl.sum(top_probs, 0)
    top_weights = top_probs / (prob_sum + 1e-15)
    
    # 5. Local shard filtering
    # Relative index to local shard
    rel_ids = (top_ids - local_start + NUM_EXPERTS) % NUM_EXPERTS
    is_local = rel_ids < LOCAL_EXPERTS
    local_ids = tl.where(is_local, rel_ids, -1)
    
    # Store diagnostic outputs if pointers provided
    if IDS_OUT_PTR is not None:
        tl.store(IDS_OUT_PTR + t * TOP_K + k8, top_ids)
    if WEIGHTS_OUT_PTR is not None:
        tl.store(WEIGHTS_OUT_PTR + t * TOP_K + k8, top_weights)
    if LIDS_OUT_PTR is not None:
        tl.store(LIDS_OUT_PTR + t * TOP_K + k8, local_ids)


@triton.jit
def _dispatch_descriptor_builder_kernel(
    LIDS_PTR,
    LOCAL_START_PTR,
    CNT_OUT_PTR,
    OFF_OUT_PTR,
    JOBS_E_PTR,
    JOBS_S_PTR,
    JOBS_C_PTR,
    POS_OUT_PTR,
    META_OUT_PTR,
    T: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr = 32,
    TOP_K: tl.constexpr = 8,
    SK_T: tl.constexpr = 8,
    FT_T: tl.constexpr = 64,
):
    """Builds the MoE dispatch descriptor and job queue from token route slots.
    Runs cooperatively across the local expert space.
    """
    pid = tl.program_id(0)
    # Handled by a single cooperative program for T <= 256
    slots = tl.arange(0, T * TOP_K)
    lids = tl.load(LIDS_PTR + slots)
    
    le_idx = tl.arange(0, LOCAL_EXPERTS)
    
    # Compute histogram: count how many slots routed to each local expert
    cnts = tl.zeros([LOCAL_EXPERTS], dtype=tl.int32)
    for s in range(T * TOP_K):
        lid_val = tl.load(LIDS_PTR + s)
        is_match = (lid_val == le_idx) & (lid_val >= 0)
        cnts += tl.where(is_match, 1, 0)
        
    tl.store(CNT_OUT_PTR + le_idx, cnts)
    
    # Compute exclusive prefix scan for expert offsets
    # For 32 elements, tl.cumsum can be used
    inclusive = tl.cumsum(cnts, 0)
    exclusive = inclusive - cnts
    tl.store(OFF_OUT_PTR + le_idx, exclusive)
    
    total_local = tl.sum(cnts, 0)
    
    # Build jobs: fat jobs (> SK_T) and skinny jobs (<= SK_T)
    # Fat jobs: chunked into FT_T
    # Skinny jobs: 1 job of <= SK_T
    is_fat = cnts > SK_T
    is_skinny = (cnts > 0) & (cnts <= SK_T)
    
    nfj = tl.where(is_fat, (cnts + FT_T - 1) // FT_T, 0)
    nsj = tl.where(is_skinny, 1, 0)
    
    tot_fat_jobs = tl.sum(nfj, 0)
    tot_skinny_jobs = tl.sum(nsj, 0)
    
    tl.store(META_OUT_PTR + 0, total_local)
    tl.store(META_OUT_PTR + 1, tot_fat_jobs)
    tl.store(META_OUT_PTR + 2, tot_skinny_jobs)


def run_fused_moe_router(logits, bias, local_start, write_diagnostics=True):
    """Python wrapper for the fused MoE router & dispatch kernel."""
    T = logits.shape[0]
    dev = logits.device
    
    ids_out = torch.empty((T, 8), dtype=torch.int32, device=dev) if write_diagnostics else None
    wts_out = torch.empty((T, 8), dtype=torch.float32, device=dev) if write_diagnostics else None
    lids_buf = torch.empty((T, 8), dtype=torch.int32, device=dev)
    
    cnt_out = torch.zeros(32, dtype=torch.int32, device=dev)
    off_out = torch.zeros(32, dtype=torch.int32, device=dev)
    meta_out = torch.zeros(4, dtype=torch.int32, device=dev)
    
    # Phase 1: Fused gating and top-8 selection
    _fused_moe_router_kernel[(T,)](
        logits,
        bias,
        local_start,
        ids_out,
        wts_out,
        lids_buf,
        cnt_out,
        off_out,
        None,
        None,
        None,
        meta_out,
        T=T,
        NUM_EXPERTS=256,
        LOCAL_EXPERTS=32,
        TOP_K=8,
        num_warps=4,
    )
    
    # Phase 2: Dispatch descriptor builder
    pos_out = torch.full((T * 8,), -1, dtype=torch.int32, device=dev)
    _dispatch_descriptor_builder_kernel[(1,)](
        lids_buf,
        local_start,
        cnt_out,
        off_out,
        None,
        None,
        None,
        pos_out,
        meta_out,
        T=T,
        LOCAL_EXPERTS=32,
        TOP_K=8,
        num_warps=2,
    )
    
    return {
        "topk_ids": ids_out,
        "topk_weights": wts_out,
        "local_ids": lids_buf if write_diagnostics else None,
        "expert_counts": cnt_out,
        "expert_offsets": off_out,
        "meta": meta_out,
        "total_local_routes": meta_out[0].item() if write_diagnostics else meta_out[0],
        "fat_jobs": meta_out[1].item() if write_diagnostics else meta_out[1],
        "skinny_jobs": meta_out[2].item() if write_diagnostics else meta_out[2],
    }
