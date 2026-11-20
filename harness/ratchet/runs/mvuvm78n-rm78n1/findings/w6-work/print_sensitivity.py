import json

# Baseline values
base = {
    "proj": 4830.7268,
    "attn": 3766.3305,
    "moe": 13448.5393,
    "dec": 22620.2471,
    "prefill": 387403.35
}

# StarAtNyte current
current_sp = {
    "proj": 1.5395,
    "attn": 2.6177,
    "moe": 2.3820,
    "dec": 2.2603,
    "prefill": 1.7874
}

# Leaders
leader_sp = {
    "proj": 1.6498,   # naklecha
    "attn": 3.2221,   # tjiang-fair
    "moe": 2.7578,    # dolanzhao
    "dec": 2.4178,    # mpeng19
    "prefill": 2.0325 # zzz0906
}

with open("/tmp/board_muse_decode_layer.json") as f:
    dec_rows = json.load(f)["rows"]

def get_rank_for_sp(sp):
    return 1 + sum(1 for r in dec_rows if r["speedup"] > sp)

our_dec_time = base["dec"] / current_sp["dec"] # 10007.63 us

print("=== MARGINAL SENSITIVITY TABLE ===")
print(f"Current Decoder Speedup: {current_sp['dec']:.4f}x (Time: {our_dec_time:.2f} us, Rank #{get_rank_for_sp(current_sp['dec'])})")

results = []
for name in ["moe", "proj", "attn", "dec", "prefill"]:
    if name == "prefill":
        results.append({
            "board": name,
            "dTime_per_001x": 0.0,
            "dDecSp_per_001x": 0.0,
            "dec_sp_plus_001x": current_sp["dec"],
            "rank_gain_001x": 0,
            "max_transfer_sp": current_sp["dec"],
            "max_rank": get_rank_for_sp(current_sp["dec"]),
            "leader_sp": leader_sp[name]
        })
        continue

    if name == "dec":
        # Direct decoder improvement
        target_sp = current_sp["dec"] + 0.01
        gain = get_rank_for_sp(current_sp["dec"]) - get_rank_for_sp(target_sp)
        results.append({
            "board": name,
            "dTime_per_001x": base["dec"]/current_sp["dec"] - base["dec"]/(current_sp["dec"]+0.01),
            "dDecSp_per_001x": 0.01,
            "dec_sp_plus_001x": target_sp,
            "rank_gain_001x": gain,
            "max_transfer_sp": leader_sp["dec"],
            "max_rank": 1,
            "leader_sp": leader_sp[name]
        })
        continue

    # Component boards:
    b_val = base[name]
    sp0 = current_sp[name]
    sp1 = sp0 + 0.01
    dt = (b_val / sp0) - (b_val / sp1) # time saved
    new_dec_time = our_dec_time - dt
    new_dec_sp = base["dec"] / new_dec_time
    d_dec_sp = new_dec_sp - current_sp["dec"]
    gain = get_rank_for_sp(current_sp["dec"]) - get_rank_for_sp(new_dec_sp)

    # If component reaches leader:
    dt_leader = (b_val / sp0) - (b_val / leader_sp[name])
    leader_dec_time = our_dec_time - dt_leader
    leader_transfer_dec_sp = base["dec"] / leader_dec_time
    leader_transfer_rank = get_rank_for_sp(leader_transfer_dec_sp)

    results.append({
        "board": name,
        "dTime_per_001x": dt,
        "dDecSp_per_001x": d_dec_sp,
        "dec_sp_plus_001x": new_dec_sp,
        "rank_gain_001x": gain,
        "max_transfer_sp": leader_transfer_dec_sp,
        "max_rank": leader_transfer_rank,
        "leader_sp": leader_sp[name]
    })

for r in results:
    print(f"\nBoard: {r['board'].upper()}")
    print(f"  Leader speedup: {r['leader_sp']:.4f}x")
    print(f"  Time saved per +0.01x: {r['dTime_per_001x']:.2f} us")
    print(f"  Decoder speedup change per +0.01x: +{r['dDecSp_per_001x']:.5f}x")
    print(f"  Immediate overall rank gain per +0.01x: {r['rank_gain_001x']} ranks")
    print(f"  If transferred at leader speedup: Decoder = {r['max_transfer_sp']:.4f}x -> Rank #{r['max_rank']}")
