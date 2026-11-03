import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

# Let's inspect base times
# base_proj = 4830.7268
# base_attn = 3766.3305
# base_moe = 13448.5393
# base_dec = 22620.2471
# Base decode step = base_dec = 22620.25 us.
#
# If a team submits muse_decode_layer, their decode step time is dec_value (or speedup = base_dec / dec_val).
# If a team does NOT submit muse_decode_layer, but submits components:
# What is their synthetic decode step time?
# If component problems replace their respective baseline parts:
# synthetic_time = base_dec - (base_proj - proj_val) - (base_attn - attn_val) - (base_moe - moe_val)
#               = (base_dec - base_proj - base_attn - base_moe) + proj_val + attn_val + moe_val
#               = 574.6505 + proj_val + attn_val + moe_val
#
# Let's test this for taroigarashi2001, ymiled, and others!

base_proj = 4830.726836141095
base_attn = 3766.330521994092
base_moe = 13448.539316480603
base_dec = 22620.247130211792
overhead = base_dec - (base_proj + base_attn + base_moe) # 574.6505

for t in ["taroigarashi2001", "ymiled", "vnicula", "k9thebeast", "jkbpvsc"]:
    pv = board_data["muse_bf16_proj"].get(t, {}).get("value", base_proj)
    av = board_data["muse_attention_decode"].get(t, {}).get("value", base_attn)
    mv = board_data["muse_moe_w4a16"].get(t, {}).get("value", base_moe)
    synth_time = overhead + pv + av + mv
    synth_sp = base_dec / synth_time
    
    # What about if overhead is not added?
    # What if each problem's speedup is used with load weights?
    print(f"Team {t:18s}: synth_time={synth_time:.2f}, synth_sp={synth_sp:.4f}")
