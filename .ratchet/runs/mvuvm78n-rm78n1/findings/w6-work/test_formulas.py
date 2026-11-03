import json
import math

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

# Let's inspect the teams in order of overall rank:
teams_in_ov = [r["team"] for r in overall]

# Let's check different possible formulas:
# Hypothesis 1: Geometric mean of speedup (what if missing is 1.0?)
# Hypothesis 2: Average rank (what if missing rank is N+1?)
# Hypothesis 3: Sum of values (time) (what if missing is stock baseline time?)
# Hypothesis 4: Something with harmonic mean?
# Let's compute for the top 40 teams and teams 360-390:

def test_team(t):
    sp = [board_data[b].get(t, {}).get("speedup") for b in boards]
    rk = [board_data[b].get(t, {}).get("rank") for b in boards]
    val = [board_data[b].get(t, {}).get("value") for b in boards]
    return sp, rk, val

print("Comparing adjacent pairs around 365:")
for t in ["bharath03-a", "taroigarashi2001", "trviv", "osirishorus", "ai-hpc"]:
    sp, rk, val = test_team(t)
    print(f"{t:18s} sp={sp} rk={rk}")
