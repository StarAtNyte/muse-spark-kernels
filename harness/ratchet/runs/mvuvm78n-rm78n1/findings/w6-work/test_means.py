import json
import math

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

print("Testing top 10 teams:")
for r in overall[:10]:
    team = r["team"]
    sp = [board_data[b].get(team, {}).get("speedup", 1.0) for b in boards]
    geomean = math.prod(sp) ** (1.0 / len(sp))
    arithmean = sum(sp) / len(sp)
    ranks = [board_data[b].get(team, {}).get("rank", 9999) for b in boards]
    avgrank = sum(ranks) / len(ranks)
    dec_sp = board_data["muse_decode_layer"].get(team, {}).get("speedup", 1.0)
    print(f"OV {r['rank']:2d} {team:18s}: Dec_sp={dec_sp:.4f} Geomean={geomean:.4f} Arith={arithmean:.4f} AvgRank={avgrank:.1f}")
