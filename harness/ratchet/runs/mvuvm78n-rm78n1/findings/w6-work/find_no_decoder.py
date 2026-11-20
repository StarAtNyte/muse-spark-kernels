import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

no_dec = []
for r in overall:
    team = r["team"]
    if team not in board_data["muse_decode_layer"]:
        no_dec.append(r)

print(f"Total teams in overall without decoder submission: {len(no_dec)}")
print("First 15 teams in overall without decoder submission:")
for r in no_dec[:15]:
    team = r["team"]
    ranks = {b: board_data[b].get(team, {}).get("rank") for b in boards}
    speedups = {b: board_data[b].get(team, {}).get("speedup") for b in boards}
    print(f"OV {r['rank']:4d} {team:20s} (solved={r['solved']}): ranks={ranks}")
