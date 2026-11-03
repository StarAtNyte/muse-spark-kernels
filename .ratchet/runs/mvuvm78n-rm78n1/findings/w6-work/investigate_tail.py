import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

print(f"Total decoder participants: {len(board_data['muse_decode_layer'])}")
print(f"Total overall rows: {len(overall)}")

# Check rows 360 to 380 in overall
for r in overall[360:385]:
    team = r["team"]
    ranks = {b: board_data[b].get(team, {}).get("rank") for b in boards}
    speedups = {b: board_data[b].get(team, {}).get("speedup") for b in boards}
    print(f"OV rank {r['rank']:3d} ({team:20s}, solved={r['solved']}): dec_rank={ranks['muse_decode_layer']}, all_ranks={ranks}")
