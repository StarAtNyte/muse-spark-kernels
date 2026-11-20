import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

ov_row = next(r for r in overall if r["team"] == "iemAnshuman")
print("iemAnshuman in overall:", ov_row)

for b in boards:
    r = board_data[b].get("iemAnshuman")
    if r:
        print(f"{b:25s}: rank={r['rank']:4d}, speedup={r['speedup']:.4f}, value={r['value']:.2f}")
    else:
        print(f"{b:25s}: None")
