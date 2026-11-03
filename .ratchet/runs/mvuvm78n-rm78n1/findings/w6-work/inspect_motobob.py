import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

print("--- motobob ---")
for b in boards:
    r = board_data[b].get("motobob")
    if r:
        print(f"{b:25s}: rank={r['rank']:4d}, speedup={r['speedup']:.4f}, value={r['value']:.2f}")
    else:
        print(f"{b:25s}: None")

# Let's inspect teams around OV 976:
print("\nAround OV 976:")
for r in overall[970:982]:
    t = r["team"]
    ranks = {b: board_data[b].get(t, {}).get("rank") for b in boards}
    speedups = {b: board_data[b].get(t, {}).get("speedup") for b in boards}
    print(f"OV {r['rank']:4d} {t:20s} (solved {r['solved']}): ranks={ranks}")
