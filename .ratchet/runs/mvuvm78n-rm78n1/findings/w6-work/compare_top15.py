import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

print(f"{'Team':16s} {'OV':3s} {'Dec Val':10s} {'Dec Sp':7s} {'MoE Val':10s} {'Attn Val':10s} {'Proj Val':10s}")
for r in overall[:15]:
    t = r["team"]
    dv = board_data["muse_decode_layer"].get(t, {}).get("value", 0)
    ds = board_data["muse_decode_layer"].get(t, {}).get("speedup", 0)
    mv = board_data["muse_moe_w4a16"].get(t, {}).get("value", 0)
    av = board_data["muse_attention_decode"].get(t, {}).get("value", 0)
    pv = board_data["muse_bf16_proj"].get(t, {}).get("value", 0)
    print(f"{t:16s} {r['rank']:3d} {dv:10.2f} {ds:7.4f} {mv:10.2f} {av:10.2f} {pv:10.2f}")
