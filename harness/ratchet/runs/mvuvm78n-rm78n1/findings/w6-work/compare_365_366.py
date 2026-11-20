import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

for t in ["bharath03-a", "taroigarashi2001", "trviv", "ymiled"]:
    print(f"\n--- {t} ---")
    for b in boards:
        r = board_data[b].get(t)
        if r:
            print(f"{b:25s}: rank={r['rank']:4d}, speedup={r['speedup']:.4f}, value={r['value']:.2f}")
        else:
            print(f"{b:25s}: None")
