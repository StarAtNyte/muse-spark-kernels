import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]

for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        d = json.load(f)
    print(f"\n--- {b} ---")
    print("measure:", d.get("measure"))
    rows = d.get("rows", [])
    if rows:
        r0 = rows[0]
        # value is presumably time? speedup = baseline / value ? Or value / baseline?
        # Let's check r0['value'] * r0['speedup']
        print(f"Top row: team={r0['team']} speedup={r0['speedup']} value={r0['value']}")
        print(f"Product (value * speedup) = {r0['value'] * r0['speedup']}")
