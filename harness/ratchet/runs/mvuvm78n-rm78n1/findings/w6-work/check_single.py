import json

with open("/tmp/board_muse_moe_w4a16.json") as f:
    moe = {r["team"]: r for r in json.load(f)["rows"]}

for t in ["JamesEBall", "sai-lalith", "zsophiaaa"]:
    r = moe.get(t)
    print(t, r["rank"], r["speedup"], r["value"])
