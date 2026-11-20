import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]
top_teams = ["naklecha", "tjiang-fair", "dolanzhao", "mpeng19", "zzz0906", "vjeux", "colinlikescode", "pjmullin3", "alygg77"]

for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        d = json.load(f)
    print(f"\n=== {b} ===")
    team_map = {r["team"]: r for r in d["rows"]}
    for t in top_teams:
        if t in team_map:
            r = team_map[t]
            print(f"  {t:15s}: rank={r['rank']:3d}, speedup={r['speedup']:.4f}, job={r.get('job')}")
