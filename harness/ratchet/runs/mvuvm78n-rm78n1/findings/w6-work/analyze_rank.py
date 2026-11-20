import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]

team_ranks = {}
team_speedups = {}

for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        data = json.load(f)
    for row in data["rows"]:
        team = row["team"]
        if team not in team_ranks:
            team_ranks[team] = {}
            team_speedups[team] = {}
        team_ranks[team][b] = row["rank"]
        team_speedups[team][b] = row["speedup"]

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)

print("Total overall rows:", len(overall["rows"]))
for r in overall["rows"][:35]:
    team = r["team"]
    ranks = [team_ranks.get(team, {}).get(b) for b in boards]
    speedups = [team_speedups.get(team, {}).get(b) for b in boards]
    print(f"Rank {r['rank']:2d}: {team:20s} (solved={r['solved']}) ranks={ranks} speedups={[round(s, 3) if s else None for s in speedups]}")
