import json

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

with open("/tmp/board_muse_decode_layer.json") as f:
    dec = json.load(f)["rows"]

dec_team_to_rank = {r["team"]: r["rank"] for r in dec}
dec_rank_to_team = {r["rank"]: r["team"] for r in dec}

print(f"Total overall: {len(overall)}, Total dec: {len(dec)}")

# Check for each dec team, what is their ov rank?
rank_diffs = []
for r in dec:
    team = r["team"]
    # find team in overall
    ov_matches = [o for o in overall if o["team"] == team]
    if ov_matches:
        ov_rank = ov_matches[0]["rank"]
        diff = ov_rank - r["rank"]
        rank_diffs.append((r["rank"], ov_rank, diff, team))

print("First 20 dec teams in overall:")
for d_r, o_r, diff, team in rank_diffs[:20]:
    print(f"Dec #{d_r:4d} -> OV #{o_r:4d} (diff={diff:2d}) : {team}")

print("\nLast 20 dec teams in overall:")
for d_r, o_r, diff, team in rank_diffs[-20:]:
    print(f"Dec #{d_r:4d} -> OV #{o_r:4d} (diff={diff:2d}) : {team}")

max_diff = max(d[2] for d in rank_diffs)
min_diff = min(d[2] for d in rank_diffs)
print(f"\nMin diff: {min_diff}, Max diff: {max_diff}")
