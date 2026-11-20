import json

with open("/tmp/board_muse_decode_layer.json") as f:
    dec_rows = json.load(f)["rows"]

with open("/tmp/live_overall.json") as f:
    ov_rows = json.load(f)["rows"]

# Current StarAtNyte position:
# Dec speedup: 2.2603, Rank: 28
# Find teams above us:
print("--- TEAMS ABOVE US ON DECODER (Rank 1 to 28) ---")
for r in dec_rows[:28]:
    print(f"Rank {r['rank']:2d}: {r['team']:18s} speedup={r['speedup']:.4f} value={r['value']:.2f}us")

# Let's calculate the derivative d(Overall_Rank) / d(Dec_Speedup) around rank 28:
# Specifically, for each +0.01x in Dec_Speedup:
our_sp = 2.2603
print("\n--- MARGINAL OVERALL RANK GAIN PER +0.01x ON DECODER ---")
for step in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15]:
    target_sp = our_sp + step
    # How many teams have speedup <= target_sp?
    # Rank of target_sp:
    target_rank = 1 + sum(1 for r in dec_rows if r["speedup"] > target_sp)
    rank_gain = 28 - target_rank
    print(f"+{step:.2f}x (speedup {target_sp:.4f}): reaches Rank #{target_rank:2d} (gains {rank_gain:2d} ranks)")
