#!/usr/bin/env bash
# Stop hook: block the stop and hand Claude a rotating kernel-research strategy.
DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude"
input=$(cat)
sid=$(printf '%s' "$input" | jq -r '.session_id // "default"' 2>/dev/null)
[ -z "$sid" ] && sid=default

# Kill switch.
[ -f "$DIR/STOP" ] && exit 0
# Kernel track closed: 2026-11-04 23:59 PT = 2026-11-05 07:59 UTC.
[ "$(date -u +%s)" -ge "$(date -u -d '2026-11-05 07:59' +%s)" ] && exit 0

# Per-session counter with hard cap.
mkdir -p "$DIR/nudge"
STATE="$DIR/nudge/$sid"
n=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$STATE"
[ "$n" -gt 60 ] && exit 0

freeze=""
[ "$(date -u +%s)" -ge "$(date -u -d '2026-11-05 01:59' +%s)" ] && \
  freeze="FEATURE FREEZE IS ACTIVE: only resubmit the best files (okc submit) on every board; no new code. "

strategies=(
 "Run the ratchet on muse_decode_layer: read okc leaderboard muse_decode_layer and research/best.json, find the slowest shape and sub-kernel from the judge profile, make one change, okc check it."
 "Port check: is any moe / attention / proj improvement not yet in muse_decode_layer? Port it, okc check, submit if it beats the best beyond the CI."
 "WILD experiment: write 10 unfiltered design-changing ideas in WILD.md (fusion boundary, data layout, scheduling, kernel structure), pick the one least like anything in results.tsv, branch wild/<name>, okc check it."
 "Roofline the weakest board relative to the leaderboard median: bytes and FLOPs per shape vs H200 (4.8 TB/s, 132 SMs). If above 80% of roofline, move to fusion or another problem; otherwise attack the worst shape."
 "Read public repos of top teams (search GitHub for Open Kernel Challenge / okc muse). Add 5 untried ideas to IDEAS.md with credit notes and run the best one."
 "Spawn a fresh subagent given only results.tsv, NOTES.md, IDEAS.md and the problem TASK.md (none of this conversation). Ask it for 5 ideas, run its most unusual one."
 "Question an assumption: launch count, CUDA-graph behavior, fusion boundary, dequant placement, data layout, or whether cuBLAS/FA3/Marlin beats our kernel at some shapes. Ablate it with okc check."
 "Free upside: check okc problems for unused scored quota and resubmit the best file of each board (leaderboard keeps the best run). Then continue the ratchet."
)
msg=${strategies[$(( (n-1) % ${#strategies[@]} ))]}
jq -cn --arg r "Do not stop. ${freeze}Strategy #$n: $msg Log every judge run in results.tsv and facts/negative results in NOTES.md. Follow PROGRAM.md." \
  '{decision:"block", reason:$r}'
