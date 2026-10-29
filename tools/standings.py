"""Snapshot the live leaderboards (our rank, score, #50 cutoff, #1) into research/standings/<date>.json and print a
Markdown table for the README.  usage: python tools/standings.py"""
import datetime, json, os, subprocess

OKC = os.path.expanduser("~/.okc/bin/okc")
BOARDS = [("muse_bf16_proj", "Projections"), ("muse_attention_decode", "Attention"), ("muse_moe_w4a16", "MoE"),
          ("muse_decode_layer", "Whole decoder layer"), ("muse_prefill_layer", "Prefill layer")]


def board(name=None):
    cmd = [OKC, "leaderboard"] + ([name] if name else []) + ["--json"]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    snap = {"captured_at_utc": now.isoformat(), "boards": {}}
    overall = board()
    snap["overall_you"] = overall.get("you")
    lines = ["| Problem | Public score | Rank | Top-50 cutoff | #1 score |", "|---|---:|---:|---:|---:|"]
    for name, title in BOARDS:
        b = board(name)
        rows = b.get("rows", [])
        r50 = next((r for r in rows if r.get("rank") == 50), None)
        you, lead = b.get("you") or {}, b.get("leader") or {}
        snap["boards"][name] = {"you": you, "leader": lead, "rank_50": r50}
        lines.append(f"| {title} | {you.get('speedup', 0):.4f}x | #{you.get('rank')} | "
                     f"{(r50 or {}).get('speedup', 0):.4f}x | {lead.get('speedup', 0):.4f}x |")
    os.makedirs("research/standings", exist_ok=True)
    path = f"research/standings/{now.date().isoformat()}.json"
    with open(path, "w") as f:
        json.dump(snap, f, indent=1)
    print("\n".join(lines))
    print(f"\nOverall rank: #{(snap['overall_you'] or {}).get('rank')} (snapshot {now:%Y-%m-%d %H:%M} UTC, {path})")


if __name__ == "__main__":
    main()
