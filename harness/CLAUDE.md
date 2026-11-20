# Project instructions

Autonomous research mode. Follow the program below in every session.

@PROGRAM.md

## Stop hook

`.claude/hooks/research_nudge.sh` blocks every stop and hands back a rotating strategy. It lets the session stop when:
- `.claude/STOP` exists (user kill switch: `touch .claude/STOP`), or
- the kernel track has closed (after 2026-11-05 07:59 UTC = 13:44 Nepal time), or
- 60 nudges have fired in the current session.

When a nudge arrives, do the strategy, log the run in `results.tsv`, and continue. Do not end the turn on a summary.
