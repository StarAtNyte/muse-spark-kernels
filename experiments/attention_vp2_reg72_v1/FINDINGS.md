# Attention VP2 72-register cap

Candidate SHA-256 `85ff6eef5bae6b5349688d08a96be2379881fe19ad62f5ec42313169260fa598` replaces the three-block VP2 launch bound with `__maxnreg__(72)`. Paired public-image H200 timing against the three-block build gave parent/candidate ratios 0.994, 0.998 and 0.989 on three verify-1 sliding inputs; all outputs passed the reference check. The cap was 0.2–1.1% slower and is rejected. Actual spill traffic and occupancy were not measured. No official check was spent on this candidate.
