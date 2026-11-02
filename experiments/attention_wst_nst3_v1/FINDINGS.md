# Balanced WST: three versus four pipeline stages

Candidate SHA-256 `9843de8a85d13fe607f1dd698562449ce10ba78566565929ed5ac4e2be49095b` changes WST's launch template from four to three stages while keeping four producer warpgroups and the B32 WST extension in both arms. Public judge-image H200 paired graph screen used three seeds each at B16 and B32 full verification, 24 alternating read-flushed replays; all outputs passed reference checks. Four-stage/three-stage ratios were 0.988/0.987/0.998 at B16 and 0.991/0.990/0.986 at B32. The three-stage version was 0.2–1.4% slower in these samples, so no official check was spent on it. Actual stall and occupancy causes were not profiled.

Raw output and driver are in this directory. The separate four-stage B32 extension is under official check `b1a3f689acbc49fe`.
