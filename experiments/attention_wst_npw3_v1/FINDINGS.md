# Balanced WST: three versus four producer warpgroups

Candidate SHA-256 `a6d004854e7c397c82fb00f6b2c87afbbb35be1e193c2941112f5f8c404b9855` changes the WST launch template from four to three producer warpgroups, with the B32 WST extension enabled in both arms. Public judge-image H200 paired graph screen used three seeds each at B16 and B32 full verification, 24 alternating read-flushed replays; all outputs passed reference checks. Four-group/three-group ratios were 0.988/0.990/0.985 at B16 and 0.987/0.985/0.986 at B32. The three-group version is consistently 1.0–1.5% slower and is rejected; no official check was used for it. Actual occupancy, memory stalls, and register counts were not profiled, so the cause is not established.

Raw output and driver are in this directory. The separate four-group B32 extension is under official check `b1a3f689acbc49fe`.
