# Attention VP2 four-block launch bound

Candidate SHA-256 `bc1cdf3dbe1d987c7f39c29635124ab564af7ea03fbcbddce52368c3e1004c36` changes VP2 launch bounds from `(256,3)` to `(256,4)`. Paired public-image H200 measurements compared against the three-block source on three independently generated batch-1 sliding verify inputs; all six outputs passed the reference check. Parent/candidate ratios were 0.952, 0.945, and 0.937. The four-block setting is 5–6% slower on this screen and is rejected. Register spills and achieved occupancy were not measured, so the cause is unproven. No official check or scored submission was made.

The two-block scored job `6f2b2edf215d4ea3` and three-block private check `f350069f2f614336` were still live at the time of this screen. Raw output, source and comparison driver are archived here.
