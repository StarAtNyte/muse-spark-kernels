# WGS activation-gather batch 16

Hypothesis: a larger batch of activation rows might expose more independent
loads than the current 8-row group. The exact source snapshot changes only
`OKC_GATHB` from 8 to 16; paired results are in `screen.log`.

All six H200 reference comparisons were bitwise exact. The candidate lost on
every point: 2.81%, 2.89%, 2.74%, 2.64%, 2.83%, and 3.31% slower for M=32,
128, 256 across the two seeds. Reject and retain 8. Driver:
`dev/modal/drv_moe_wgs_gathb16.py`.

All six H200 reference comparisons were bitwise exact, but paired timing showed
consistent regression. Keep GATHB=8.
