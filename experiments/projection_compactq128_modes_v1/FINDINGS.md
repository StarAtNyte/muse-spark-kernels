# Compact QKV-128 alternative pipeline mode

Compared the selected compact QKV configuration `(64,36,128,3,8,539,5)` with the existing alternate `(64,36,128,3,8,6683,5)` on H200, same weights, three seeds, cold-L2 alternating graph replays. Both outputs passed reference checks. Base → alternative timings were 17.984 → 17.840 µs (1.0081×), 17.904 → 17.872 µs (1.0018×), and 17.968 → 18.064 µs (0.9947×). The difference is neutral and inconsistent; retain the simpler current configuration. Results include event-timing overhead and cover only this shape. Driver and full log are retained.
