# Projection candidate without exhaustive off-score routing

V5 passed 9/9 at 1.54735× and its official profile confirmed both QKV-32 SK2 and QKV-128 compact paths. Its inherited V6 prepare routine also timed 1,738 layer/token-count cells and selected 973 individual routes for unscored sizes. Although that code attempts to improve nearby inputs rather than hard-code scored outputs, this degree of specialization warrants caution under the organizer's guidance against extensive per-shape tuning overfitted to the challenge.

This candidate disables that exhaustive routing policy and retains the ordinary fallback and both measured QKV improvements. Private check `05a1587da81d4434` is pending. Nearby-shape correctness and speed need explicit verification; disabling V6 may cause regressions and a passing scored check alone is not enough. The remaining source and precision audit is still open.

Private check `05a1587da81d4434` passed 9/9 at 1.54086× (95% CI 1.51775–1.56397×). A local check passed 35 nearby shapes, but did not compare performance. V7 adds general token-range routes to improve nearby performance without 973 individually tuned routes.
