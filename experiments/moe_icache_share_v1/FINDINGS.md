# Code-fetch share of the keell0renz MoE under the judge's cold L2 (2026-11-01)

Kushagrabainsla's `tools/icache.py` method on Modal H200 (driver.py, screen.log): cold L2 (code + data) vs warm code + cold
data (same kernels just ran on other weights/inputs), 20 graph replays, median of 7 x 3.

| bs | cold us | warm code us | code share |
|---:|---:|---:|---:|
| 1 | 29.4 | 29.1 | 1.0% |
| 8 | 61.1 | 60.5 | 1.0% |
| 16 | 58.1 | 57.2 | 1.6% |
| 32 | 100.0 | 93.0 | 7.0% |
| 128 | 174.9 | 174.9 | 0.0% |
| 256 | 217.0 | 222.5 | -2.5% |

Only bs=32 (WGS build, our weakest MoE load) shows a real code cost. Full recovery = ~+1.2% on the MoE geomean, ~+0.6% on the
decoder: below both cutoffs. Not built: it needs a no-op mode through the 6,800-line warpgroup-specialized kernel.
