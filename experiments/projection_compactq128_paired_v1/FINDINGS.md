# Compact QKV m=128 paired screen

The first screen failed at import: v2's transferred annotations referenced a missing module-level `torch` import (`modal_import_failure.log`). The corrected v3 adds it and keeps fresh-thread context binding. The paired H200 screen then prepared only the qkv m=128 parent backend and the compact kernel, checking both against the reference on each seed and timing 20 alternating cold-L2 graph replays.

| Seed | Parent µs | Compact µs | Speedup |
|---:|---:|---:|---:|
| 61 | 19.552 | 18.016 | 1.0853× |
| 62 | 19.696 | 17.984 | 1.0952× |
| 63 | 19.520 | 17.968 | 1.0864× |

All six outputs passed. Event timings include harness overhead and cover one shape; the full wrapper, other shapes, and the remaining submission precision audit are not established by this component screen. V3 private check `d96792184a114eef` is pending. V1 and v2 cannot be used: both lack the module-level import.
