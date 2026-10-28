# Projection route plan experiments — 2026-10-27

See [the cuBLASLt screen notes](../dev/cublaslt/README.md) for per-variant timings and decisions. The latest live board snapshot is `current-overall.json`; problem limits are `current-problems.json`.

The best isolated alternative was the recorded M128 verify unpooler plan, but two independent full checks scored 1.530x and 1.510x. The M128 pooler + unpooler stack scored 1.524x and 1.519x. These results do not establish a robust improvement over public 1.5177x and remain below the live top-50 cutoff 1.5331x. No public submission was made.
