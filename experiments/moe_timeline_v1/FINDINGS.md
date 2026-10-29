# MoE per-CTA phase timeline (2026-10-29)

Driver `dev/modal/drv_moe_timeline.py` (guards the source's unconditional `#define OKC_DEBUG 0` in-driver; the
submission is unchanged), raw `modal.log`. globaltimer stamps per CTA, min/median/max in us from the first start.

| bs | experts | routed | A done | B done | barrier | final |
|---:|---:|---|---|---|---|---|
| 1 | 2 | 2.3 | 12.5 / 15.4 / 18.2 | 22.5 / 24.1 / 24.3 | 25.3 | 25.3-25.9 |
| 1 | 3 | 2.3 | 18.4 / 20.7 / 23.3 | 30.2 / 31.5 / 32.0 | 32.8 | 33.0-33.5 |
| 8 | 9 | 2.3 | 44.8 / 47.9 / 49.7 | 70.7 / 71.9 / 72.7 | 73.7 | 74.0-74.5 |
| 16 | 13 | 2.8 | 62.7 / 65.5 / 69.6 | 100.4 / 101.6 / 103.2 | 104.2 | 104.4-105.0 |
| 32 | 15 | 2.0 | 78.3 / 82.4 / 86.5 | 116.2 / 120.1 / 124.9 | 125.7 | 126.0-129.5 |

- bs 1: phase A streams ~2.75 TB/s with a 5.7 us CTA spread; phase B cannot start for an expert before its slowest
  phase-A CTA, then streams ~3.1 TB/s. Overheads: routing 2.3 us (partly overlapped by FAST1 preissue), A tail ~3 us,
  barrier + final ~1.5 us.
- bs 16/32: phases stream 3.2-3.6 TB/s; in-kernel time is 8-17% above the contiguous-TMA floor
  (`experiments/h200_read_methods_v1`). Remaining MoE headroom is the A tail, the A->B wait and the end barrier.
