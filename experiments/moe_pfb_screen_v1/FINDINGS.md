# Phase-B L2 prefetch in TINY/TINY2 MoE (2026-10-29) — rejected

Hypothesis: at bs 1 (2-3 experts) the kernel is ~10 us above the contiguous-stream floor because phase B restarts the
DRAM stream after phase A; prefetching every touched expert's down weights into L2 at launch would hide it.
Candidate `candidate.py` (`OKC_PFB`), driver `dev/modal/drv_moe_pfb.py`, raw `modal.log`. All variants passed the
reference check. Paired cold-L2 medians, 30 alternating replays:

| bs | base us | PFB=1 | PFB=2 (evict_last) | PFB=2 + L2H=1 |
|---:|---:|---:|---:|---:|
| 1 (2 experts) | 30.4-30.5 | 0.93x | 0.90x | 0.96x |
| 1 (3 experts) | 38.1-38.9 | 0.89-0.92x | 0.85-0.88x | 0.91-0.93x |
| 8 (6-10 experts) | 58-87 | 0.93-0.95x | 0.91-0.93x | 0.91-0.94x |

Every variant is slower. The phase-B restart is not the bottleneck; the extra prefetch traffic competes with phase A.
