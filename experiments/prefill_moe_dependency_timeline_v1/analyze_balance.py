"""Summarize CTA and phase timing from validated compressed debug traces."""
import base64
import collections
import json
from pathlib import Path
import statistics
import zlib

log = Path(__file__).with_name('modal.log').read_text()
for line in log.splitlines():
    if not line.startswith('TRACE_ZLIB_BASE64 '):
        continue
    trace = json.loads(zlib.decompress(base64.b64decode(line.split(' ', 1)[1])))
    events = trace['events']
    assert len(events) % 96 == 0
    n1 = len(events) * 2 // 3  # phase 1 has 64 output tiles, phase 2 has 32
    by_cta = collections.defaultdict(list)
    for start, end, _, cta in events:
        assert end >= start > 0
        by_cta[cta].append((start, end))
    ends = [max(end for _, end in items) for items in by_cta.values()]
    gaps = [(items[i + 1][0] - items[i][1]) / 1000
            for items in by_cta.values() for i in range(len(items) - 1)]
    assert gaps and all(gap >= 0 for gap in gaps)
    first = min(e[0] for e in events)
    print(json.dumps({
        'seed': trace['seed'], 'ctas': len(by_cta),
        'event_span_us': (max(ends) - first) / 1000,
        'cta_end_spread_us': (max(ends) - min(ends)) / 1000,
        'phase1_last_end_us': (max(e[1] for e in events[:n1]) - first) / 1000,
        'phase2_first_start_us': (min(e[0] for e in events[n1:]) - first) / 1000,
        'inter_item_gap_median_us': statistics.median(gaps),
        'inter_item_gap_p90_us': sorted(gaps)[int(.9 * (len(gaps) - 1))],
        'inter_item_gap_max_us': max(gaps),
    }))
