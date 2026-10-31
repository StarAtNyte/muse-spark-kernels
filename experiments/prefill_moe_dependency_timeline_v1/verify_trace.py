"""Validate and summarize compressed traces emitted by the timeline driver."""
import base64
import json
from pathlib import Path
import zlib

log = Path(__file__).with_name('modal.log').read_text()
traces = []
for line in log.splitlines():
    if line.startswith('TRACE_ZLIB_BASE64 '):
        trace = json.loads(zlib.decompress(base64.b64decode(line.split(' ', 1)[1])))
        events, waits = trace['events'], trace['phase2_wait_ns']
        assert len(events) % 96 == 0 and len(waits) == len(events) // 3
        assert all(start > 0 and end >= start for start, end, _, _ in events)
        assert all(wait >= 0 for wait in waits)
        traces.append(trace['seed'])
        print(f"seed={trace['seed']} events={len(events)} phase2_waits={len(waits)} max_wait_ns={max(waits)}")
assert traces == [9400, 9407, 9417], traces
print('PASS: all three compressed traces decode and satisfy structural checks')
