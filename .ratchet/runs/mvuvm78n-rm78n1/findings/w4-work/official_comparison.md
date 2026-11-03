Separate official runs; timings do not isolate the source change.

| Case | Incumbent us | Candidate us |
|---|---:|---:|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode 1 page=1 | 68.11 | 69.22 |
| batch=1 ctx=80000 dist=soft layer=full load=decode 1 page=1 | 74.58 | 75.54 |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode 16 page=1 spread=agentic | 127.79 | 130.40 |
| batch=16 ctx=80000 dist=soft layer=full load=decode 16 page=1 spread=agentic | 206.25 | 207.80 |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode 32 page=1 spread=agentic | 148.24 | 151.47 |
| batch=32 ctx=80000 dist=soft layer=full load=decode 32 page=1 spread=agentic | 292.06 | 304.00 |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify 1 page=1 q_len=8 | 107.26 | 110.89 |
| batch=1 ctx=80000 dist=soft layer=full load=verify 1 page=1 q_len=8 | 114.39 | 117.81 |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify 16 page=1 q_len=8 spread=agentic | 237.47 | 239.06 |
| batch=16 ctx=80000 dist=soft layer=full load=verify 16 page=1 q_len=8 spread=agentic | 324.60 | 321.55 |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify 32 page=1 q_len=8 spread=agentic | 311.13 | 303.35 |
| batch=32 ctx=80000 dist=soft layer=full load=verify 32 page=1 q_len=8 spread=agentic | 470.73 | 472.63 |
