# Stack alygg77 projection routes onto our file for shapes where they won a paired judge check

Paired official checks (alygg 35499d59cf9840a8 vs ours 279eb4d73147494a) showed alygg77's file faster on decode qkv_gate (m=1/16/32: 8.8/9.5/10.8 vs 9.4/10.2/12.0 us), decode unpooler (m=1/32) and verify-256 router. Route exactly those shapes to an embedded copy of their module (Apache-2.0, attribution kept), everything else stays on our routes.
