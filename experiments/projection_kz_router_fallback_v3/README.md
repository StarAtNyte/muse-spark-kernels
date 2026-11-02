# KZ projection route assembly repair, v3

The v2 candidate retained the complete 227,420-character base64/zlib payload, but its generated wrapper omitted the `exec(compile(_zsrc(...)))` call and concatenated the route set onto the closing quote. The official run `a7bd51ec7e13491a` therefore failed at Python parse time before `prepare()` or timing. This candidate restores the missing wrapper and preserves the route set and embedded kernel bytes exactly.

The decoded embedded module and complete candidate both pass Python AST parsing. No kernel logic was changed. Candidate SHA-256: `f7841d83a6cc2e0248f89684a04102efa03add0e85ad08464218041bed3746e3`. Embedded module SHA-256: `bfa850cda31704b9ba088b482ceea8e3c902f7b4b97b4e25b6c360767369dd98`.

Official check `91f71fd3e4d9491a` has been received by the judge. Await its correctness and timing result before promotion.

Official check `91f71fd3e4d9491a` parsed and compiled the module, then failed during eager `prepare()` when the embedded KZ helper attempted its M=128 output-projection warm launch (`olayer launch failed: 1`). It produced no score. V4 disables that unscored warm path and excludes the M=32 output-projection route; other KZ routes remain selected.
