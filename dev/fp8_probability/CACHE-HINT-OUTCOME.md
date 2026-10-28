# Cache hint / head scheduling screen

Sixteen independent task checks passed and every output was identical to the
current production baseline. Across two fresh seeds and batches16/32:

- L2 prefetch64B: geometric speedup0.99998x.
- L2 prefetch128B:1.00433x.
- Interleave the two KV heads in CTA ordering:1.00033x.
- Interleaving plus128B hint:1.00845x.

The hypothesis used Muse's two64-byte KV heads per128-byte physical row.
The tested hints/scheduling changes did not establish a substantial gain.
No additional official check or public submission is justified by this screen.
Every cache hint is inside the timed attention loop; arithmetic is unchanged.
