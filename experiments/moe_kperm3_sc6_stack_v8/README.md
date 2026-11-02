# KPERM3 + SC6 WGS MoE stack, v8

V8 retains v7's dispatch to the lossless SC6 WGS module and removes every
remaining call to the optional `okc_attrs()` diagnostic, including calls in
prepare-time and disabled MFAB diagnostics. V7's first H200 check
(`1b045ada84a34097`) failed during setup because a prepare-time call remained;
it never reached correctness or timing.

The source changes are diagnostic-only plus the first-line problem marker.
Source SHA-256: `cef83f0575923da2708f227915fd478e1abaf8e1564fc2d2873bf6a6fdbd58ae`.

Official check `a5bdd94d9dc7460b` has been accepted. Performance and correctness
results are pending. This candidate must be verified to dispatch the
`-DOKC_SC6W=1` module on the timed WGS path before its timing can be treated as
SC6 evidence.
