# KPERM3 + SC6 WGS MoE stack, v8

V8 retains v7's dispatch to the lossless SC6 WGS module and removes every
remaining call to the optional `okc_attrs()` diagnostic, including calls in
prepare-time and disabled MFAB diagnostics. V7's first H200 check
(`1b045ada84a34097`) failed during setup because a prepare-time call remained;
it never reached correctness or timing.

The source changes are diagnostic-only plus the first-line problem marker.
Source SHA-256: `cef83f0575923da2708f227915fd478e1abaf8e1564fc2d2873bf6a6fdbd58ae`.

Official check `a5bdd94d9dc7460b` completed and failed on correctness setup
before timing. The bs=4 case passed, but bs=33, 257, and skewed bs=128 raised
`MFIN needs the flag region in scratch[0]`. The candidate's `_make_scratch()`
allocated only the pre-MFIN region (`8 + 1024 + EMAX` int32s), while its WGS
launcher requires an additional `MF_INTS=4096` int32 flag area. This is an
allocation/packaging defect, not an SC6 numerical or performance result. See
`profiles/submission.a5bdd94d9dc7460b.log`.

The follow-up v9 restores the incumbent scratch allocation and DF sentinel;
it is archived separately so this failed source remains byte-for-byte intact.
