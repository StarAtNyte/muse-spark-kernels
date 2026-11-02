# KPERM3 + SC6 WGS MoE stack, v7

V7 keeps v6's corrected timed dispatch to the validated `-DOKC_SC6W=1` WGS module and removes the optional `okc_attrs()` diagnostic from both `load_inline`'s Python exports and the prepare-time print path. This resolves the C++ binding failure seen in v5 and inherited by v6 without changing timed kernel math.

Source SHA-256: `db8dcb2a5438178a52fd09a3e888942f1c54cc056a7be4146ae1ea64dd2b6f24`.

Official check `1b045ada84a34097` failed during prepare: two remaining diagnostic loops called `_ext(d).okc_attrs()` even though that optional function was not exported. It did not reach correctness or timing. V8 removes the remaining calls; its check is `a5bdd94d9dc7460b`.
