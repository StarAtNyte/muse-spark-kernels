# MoE dry warm-up v2 (compile-time `dry`, n-tile variants 1 and 2 only): same +21 us, closed

v1's runtime `dry` flag was suspected of costing the wgmma pipeline. v2 makes it a template parameter and instantiates the dry
segment for n-tile counts 1 and 2 only (the real switch is unchanged). Paired result (screen.log): bs=32 0.846-0.853x,
bs=64 0.851x, bs=128 0.840x, bs=16 1.004x; correctness passes. The slowdown is the same ~+21 us at every WGS size as in v1,
so the cause is not the runtime flag or the extra code copies. Something in the first dry stage itself (before the first
real stage can proceed) costs a near-constant ~21 us; not diagnosed. The 32-token cold-code prize (~4-7 us) is not worth a
deeper dig. MoE warm-up line closed for good.
