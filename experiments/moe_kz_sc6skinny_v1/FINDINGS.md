# SC6_SKINNY=True on the keell0renz MoE: negative

Check 80970e6597a74a8f passed 5/5 at 2.384x vs 2.405x for the unchanged file (98be7cf25e0c494f) in the same window.
Skinny cases did not improve: decode 1 3.126 -> 3.097, decode 16 2.060 -> 2.000, verify 8 2.276 -> 2.247 (registers 146-151, no
spills). Same conclusion as the file's default of SC6_SKINNY = False. Not submitted.
