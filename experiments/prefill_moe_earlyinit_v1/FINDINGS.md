# Prefill MoE early shared-memory initialization

Set `MOE_EARLY_INIT=1` to initialize shared-memory barriers before the PDL dependency wait. Parent is published 1.80923× source SHA-256 `f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`; candidate SHA-256 `3f05e9e7724ffdef612aca0fd2c810581ecd3642c4ca212d07f010669ef31452` differs only in the macro. Driver shares prepared state and plans, changes the macro in the candidate module and has an inherited docstring from an older ablation.

H200 public-image app `ap-11oE1bW5KOEhtn2WvPajdE` completed with code 0; both variants passed all three reference screens. Parent/candidate median times over 16 alternating cold-L2 replays were 514.4/514.0 µs (turn-1 sliding), 3100.0/3098.8 µs (turn-4 full), and 9325.2/9328.4 µs (first-turn sliding). Differences below 0.1% establish no gain in this one-seed screen. SGLang was stubbed, attention tuning failed and attention was pinned. Do not promote or submit. Raw output is `screen.log`.
