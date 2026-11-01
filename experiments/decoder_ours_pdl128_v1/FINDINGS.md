# PDL_MAX_T 32 -> 128 on our decoder: negative

Paired private checks: ours unchanged 2.226x (job 4a963397dbc840f9), with PDL_MAX_T=128 2.191x (45e72870e6f643ed),
keell0renz file 2.264x (c1b314a3eeaa4d36). verify-16 got worse (sliding 2.163 -> 2.096, full 2.095 -> 2.075), decode-32
full 2.145 -> 2.012. The constant does not explain their verify-16 lead. Other top-level differences left to test:
RQ_USE = {('un',128),('qkv',256),('qkv',128)}, SC6_WGS (needs the SC6 merge hunks), MFAB_LAYER / PCX_CFG per-layer tuning.
