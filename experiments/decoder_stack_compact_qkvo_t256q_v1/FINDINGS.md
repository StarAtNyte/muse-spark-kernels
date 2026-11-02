# T=256 compact QKV route stacked into the decoder: reject

Date: 2026-11-02. Candidate SHA-256:
`bc47f0e1dc9cfd1efce2c29e018c7bcd237f7eecc66cbf2115678cbf087d58d4`.
The paired parent is the T=128 compact-QKV/O candidate
`decoder_stack_compact_qkvo_v1` (SHA-256
`aa354bd2472393cfb407a50260dbc25543dab3763c4b45bcffca4f450ca98b6e`).

The T=256 compact QKV kernel configuration already existed in the helper but
was disabled in its route table. This experiment enabled config index 2 for
T=256 while leaving the T=128 QKV and O paths unchanged. It tests whether the
route transfers to verify-32 in the integrated decoder.

## Paired judge-image H200 screen

Three seeds per mode, 24 alternating graph replays per source and seed. All
six outputs passed the challenge reference check.

| Mode | Seed | Parent µs | Candidate µs | Parent/candidate |
|---|---:|---:|---:|---:|
| Full | 5001 | 484.7 | 485.6 | 0.9982× |
| Full | 5002 | 477.4 | 478.3 | 0.9981× |
| Full | 5003 | 466.7 | 469.9 | 0.9932× |
| Sliding | 5001 | 305.9 | 309.7 | 0.9880× |
| Sliding | 5002 | 314.1 | 313.6 | 1.0017× |
| Sliding | 5003 | 304.3 | 305.3 | 0.9968× |

Geometric gain was **0.99599× overall**, **0.99650× full**, and **0.99548×
sliding**. Reject the T=256 route. The T=128 stack remains unchanged; its small
screened gain is independent evidence and does not imply this configuration
will help other token counts.

Modal run: https://modal.com/apps/baptinjean20/main/ap-CJxyYbHucAw7DBt1MzqnZB
Raw paired rows and launch details are preserved in `paired_screen.json` and
`pair_driver.py`. No official quota was spent.
