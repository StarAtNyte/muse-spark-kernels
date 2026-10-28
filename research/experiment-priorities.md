Effect of reducing ONE shape latency by 20%; gains refer to its own problem score.
| Problem | Load | Layer | Current us | Load share | Track gain | Optimistic floor gain |
|---|---|---|---:|---:|---:|---:|
| muse_moe_w4a16 | decode 1 | experts | 31.16 | 100.0% | 3.79% | 18.22% |
| muse_moe_w4a16 | decode 16 | experts | 103.19 | 100.0% | 3.79% | 11.91% |
| muse_moe_w4a16 | decode 32 | experts | 151.85 | 100.0% | 3.79% | 13.57% |
| muse_moe_w4a16 | verify 1 | experts | 76.65 | 100.0% | 3.79% | 12.74% |
| muse_moe_w4a16 | verify 16 | experts | 228.94 | 100.0% | 3.79% | 14.55% |
| muse_moe_w4a16 | verify 32 | experts | 287.40 | 100.0% | 3.79% | 17.55% |
| muse_attention_decode | decode 32 | full | 183.14 | 81.0% | 2.99% | 4.94% |
| muse_attention_decode | verify 32 | full | 337.57 | 80.6% | 2.97% | 12.38% |
| muse_attention_decode | decode 16 | full | 96.81 | 73.9% | 2.70% | 5.07% |
| muse_decode_layer | decode 1 | sliding | 108.95 | 73.2% | 2.67% | 10.91% |
| muse_attention_decode | verify 16 | full | 172.67 | 72.8% | 2.66% | 10.85% |
| muse_decode_layer | verify 1 | sliding | 170.93 | 72.7% | 2.65% | 10.88% |
| muse_decode_layer | decode 16 | sliding | 172.91 | 66.6% | 2.41% | 8.01% |
| muse_decode_layer | verify 16 | sliding | 327.84 | 65.1% | 2.35% | 8.87% |
| muse_attention_decode | verify 1 | sliding | 13.48 | 63.4% | 2.29% | 17.93% |
| muse_decode_layer | verify 32 | sliding | 432.98 | 61.5% | 2.21% | 9.22% |
