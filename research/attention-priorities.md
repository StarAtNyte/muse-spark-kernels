Effect of reducing ONE shape latency by 20%; gains refer to its own problem score.
| Problem | Load | Layer | Current us | Load share | Track gain | Optimistic floor gain |
|---|---|---|---:|---:|---:|---:|
| muse_attention_decode | decode 32 | full | 183.14 | 81.0% | 2.99% | 4.94% |
| muse_attention_decode | verify 32 | full | 337.57 | 80.6% | 2.97% | 12.38% |
| muse_attention_decode | decode 16 | full | 96.81 | 73.9% | 2.70% | 5.07% |
| muse_attention_decode | verify 16 | full | 172.67 | 72.8% | 2.66% | 10.85% |
| muse_attention_decode | verify 1 | sliding | 13.48 | 63.4% | 2.29% | 17.93% |
| muse_attention_decode | decode 1 | sliding | 7.60 | 57.6% | 2.06% | 14.99% |
| muse_attention_decode | decode 1 | full | 16.08 | 42.4% | 1.49% | 6.42% |
| muse_attention_decode | verify 1 | full | 22.37 | 36.6% | 1.27% | 6.02% |
| muse_attention_decode | verify 16 | sliding | 22.40 | 27.2% | 0.94% | 4.89% |
| muse_attention_decode | decode 16 | sliding | 11.89 | 26.1% | 0.90% | 4.34% |
| muse_attention_decode | verify 32 | sliding | 28.18 | 19.4% | 0.66% | 3.11% |
| muse_attention_decode | decode 32 | sliding | 14.96 | 19.0% | 0.65% | 2.71% |
