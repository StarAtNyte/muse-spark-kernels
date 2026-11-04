# attn_b1_v3: new compact B=1 attention kernel (rejected)

Design: one CTA per 64-key tile (32/33 CTAs), page indices loaded into registers and the gathers issued on arrival; q norm, RoPE and the FP8 write overlap the gather. QK/PV use FP16 mma.sync with FP32 accumulation. Merge without atomics: FP32 partials with a NaN sentinel per word, polled by the finalizing CTA.
Predicted vs measured, decode-1 sliding (us from first CTA): indices 0.7/0.7, tile landed 1.35/1.44, partials stored 1.9/2.19, partials seen 2.6/4.62, end 2.9/5.2 (parent 4.1).
Failure: cross-SM polling takes 2.3-3.6 us to observe a store, although the data is visible after about 0.7 us. The parent's atomics + fence + ticket (~1.8 us) beat every polling variant. Per-tile mma.sync compute is 2-4x slower than the parent's wgmma (q_len 8: 3.4 vs 0.83 us).
Paired vs parent: best (round 9) decode-1 sliding 0.83-0.86, verify-1 sliding 0.68-0.69; full layers 0.75/0.54 in v1.
Lesson: on H200 under load, never merge by polling; atomics + fence + last-CTA ticket is the fastest cross-SM merge measured. The parent B=1 path is within ~1.2 us of its merge-design floor.
