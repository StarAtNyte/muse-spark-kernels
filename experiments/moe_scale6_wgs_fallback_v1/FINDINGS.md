# Six-bit WGS with scale fallback

This builds on `../moe_scale6_wgs_v1/`: prepare checks all local scales. If all are in `0x50..0x8f`, WGS uses the six-bit packed format; otherwise it packs the original 36 KiB WGS format and selects the original WGS CUDA source for the positive-normal fast path. Scales that cannot use the original PRMT path retain the parent's generic fallback. The recovered uncompressed CUDA source was checked byte-for-byte against the parent.

H200 fallback screen (`probe.log`, app `ap-UBsYDSmizAAriIgRzSc03b`): one scale was changed to positive-normal `0x40` outside the compact range. The uncompressed fallback was selected and both 32- and 128-token outputs passed reference and were bit-identical to the parent on one seed. This does not cover every scale, routing pattern, or small-token path. Official check `54102a958b0640cf` was launched to validate the complete submission and measure its score.
