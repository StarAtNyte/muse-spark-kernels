# Decoder SC6 skinny transfer: rejected

The decoder already had the lossless SC6 pack/unpack implementation, but its M<=16 tiny expert builds use an alternate `moe_helper0` source without SC6 decode support. The first candidate switched those builds to the SC6-aware source and compressed each 16-row expert unit from 36,864 to 35,840 bytes.

## Local H200 results

`modal.log` records the paired whole-layer screen on CUDA 13 / H200. Output checks passed for M<=8. For B=1, q_len=8 sliding, the candidate was 1.0145x, 1.0212x, and 1.0177x over three seeds. Decode-1 was effectively neutral. The M=16 candidate failed all three reference checks (residual relative L2 about 1.01–1.06), so that version is rejected.

The follow-up candidate enables SC6 only for M<=8 and preserves the incumbent source and weights for M=9–16. Official check `6194823f78374e67` passed 13/13, scoring **2.23270x** (95% CI **[2.19921, 2.26619]**). This is below the published **2.26032x** decoder score and does not establish a gain; it is not promoted. The intervals overlap.

The initial failed-mid source and its raw Modal log are retained in `../decoder_sc6_skinny_v1/`; the restricted source, exact check profile, and status are retained in this directory. No leaderboard submission was made.
