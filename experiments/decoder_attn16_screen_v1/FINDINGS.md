# Decoder attention plan for 16-request sliding decode (8 splits, no coop)

Paired screen (`dev/modal/drv_decoder_pair_d16.py`, CUDA 13 H200, reference-checked, 3 seeds): decode 16 sliding whole layer 1.067-1.074x (165.4/171.5/164.9 -> 154.1/160.8/154.6 us); controls decode 32 sliding 1.001-1.004x, decode 1 sliding 0.995-1.006x. Candidate `candidate.py` (_TUNED7 entry).
