# Decoder attention global three-block budget

Candidate SHA-256 `a5b5bfaa37d4cac2055e20a350989ec58ec2152b6f3336826f00780a1c1d3c91` changes `OKC_WSMINB` from 2 to 3 globally. CUDA 13 H200 compilation failed: ptxas reported C7602 for the L=8 flat attention kernel, which required at least 66 registers while the global budget limited it to 56. No timing or correctness result was obtained. See `modal.log`. The L=1-only follow-up is `../decoder_attn_wsminb3_l1_v1/`.
