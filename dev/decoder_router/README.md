# FP32 small-router adaptation

Adapted from Nathen Fernandes' MIT-licensed ROUTER_SCALAR, preserving the
copyright in kernel.cu. Both input activations and weights remain FP32, unlike
the BF16-input projection interface. No quantization or reduced-precision
matrix operand is introduced. Outputs accumulate FP32 FMAs.

screen.log: all 34 comparisons pass versus FP64 references on two seeds at
M1/2/4/8. At M8, selected NT256/ST1 gives ~13.3 µs versus ~27.9 µs for Torch
F.linear including replay cost; the computation-only difference is larger.
At M1, NT512/ST2 wins consistently. The 512-thread M8/ST2 configuration is
not selected because of register pressure. The integration retains the real
router-side stream, bias/top8 logic and join; its effect must be measured at
whole-decoder scope. Full official job f0a4978fa68d4929 is pending.
