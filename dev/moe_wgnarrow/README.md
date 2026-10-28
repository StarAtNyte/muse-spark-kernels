# Narrow-token Hopper WGMMA MoE

The existing warpgroup prototype only instantiated N=32/64/128, despite many
experts having <=8 tokens. Production instead uses warp MMA. This experiment
adds legal WGMMA BF16 N=8 and N=16 operators, retaining N=32 for larger groups.
NVIDIA's supported shapes include (64,8,16) and (64,16,16):
https://github.com/NVIDIA/cutlass/blob/main/python/cutlass_library/sm90_shapes.py

Each warpgroup owns 64 weight rows per gate/up branch. Four warps decode their
original lossless NVFP4 fragments into BF16 registers and cooperatively issue
WGMMA. Activations stay BF16, accumulation FP32. Sparse token groups have four
accumulators per thread per branch instead of the old N32 minimum of sixteen.

The N8 shared-memory stage needs only 64 sixteen-byte granules. The original
NCH/2 loop would execute zero iterations at NCH=1; this implementation rounds
up and guards id<N*8 so inactive threads cannot overwrite the next stage.
The NMAX guards also remove unreachable wider paths for small shape buckets.

A timed activation permutation matches the existing packed weight operand
order. Gate/up writes intermediates in that same permutation, so down GEMM
consumes them directly; its final output is in natural order. This permutation
is part of the computation timed, never cached from inputs.

CPU `compile.log` proves the new narrow PTX instructions assemble with no
spills in the first version. A later NMAX guard removes unreachable wider
paths. `screen.log` validates end-to-end task correctness before comparing
paired cold-L2 CUDA graphs at M1/8/16/32 on two seeds and three stage counts.
The dispatcher is diagnostic only: it prunes unused fallback instantiations.
A winner must be integrated with complete production source and pass the
full official checker before promotion.

## Outcomes

`screen.log`: all 24 task checks passed; the narrow instructions and operand
permutation are functionally valid. Best stage-4 end-to-end ratios were about
0.49/0.44× at M1, 0.63/0.63× at M8, 0.67/0.65× at M16, and 0.88/0.83× at M32.
The sequential implementation is rejected; no official quota used.

`make_pipeline.py` leaves one WGMMA group in flight, uses two named register
buffers, and keeps two free shared stages. Refill changes from i+S-1 to i+S-2:
the former can overwrite the previous group's B operand while it is still in
flight. Wait-group 1 retires group i-2 before that stage or register buffer is
reused. The final wait-group 0 precedes epilogue reads. Compiler-fence operands
keep A-buffer lifetimes explicit. The CPU compile passed without spills; PTXAS
reported injected warpgroup-arrive instructions, whose performance effect has
not been isolated.

`pipeline_screen.log`: all 24 task checks passed. Stage-4 ratios were about
0.48/0.43× at M1, 0.63/0.63× at M8, 0.66/0.65× at M16, and 0.87/0.81× at M32.
No material gain over the sequential version; still substantially slower than
production. Reject without an official check. Narrow WGMMA availability alone
does not imply a faster small routed-expert kernel. Do not repeat stage-count
searches without a new bottleneck measurement or substantive dataflow change.
