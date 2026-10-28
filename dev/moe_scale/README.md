# Lossless scale decode on H200

Production stores eight FP8-derived scale bytes per lane: four K chunks for
each of two rows. The decoded BF16 pair is exactly
`((((q >> (8*c)) & 255) << 4) + 0x7600) * 0x10001`.

Two alternatives are screened against a frozen production snapshot:

* `PRMT`: retain the original scale layout. A generic PTX byte permute extracts
  the selected byte into both low halfword bytes, then shift/add constructs the
  same BF16 pair. The integer identity passes all 256 bytes × 4 selectors.
* `EXPANDED`: store the exact decoded BF16 scale bits during weight preparation.
  Warp split-K kernels load a uint4 instead of uint2 and use `prmt.b32.rc16` to
  duplicate the selected halfword. This doubles scale bytes (about 11% more
  total packed weight+scale bytes), but reduces inner-loop scale conversion.
  Row-split kernels retain the old scales and are not part of this ablation.

The four FP4 bits per weight, their exact values, scale values, BF16 activations,
FP32 accumulation and output rounding remain unchanged. Expanded metadata is a
lossless representation of existing scales, not quantization of weights.

`screen.log` failed before compilation because Triton JIT functions cannot be
loaded from an anonymous exec string. `screen_v2.log` loads the frozen parent
from a physical Python file. A diagnostic macro parameter was also renamed to
avoid substituting the warp-count integer into the weight-pointer argument.
No official quota was spent on that setup failure.

The screen checks bitwise equality with production, task reference accuracy,
and alternating paired cold-L2 CUDA graphs. Full official source and coverage
are required before promotion. The diagnostic dispatcher instantiates only the
production configurations used by the scored shapes; it is not a submission.

The second screen compiled base and PRMT but the expanded helper split a
template declaration from its function, so no numerical results were produced.
The repaired source passed CPU-only NVCC and PTXAS in `compile_expanded.log`.
Expanded batch-16/32 kernels spill under the original four-block launch bound;
`screen_v3.log` includes a two-block launch-bound variant to test the occupancy
versus register-capacity tradeoff. All tests still require bit-identical output.

## Result

`screen_v3.log` passed all 28 end-to-end task checks with bit-identical outputs.
PRMT geometric mean was 0.9852× across the tested shape/seed comparisons.
Expanded scales were 0.9334×; the relaxed bound was 0.8787×. Reject all three;
no official check or submission. Increased scale traffic and register capacity
cost more than the avoided conversions in these implementations. This result
rejects the tested representations, not every possible scale layout.
