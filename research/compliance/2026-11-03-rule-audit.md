# Rule audit — 2026-11-03

The organizer clarified that submissions must be readable source, without embedded or loaded prebuilt binaries or compressed, encoded, or obfuscated code. The posted withdrawal deadline for violating submissions has passed. Official [kernel restrictions](https://openkernelchallenge.dev/rules#kernels-restrictions).

Justin Wang also clarified that BF16 projection partial sums must remain FP32 until final output. The shared attention clarification permits cheaper exp2 approximations from FP16-rounded scores and E4M3 probability storage at a different power-of-two scale, conditional on FP32 QK, PV and all partials. It does not establish that FP16 QK accumulation is permitted. Routing nondeterminism is allowed when each call meets tolerances; the quoted conversation did not resolve a numerical threshold for occasional misses or whether the projection clarification extends to MoE.

## Findings

- An AST scan of locally archived scored sources identified 19 completed jobs and one then-running job that decode compressed source. Exact IDs, source paths, hashes, and captured official status are in [the machine-readable inventory](encoded-scored-jobs-2026-11-03.json). This is not necessarily an exhaustive inventory of all historical submissions.
- Current published attention best `6f2b2edf215d4ea3` (2.64939×) and decoder best `b72f02c7eb8745f5` (2.26032×) are among the encoded-source jobs. Their published scores are at risk and are not confirmed eligible. User authorization to withdraw the identified jobs has been requested; no withdrawal is implied by this record.
- Decoder stack `db40bea1008841dc` finished 13/13 at 2.242×, below the prior best, and includes encoded CUDA. Its result is archived, not promoted.
- The decoder rs128 BF16 projection backend uses `cp.reduce.async.bulk.tensor` to add BF16 partials and is selected for some QKV and unpooler shapes. It needs to be disabled or replaced with an FP32 reduction. The projection board has the same backend in its selected dispatcher; actual selected plans require investigation.
- The encoded attention backend contains FP16-accumulator WGMMA instructions. Whether each instruction is used by a scored route, and whether its complete numerical path meets the clarification, requires a separate audit.

## Replacement work

- `experiments/attention_readable_expanded_v1` expands the encoded module into readable source byte-for-byte; it is not cleared for submission because of the precision audit.
- `experiments/attention_readable_fp32_path_v1` removes that backend and uses the existing unencoded routes. Private check `12167249d81c46ee` was queued at last poll. Static inspection found FP32 MMA accumulator instructions but does not by itself prove all paths compliant.
- `experiments/decoder_readable_fp32_reduction_v1` expands encoded CUDA and the escaped single-line helper into readable source and disables the BF16 rs128 backend. It is 895 KB, parses, and still requires a complete precision audit and judge check. Other fast routes are retained pending verification.

The 5-producer attention WST local screen is preserved in `experiments/attention_wst_npw5_v1/modal.log`: measured target cases were 2.4–3.5% slower; its encoded-source candidate is not eligible.
