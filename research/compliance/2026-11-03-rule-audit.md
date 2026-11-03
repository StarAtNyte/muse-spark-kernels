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
- `experiments/attention_readable_fp32_path_v1` removes that backend and uses the existing unencoded routes. Private check `12167249d81c46ee` passed at 2.60414× (95% CI 2.56508–2.64320×). Static inspection found FP32 MMA accumulator instructions but does not by itself prove all paths compliant.
- `experiments/decoder_readable_fp32_reduction_v1` expands encoded CUDA and the escaped single-line helper into readable source and disables the BF16 rs128 backend. It is 895 KB, parses, and private check `bbe34f0cbe494c88` passed 13/13 at 2.23574× (95% CI 2.20220–2.26928×). It still requires a complete precision audit. Other fast routes are retained pending verification.

The 5-producer attention WST local screen is preserved in `experiments/attention_wst_npw5_v1/modal.log`: measured target cases were 2.4–3.5% slower; its encoded-source candidate is not eligible.

## Decoder projection follow-up

The readable decoder candidate's embedded `bfp_prepare` allocates both skinny and GEMM split-K workspaces as `torch.float32`; the respective CUDA kernel entry points declare these workspaces as `float*`. The Triton `dense` split-K path allocates FP32 partials, and the paired compact QKV route uses `float4` cluster exchange and sums before its final BF16 store. These observations support those particular paths but are not a proof covering every epilogue, library dispatch, router operation or dynamic route. The removed rs128 backend was a separate path. The official decoder check passed; the static inspection does not establish full compliance.

The standalone projection candidate `experiments/projection_fp32_splits_v1` additionally removes the forced unpooler BF16 atomic split reduction, which the earlier rs128-only mitigation did not cover. Its private check is pending and the rest of its backends remain under audit.

Additional decoder GQ inspection: the cluster-GEMM path copies peer accumulators as `float`/`float4`, adds them to FP32 `acc`, and only then stages or writes BF16. This supports the split reduction in this specific path; it does not establish eligibility of the entire decoder. In the projection candidate, the explicit `1 << 26` atomic-add entries remaining in the older GVC table are router-only; router results use FP32 output. This is static dispatch evidence, not a substitute for end-to-end testing or a complete backend audit.

## Projection backend follow-up

For the v3 projection candidate, the `nb` output-projection CUDA path accumulates into FP32 registers over K and converts to BF16 only after the WGMMA loop. The `nb` router path combines its high/low BF16 components into `float4` partials and uses FP32 cluster transport; router output is FP32. The pinned `ag` cuBLASLt wrapper specifies compute enum 68 (NVIDIA documents this as `CUBLAS_COMPUTE_32F`) and sets split-K to 1; the [cuBLAS compute-mode documentation](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t) describes this mode as using at least 32-bit intermediate precision. These are targeted code findings, not proof that every dynamic fallback satisfies the rules. In particular, library fallback and dynamically selected routes still need tracing; do not infer their split-reduction behavior solely from output dtype. The v3 compact QKV private check is still pending.

PyTorch documents that BF16 GEMM reduced-precision reductions may be enabled by default ([numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms)). This is a risk for BF16-output `F.linear` and `mm` fallbacks, not evidence that a specific observed run used a reduced-precision reduction. `projection_fp32_compactq128_safe_v4` explicitly requests FP32 results for the identified GPU fallback calls before casting to BF16; check `6dea36df13e74a6a` is pending. The CPU-only fallback and other custom/library paths remain separate audit items.

The readable decoder also has two BF16-output PyTorch projection paths (`bfp__run_torch` and the direct attention QKV `torch.mm`). As in the projection board, the default PyTorch setting permits reduced-precision BF16 reductions; the particular algorithms selected have not been established. Candidate `decoder_readable_fp32_fallback_v2` requests FP32 matmul output and converts only at the end for both paths. Private check `5efc65e1619a483c` is pending. This is not a completed decoder precision audit.

Follow-up: decoder fallback V2 missed a third BF16-output PyTorch call, the small-token pooler `F.linear` in `_pool_router_fp32`. `decoder_readable_fp32_fallback_v3` changes it to an FP32-output matmul followed by the final cast; its private check is `f08d0cc2e617467e` and is pending. An AST scan of V3's remaining Python matrix-multiply operators found only FP64 reference/preparation calculations, but this does not cover CUDA templates or library internals.

Decoder MoE follow-up: the main non-B4 WGS path of decoder V3 stores completed per-expert values in BF16 `ybuf` before final normalization and route combination. The organizer thread quoted by the user did not answer whether the MoE precision restriction extends to this case, so V3 has an unresolved interpretation issue. `decoder_readable_fp32_ybuf_v4` uses the existing FP32 `ybuf` path instead; its private check is pending. This is a conservative change, not a claim that organizers have prohibited the original completed expert values.
