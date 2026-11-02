# Fold softmax scaling into attention RMS

Hypothesis: the attention output accumulators `u` are not rounded before RMS.
The current `pr8()` helper returns its input unchanged. For `v = vscale*u/l`,

    RMS(v) = u / sqrt(mean(u^2) + eps*(l/vscale)^2)

so compute the RMS factor from the unnormalized accumulator and remove the
per-element `vscale/l` multiplies in all five finalizer paths. Empty rows keep
zero output. The change preserves the real-valued result; floating-point
rounding differs slightly and must pass the task reference.

Candidate: `candidate.py`. It is compared against the incumbent on all 12
scored shapes with reference checks and alternating cold-L2 CUDA graph replays
using `dev/modal/drv_attn_pair.py` on the public H200 image. This screen is not
an official check or leaderboard score.

## Result

Modal H200 app `ap-ihK55SN48b5vuDncg9ckHt` completed the paired driver with
return code 0. Both parent and candidate passed the reference on all 12
shapes. Candidate/parent median ratios ranged from 0.997 to 1.009; their
geometric mean was 1.00275. The largest gains were 0.9% on verify-16 sliding
and 0.6% on decode-16 sliding, while verify-16 full was 0.3% slower. This is
too small and uneven to promote or spend an official check on. Keep the
incumbent. Raw rows are in `modal.log`.

The Modal CLI command used `--write-result`, which is incompatible with this
void local entrypoint and made the CLI exit 1 after the remote function had
completed successfully. The complete remote stdout is preserved in `modal.log`;
future runs should omit that flag or add a return value.
