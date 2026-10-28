# Gate/up epilogue writes the down-input layout

Frozen v22 baseline; larger dataflow follow-up to moe_xpack. Standalone activation packing lost. This version directly writes each valid BF16 gate/up intermediate into `[route item,K64 chunk,token fragment,lane,8 BF16]` layout. Down's contiguous activation-copy path consumes it directly, with invalid token columns explicitly zero-filled by cp.async predicate. No packing kernel, extra arithmetic, or cross-call cached data is used. The SiLU multiplication and BF16 rounding happen at the same point as baseline. Down output and final norm/weighted sum retain their layouts.

The packed allocation is larger but only valid entries are written/read; all allocations and input-dependent transformations remain inside the timed call. Tradeoff: gate/up epilogue stores may coalesce less well. Test eight fresh M128/M256 inputs, four seeds each, independent reference and bit-equality checks, then 24 alternating cold-L2 pairs.

Diagnostic only; not a submission. Build.py, compile.log, screen.log, ../modal/drv_moe_xfused.py provide evidence.

## Outcome

All eight independent reference checks passed and all outputs were bit-identical. M128 geometric mean paired ratio 1.00147x. M256 geometric mean paired ratio 1.00442x. Not a robust standalone gain: M128 includes a 4% regression, M256 mostly within noise. No official check or promotion. This tests both fresh and in-pipeline transformed activation layout including real store costs; it gives no support for further standalone activation-packing tuning without another reuse benefit. GPU run terminal.
