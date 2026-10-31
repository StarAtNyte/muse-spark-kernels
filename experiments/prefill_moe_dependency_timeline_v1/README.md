# Prefill MoE phase-dependency timeline

Short-prefill NCU counters report substantial wait and barrier categories but
do not identify the contributing instruction. The fused MoE producer waits
for all first-GEMM output tiles needed by a second-GEMM item. This diagnostic
uses the published best source and its existing item trace, plus experimental
timestamps around that producer wait, to measure the wait distribution for
753-token B=1 sliding prefill on three seeds. It validates the layer output
against the reference before reporting, and preserves a compressed raw trace
in `modal.log`. The added timing instructions perturb scheduling and this is
not a score benchmark or proof of a critical path.

The initially launched driver had an incorrect assertion: it checked the
unused event area for zeros even though the producer-wait trace intentionally
occupies part of that area after the valid events. The checked-in driver fixes
that assertion; if the first run fails there, retain its log and rerun the
corrected driver. The public H200 image and 300-second runner limit apply.

The first run `ap-L90buMFShMgCVVVOOP5raV` passed the seed-9400 reference
check, then hit that reader assertion with 1344 recorded items. Its failure
is preserved in `initial_failure.log`; it yielded no analyzed timing. A run
with the corrected reader completed as `ap-mXBTP92C5UXnIQ7olmMRd0`; raw output
is in `modal.log`.

All three reference checks passed. For seeds 9400, 9407 and 9417, respectively,
the trace contained 448, 512 and 512 second-phase items. Median measured
producer dependency waits were 288, 256 and 288 ns; the 90th percentile was
384 ns for each, and maxima were 704, 800 and 512 ns. None exceeded 1 µs.
The interval includes timestamp and polling overhead, and instrumentation can
change scheduling. This is evidence against a large per-item dependency wait
in these three inputs, not a proof that all queue waiting or barrier stalls
are negligible. The recorded event spans were 220–232 µs.

The raw per-item records and wait arrays are preserved as compressed base64
in the log. `verify_trace.py` successfully decoded all three and checked item
counts and basic timestamp consistency. Focus next on other synchronization,
producer/consumer stalls, or work distribution rather than assuming these
particular dependency polls dominate.
