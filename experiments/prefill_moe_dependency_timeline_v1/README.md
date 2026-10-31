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
with the corrected reader has been launched and writes `modal.log`.
