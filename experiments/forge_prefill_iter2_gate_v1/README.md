# Forge prefill boundary numerical gate

Forge workspace commit `0b0cb04`, source SHA-256 `839f34e9f6e13bb0bfa04ee6be1cc338f93802bf91fb9ae94c479915b56bccad`, was tested in job `1793428478`. The job itself completed successfully, but the focused three-workload gate failed numerical correctness on the first workload (753 rows), `sout` normalized error 0.8101326558 against tolerance 3.1775366e-7. The other two workloads were not checked and performance is null. See `log.txt` for the full result.

The earlier smoke test checked launch, output shape, and dtype only. This candidate is incorrect by this focused gate and is not promoted or submitted. The focused gate covers one boundary operation, not the full official challenge; any repair still needs its own full validation.
