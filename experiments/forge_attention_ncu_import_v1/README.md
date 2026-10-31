# Nsight report import issue

Forge reference job `1793447566` ran the revised `p2_ncu.py`. Its process returned success, but its own ORCH and structure self-checks failed: the raw output shows `Invalid option --clock-control specified for --import`, zero parsed instructions and no useful structure. See `log.txt`. A successful process status must not be treated as a valid profile.

The Modal `ncu` wrapper was adding `--clock-control none` to every invocation, including report imports. Forge clone commit `5d056b8` changes the wrapper to pass import commands through unchanged and retain the option for collection. `modal_worker.py` passed Python syntax compilation. A new GPU probe is needed to verify report import and the self-checks end to end.
