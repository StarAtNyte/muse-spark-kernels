# H200 CUDA architecture probe

The corrected `driver.py` builds a trivial extension twice in separate processes, inspects each binary with `cuobjdump --list-elf`, checks `cudaGetLastError` immediately after launch, and synchronizes. On the public Modal H200 image, `sm_90a` launched successfully; the binary containing only `sm_100a` returned `no kernel image is available for execution on the device`. See `checked.log` (app `ap-rnz4FI1r0RRa87F4Txq9Dv` is the earlier ELF run, not the corrected run).

The earlier `modal.log` and `elf.log` reported `KERNEL PASS` for both binaries because the launcher omitted the immediate CUDA launch error check. Those apparent successes are invalid and must not be used as evidence of compatibility. `unchecked_driver.py` retains the unchecked variant; the earliest run also omitted the ELF listing.

This establishes the architecture incompatibility for the controlled extension. The Forge candidate smoke test independently failed before the explicit H200 flag and passed after adding it, in separate containers; the error's precise reporting site differed, so this does not prove the full causal chain of that particular failure.
