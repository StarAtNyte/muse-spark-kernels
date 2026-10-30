# Four-stage WGS ring by reusing routing scratch

The source lays out Smem then the weight and activation rings. With NS=4, NX=2 and FT_T=64, a ctypes reconstruction estimates Smem=22,232 B, rounded 22,272 B, plus 212,992 B of rings = 235,264 B. NVIDIA's Hopper guide describes a 227 KiB per-block limit: https://docs.nvidia.com/cuda/hopper-tuning-guide/. The baseline NS=3, NX=3 estimates 231,168 B.

Candidate moves route-only `eid[3072]` to the end of Smem, and starts the ring at the next 128-byte boundary after its beginning. The estimated request becomes 232,192 B, 256 B below 227 KiB; exact compiler layout, static allocation and device attributes must be checked at runtime. It uses the corrected barrier indices and captures CUDA attribute/launch return values.

Safety hypothesis: `route` completes all accesses to `eid` before the compute warps arrive at `WS_JOIN`; both WGS producer warps wait at that join before writing the ring. The `FAST1` prefetch path is disabled in the WGS main build. Fields `ent` and `pos`, required later, are placed before `eid` and do not overlap the ring. This argument requires GPU correctness and race testing, not just code inspection. The optimization is isolated to this experiment; source submissions remain unchanged. Test candidate-only correctness and graph capture before paired performance; do not promote on capture alone.
