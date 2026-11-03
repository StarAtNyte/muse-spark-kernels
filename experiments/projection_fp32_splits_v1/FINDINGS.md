# Disable BF16 split reductions in projection baseline

The current projection dispatcher can select rs128/rs32 kernels that round split-K partials to BF16 and combine them using `cp.reduce.async.bulk.tensor`, and it force-selects the `uk-s4x` unpooler-m8 path, which rounds partials to BF16 and combines them with `red.global.add.noftz.bf16x2`. Both conflict with the organizer's FP32-until-final-output clarification for non-router BF16 projections.

This candidate sets `RQ = False` to prevent the rs backends from preparing, removes non-router `GRD` candidates from the autotune set, and removes the force-selection of unpooler-m8. Router GRD variants use FP32 atomics and are retained. Other routes and selection policies are unchanged. This is a targeted mitigation, not a completed audit of all other backends; they require separate investigation before public submission.

Python parsing passed. Private check `65ff350aa9884888` submitted; correctness and timing pending. Compare per-shape results against the current archived projection profiles. No public submission or score improvement is claimed.
