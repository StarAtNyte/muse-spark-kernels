# Physical cache ordering rejected

Attention over historical keys is permutation invariant in real arithmetic.
We sorted historical page-one IDs while retaining the new-token causal tail,
using CUB segmented radix sort and a globally encoded uint32 radix sort.
All eight independent attention checks passed, as did metadata permutation,
tail preservation, and CUDA-graph replay after input/length changes.

End-to-end sorting loses badly. Segmented sort adds242–255 us; global radix
sort adds93–147 us. Even with sorting excluded diagnostically, presorted
attention improves only about1%. Whole-pipeline speedups0.41–0.71x.
This rules out this locality mechanism as a useful optimization here; do not
optimize the sort further. No public or official candidate submitted.

The diagnostic presorted timings are not valid submission timings. The actual
measured whole path sorts metadata on every invocation and retains every key.
