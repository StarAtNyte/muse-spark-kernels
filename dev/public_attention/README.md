# Shape-dispatched attention stack

Source: https://github.com/operatorstack/open-kernel-challenge-solutions at commit 6d9b77656b6c295e421be9e1424b0f8f99299b0a, team bigboateng, MIT. Original attention source SHA256 c5a94b6031bd1df0392773dcee7501a5dcc0d6de8e28eda4cec6d2109e29ad64. LICENSE.operatorstack and NOTICE.operatorstack preserve attribution. Upstream reports 1.8228x attention and 1.7352x decoder; these are their results, not ours.

## Independent checks

- attention_operatorstack_v1: attributed upstream attention, unchanged executable code. Job af2ead39acc24eba.
- attention_hybrid_full_v1: frozen attention_fp8p_cluster_v1 plus namespaced upstream. Full attention B>4/page1/Hq16/Hkv2/D64 selects upstream; other shapes retain our native/cluster paths. Shape-only dispatch. Job e95463a5ae7d49fa.
- decoder_stack_v3: frozen decoder_public_projection_router_v2 retains Nathen projection backend, own FP32 small router and pre-norm overlap; replaces its older attention block with the namespaced hybrid attention. Unused legacy attention helpers removed. Fixed native CUDA source is losslessly zlib/base64 encoded, decoded during import, preserving identical text and arithmetic. This fits the server's 1,000,000-byte limit (964125 bytes). build_decoder_stack.py reproduces it. Full official decoder coverage required.

Initial decoder_stack_v1/v2 launch attempts were rejected before job creation due to file size (incorrect 1MiB assumption); no GPU correctness/timing results exist for those names. v3 is the valid-size version. No public promotion until all correctness/timing coverage passes and aggregate improves.

## Reviewed alternatives

lee101/openkernels at ba37f4ff2bb83519d529d113a41a61d2aab865ee (Apache-2.0) reports weaker aggregate scores than current bests, but documents runtime WGMMA scale-D to prevent compiler serialization. Keep as an instruction-scheduling research lead, not a demonstrated gain. ozgureyilmaz/okc-muse-kernels (Apache-2.0) has a 1.0139x projection result and supplies no stronger replacement here. Unlicensed repositories are not copied.

## First independent result and shape selection

attention_operatorstack_v1 DONE passed complete coverage at1.7866902881x, CI1.7598899–1.8190903. Per-case comparison with frozen native/cluster attention shows public sliding verify wins1.541x(B1),1.176x(B16),1.304x(B32); full verify B16/B32 wins1.074x/1.086x; decode full B16/B32 ties. Some small cases regress. These are ratios of baseline-normalized per-case speedups from separate fresh-input runs, not an isolated controlled increment.

attention_hybrid_verify_v2 (job17f2722ab1f8495f) therefore selects public on q_len8/page1 when sliding or B>4, retaining our kernels elsewhere. Full official hybrid validation required. The initial full-only hybrid was deliberately preliminary and misses public sliding-verify wins.

Whole decoder: decoder_stack_bucket_v1 joba37d5a378c004866 adds our validated M256 bucket path to decoder_stack_v3. New source-only variants decoder_stack_verify_v4.py and decoder_stack_verify_bucket_v2.py incorporate the evidence-driven attention selector; they are not submitted yet. Build with:

    python dev/public_attention/build_decoder_stack.py --attention dev/public_attention/hybrid_verify_v2.py --out dev/public_attention/decoder_stack_verify_v4.py
    python dev/public_attention/build_decoder_bucket.py --base dev/public_attention/decoder_stack_verify_v4.py --out dev/public_attention/decoder_stack_verify_bucket_v2.py

No new public score claimed until scored results finish. Preserve attribution for all adopted public code.

The full evidence-driven decoder stack with M256 bucketing is now in official private evaluation as decoder_stack_verify_bucket_v2. Its source is973938bytes; no public promotion until complete coverage and aggregate improvement. Previous decoder variants remain independent integration controls.

## Validated and promoted integration

Decoder_stack_v3 completed13 tests/12 timed cases, all pass, speedup1.7077228293x, CI1.682107–1.733339, versus priorprivate1.639623. Promoted and public submissiondf660fdeb1c241a7 launched. The narrower full-only attention hybrid independently passed16/12at1.768089x; the all-public attention independently passed16/12at1.786690x and was promoted/public jobbf038389a64745fa. Public results remain pending until collected. Updated verification-selector and MoE-bucket stacks are separate pending checks.

Public attention bf038389a64745fa DONE1.7821934818x, CI1.755461–1.814129. Decoder bucket variant passed13/12at1.7206833774x, but increment over1.707723x is under1% with overlapping uncertainty; not independently established as a bucket gain. Keep provisional while verification-selector stack is evaluated.

Selective attention hybrid17f2722ab1f8495f DONE,16tests/12timings,1.8798128859x(CI1.851616–1.908010). Promoted; public jobb34f8afca2b14db6 pending. This is the verified combined implementation, not a forecast. Keeps original decode paths and small full verification; selects MIT public verification winners.

Public decoder df660fdeb1c241a7 DONE1.7155856217x(CI1.689852–1.741319). New verification-selector+bucket decoder539d6a054abc43a4 passed13/12at1.718880x, within uncertainty of earlier candidates, so not promoted separately. The public1.715586x decoder remains production.

Public selective attention hybrid b34f8afca2b14db6 DONE1.8672354966x(CI1.839227–1.903170), versus previous public1.7308386342x. All listed gains preserve upstream credit; this is a tested shape-level composition, not an independent claim of the public algorithms.
