# Lee SWK attention transfer, v2

V1's target branch referenced `window`, `page_table`, and `out` before `_launch()` initialized them. A bounded Modal screen caught an `UnboundLocalError` before timing. V2 moves those three initializations above the early dispatch; the CUDA kernel and dispatch predicate are unchanged.

V2 also adds the standalone SWK path's missing `D`/`WGEMU` defaults and its RoPE, E4M3 conversion, half packing, and `mma.sync` helpers to our CUDA translation unit. Without these, the first Modal build failed at compile time. Corrected source SHA-256: `ef957fe4f5d26f6e027a88c7763ef39685a68d5e55a5dec7d5940e8c718332ea`.

## Result: reject

The bounded paired Modal screen passed reference checks on all four cases, but the path is much slower than the incumbent: batch 16 median speedup **0.544×**, batch 32 **0.584×** (candidate over incumbent). Full timings are in [`modal_screen.log`](modal_screen.log); Modal app `ap-ZpFIxQC7oqGI4931trMfgd`. Do not submit this path. The V1 official check (`1448d706cad0436d`) failed during prepare with an uninitialized `window`; its exact status and log remain in the V1 folder. No scored kernel or leaderboard source changed.
