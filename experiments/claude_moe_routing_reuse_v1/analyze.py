"""Aggregate RUN lines of modal_raw.log into a per-group summary (stdlib only)."""
import json, statistics as st, sys
runs = [json.loads(l[4:]) for l in open(sys.argv[1] if len(sys.argv) > 1 else "modal_raw.log") if l.startswith("RUN ")]
def key(r):
    if r["src"].startswith("moe"): return ("moe_w4a16", f"bs{r['bs']}", r["dist"])
    return ("decoder", f"T{r['T']}", r["layer"], "default" if r["router"].startswith("default") else "alt-router")
groups = {}
for r in runs: groups.setdefault(key(r), []).append(r)
F = ["routes", "distinct", "max_cnt", "frac_routes_gt64", "n_experts_gt64", "fat_jobs", "skinny_jobs", "extra_jobs_dup_weights",
     "frac_weight_loads_redundant", "hypothesis_formula_w_times_1_minus_1_over_r", "fat_units_A", "weight_GB_unique"]
print("group n | " + " | ".join(f"{f} min/med/max" for f in F))
for k, rs in sorted(groups.items()):
    cells = []
    for f in F:
        v = [r[f] for r in rs]
        cells.append(f"{min(v):.3g}/{st.median(v):.3g}/{max(v):.3g}")
    print(" ".join(k), len(rs), "|", " | ".join(cells))

# ---- extra-job composition and HBM upper bound (needs only the per-expert count vectors) ----
EB = 28311552  # bytes of one expert's NVFP4 weights + block scales (w13 + w2), probe.py EXPERT_BYTES
print("\nextra (duplicate-weight) jobs: full-64 chunks vs partial remainder chunks; modeled bytes and peak-bandwidth equivalent, not measured HBM traffic or a latency bound")
print("group n | runs with any extra | extra full-chunk jobs (sum) | extra remainder jobs (sum) | remainder tokens median | extra MB/run med,max | us@4.8TB/s med,max")
for k, rs in sorted(groups.items()):
    full = rem = anyx = 0; remtok = []; mb = []
    for r in rs:
        fx = rx = 0
        for c in r["cnt"]:
            if c > 64:
                n = -(-c // 64)
                rt = c - 64 * (n - 1)            # tokens in the last chunk (64 if cnt is a multiple of 64)
                # chunk 0 is the 'first' load; chunks 1..n-1 are duplicates
                fx += (n - 1) - (1 if rt < 64 else 0)
                if rt < 64: rx += 1; remtok.append(rt)
        full += fx; rem += rx; anyx += (fx + rx) > 0
        mb.append((fx + rx) * EB / 1e6)
    mbs = sorted(mb)
    print(" ".join(k), len(rs), "|", anyx, "|", full, "|", rem, "|", st.median(remtok) if remtok else "-", "|",
          f"{st.median(mbs):.1f},{max(mbs):.1f}", "|", f"{st.median(mbs)/4.8:.1f},{max(mbs)/4.8:.1f}")
