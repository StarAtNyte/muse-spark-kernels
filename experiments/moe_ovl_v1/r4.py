
# ---- moe_ovl_v1 round 4 (diagnostic): where the phase-B end spread comes from
S = load_sub("/w/run/sub.py", "sub")
a = patch_common(S.CUDA_SRC)
S.ALT_SRCS["d"] = a
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
D2 = B.MAIN + ("@d", "-DOKC_DEBUG=2")
S._ext(D2)
st = S.prepare(w); print("PREPARED", el(), flush=True)
B.setup(st)
CA = [112, 135, 140, 158, 173, 185, 200, 207]; CB = [100, 104, 115, 123, 132, 144, 156, 174]
def corr(x, y):
    x = x - x.mean(); y = y - y.mean(); return round(float((x * y).sum() / ((x * x).sum().sqrt() * (y * y).sum().sqrt() + 1e-9)), 3)
for bs, seed in ((256, 2563), (256, 2561), (128, 1281), (128, 1284)):
    d = R.generate_input(bs, "ep8", seed)
    res = B.timeline(D2, d, reps=10)
    s2 = torch.stack([r[0][:, 2] for r in res]); s3 = torch.stack([r[0][:, 3] for r in res]); s5 = torch.stack([r[0][:, 5] for r in res])
    ex = res[0][1]
    hA = ex[:, 0:8]; hB = ex[:, 12:20]; uA = ex[:, 8]; uB = ex[:, 20]
    cA = (hA * torch.tensor(CA, dtype=torch.float64)).sum(1) + 80 * uA
    cB = (hB * torch.tensor(CB, dtype=torch.float64)).sum(1) + 70 * uB
    spin = torch.stack([r[1][:, 30] for r in res]).median(0).values / 1980.0
    m2 = s2.median(0).values; m3 = s3.median(0).values; dB = m3 - m2
    print("BS", bs, seed, "s2 min/med/max", round(m2.min().item(), 1), round(m2.median().item(), 1), round(m2.max().item(), 1),
          "s3", round(m3.min().item(), 1), round(m3.median().item(), 1), round(m3.max().item(), 1),
          "s5max", round(s5.median(0).values.max().item(), 1), flush=True)
    print("  rep-to-rep corr s3:", corr(s3[0] - s3[0].mean(), s3[1]), corr(s3[2], s3[3]), " s2:", corr(s2[0], s2[1]),
          " corr(s3,s2)", corr(m3, m2), " corr(dB,costB)", corr(dB, cB), " corr(s2,costA)", corr(m2, cA), flush=True)
    print("  per-rep s3 spread", [round((x.max() - x.min()).item(), 1) for x in s3], " spin(us) med/max", round(spin.median().item(), 2), round(spin.max().item(), 2), flush=True)
    sd = (s3 - s3.mean(1, keepdim=True)).std(0)
    print("  per-CTA s3 rep std med", round(sd.median().item(), 2), " CTA-mean s3 spread", round((s3.mean(0).max() - s3.mean(0).min()).item(), 1), flush=True)
    o = torch.argsort(m3, descending=True)
    for c in list(o[:6].tolist()) + list(o[-4:].tolist()):
        print(f"  cta {c:3d} s2 {m2[c]:.1f} s3 {m3[c]:.1f} dB {dB[c]:.1f} cB {cB[c]:.0f} cA {cA[c]:.0f} spin {spin[c]:.2f} hB {hB[c].int().tolist()} uB {int(uB[c])} hA {hA[c].int().tolist()}", flush=True)
print("DONE", el(), flush=True)
