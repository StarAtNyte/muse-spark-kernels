import csv, sys
rows = list(csv.reader(open('/w/run/src.csv')))
hdr = None
for i, r in enumerate(rows):
    if r and r[0] == 'Line No' and len(r) > 10:
        hdr = r; start = i + 1; break
idx = {name: k for k, name in enumerate(hdr)}
ks = hdr.index('Warp Stall Sampling (All Samples)')
ke = hdr.index('Instructions Executed')
out = []
for r in rows[start:]:
    if len(r) > ke and r[0].isdigit():
        try:
            out.append((int(r[ks]) if r[ks].isdigit() else 0, int(r[ke]) if r[ke].isdigit() else 0, int(r[0]), r[1][:110]))
        except Exception:
            pass
tot = sum(o[0] for o in out)
print("TOTAL samples", tot)
for s, e, ln, src in sorted(out, reverse=True)[:45]:
    print("SRC %6d %5.1f%% exec=%8d L%-4d %s" % (s, 100.0 * s / max(tot, 1), e, ln, src))
