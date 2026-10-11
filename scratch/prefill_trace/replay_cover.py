#!/usr/bin/env python3
"""Option 2 sizing ("restore the decoder layers during the last window"): over a b2win span covering a prefill phase
and its replay, with the replay window R = [R0, R1] (box-2 wall ISO):
  - replay demand misses on layers >= 20 (src 0 reads inside R);
  - DISPLACED = (layer >= 20, expert) evicted (as a victim) inside the span before R0, not re-read before R0;
  - coverage = replay misses that are in DISPLACED; precision = that / |DISPLACED|;
  - the same per decile of displacement time (were the replay's misses evicted early or late in the phase).
  replay_cover.py TXT SPAN0_ISO R0_ISO R1_ISO"""
import datetime, sys
path, s0, r0, r1 = sys.argv[1:5]
F, reads = {}, []
for line in open(path):
    if line.startswith("#F "):
        p = line.split()
        F[p[1]] = p[2:]
        continue
    if line.startswith("b2_read ") and "b2_read" in F:
        reads.append(dict(zip(F["b2_read"], (float(x) for x in line.split()[1:]))))
lo = int([l for l in open(path) if l.startswith("# ranges raw ")][0].split()[-1].split(":")[0])
iso = lambda s: datetime.datetime.fromisoformat(s).timestamp()
R0 = lo + (iso(r0) - iso(s0)) * 1e9
R1 = lo + (iso(r1) - iso(s0)) * 1e9
t = lambda r: r["t_read_start"] if r["t_read_start"] > 0 else r["t_pop"]
reads.sort(key=t)
disp = {}
for r in reads:
    if t(r) >= R0:
        break
    k = (int(r["layer"]), int(r["expert"]))
    disp.pop(k, None)  # re-read: resident again
    if r["victim_layer"] == r["victim_layer"] and r["victim_layer"] >= 20:
        disp[(int(r["victim_layer"]), int(r["victim_expert"]))] = t(r)
miss = {(int(r["layer"]), int(r["expert"])) for r in reads if R0 <= t(r) <= R1 and int(r["src"]) == 0 and r["layer"] >= 20}
cov = miss & set(disp)
print(f"replay demand misses (layers >= 20): {len(miss)}")
print(f"displaced decoder-layer experts during the phase before the replay: {len(disp)} (~{len(disp) * 18.8 / 1000:.1f} GB)")
print(f"coverage: {len(cov)} of {len(miss)} = {len(cov) / max(len(miss), 1):.0%}; precision {len(cov) / max(len(disp), 1):.0%}")
if disp:
    ts = sorted(disp.values())
    span = (R0 - ts[0]) / 1e9
    print(f"displacements spread over {span:.1f} s before the replay; covered misses by when displaced (s before replay):",
          sorted(round((R0 - disp[k]) / 1e9, 1) for k in cov)[:: max(1, len(cov) // 10)])
    per = {}
    for (l, e) in disp:
        per[l] = per.get(l, 0) + 1
    print("displaced per layer:", dict(sorted(per.items())))
