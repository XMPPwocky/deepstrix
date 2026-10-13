"""summ.py <bench log> -- table of paired p50 differences vs g3 (round 1) per kind / b / regime,
plus the chain and g3 medians."""
import re
import sys

cur = None
rows = {}
order = []
for ln in open(sys.argv[1]):
    m = re.match(r"== (\w+) b = (\d+) (COLD|warm)", ln)
    if m:
        cur = (m.group(1), int(m.group(2)), m.group(3))
        rows[cur] = {}
        order.append(cur)
        continue
    if cur is None:
        continue
    m = re.match(r"\s+(\S.*?)\s+p10\s+([\d.]+)\s+p50\s+([\d.]+)\s+p90\s+([\d.]+)", ln)
    if m:
        rows[cur]["abs:" + m.group(1).split(" (")[0]] = float(m.group(3))
        continue
    m = re.match(r"\s+(\S+) - (g3|chain)\s+p50\s+([+-][\d.]+)", ln)
    if m:
        rows[cur][m.group(1) + "-" + m.group(2)] = float(m.group(3))
    m = re.match(r"\s+A/A\s+chain#AA - chain\s+p50\s+([+-][\d.]+) \(p25 ([+-][\d.]+), p75 ([+-][\d.]+)\)", ln)
    if m:
        rows[cur]["aa"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
arms = sorted({k[:-3] for r in rows.values() for k in r if k.endswith("-g3")})
print(f"{'kind':<7}{'b':>3} {'reg':<5}{'chain':>7}{'g3':>7}{'A/A p25..p75':>15}  " + "".join(f"{a:>10}" for a in arms))
for key in order:
    r = rows[key]
    aa = r.get("aa", (0, 0, 0))
    print(f"{key[0]:<7}{key[1]:>3} {key[2]:<5}{r.get('abs:chain', float('nan')):7.1f}{r.get('abs:g3', float('nan')):7.1f}"
          f"{aa[1]:+7.1f}..{aa[2]:+.1f}  " + "".join(f"{r.get(a + '-g3', float('nan')):+10.1f}" for a in arms))
