import sys, time
def snap():
    d = {}
    for l in open('/proc/stat'):
        if l.startswith('cpu') and l[3].isdigit():
            f = l.split(); v = list(map(int, f[1:9]))
            d[f[0]] = (sum(v), v[3] + v[4], v[5], v[6])  # total, idle, irq, softirq
    return d
a = snap(); time.sleep(float(sys.argv[1])); b = snap()
rows = []
for c in a:
    t = b[c][0] - a[c][0]; idle = b[c][1] - a[c][1]; irq = b[c][2] - a[c][2]; si = b[c][3] - a[c][3]
    if t: rows.append((100 * (t - idle) / t, 100 * si / t, 100 * irq / t, c))
rows.sort(reverse=True)
print("busiest cores (busy% softirq% irq%):", " ".join(f"{c}:{b:.0f}/{s:.0f}/{i:.0f}" for b, s, i, c in rows[:6]))
