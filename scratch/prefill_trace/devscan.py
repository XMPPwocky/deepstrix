#!/usr/bin/env python3
"""Stream a hub EVT1 file; keep the STR table and Tier B `dev` intervals inside a WALL-clock UTC range (converted to this
box's CLOCK_MONOTONIC_RAW here: run on the box that wrote the file). Low memory: never loads the whole file.
Prints per (device, stage): count, total s, p50 / p90 us; then per prefill lane-unit-layer (unit, layer) the sum of
device-busy per device (union of that unit-layer's intervals on that device) and the median over unit-layers.

  devscan.py FILE 2026-10-10T23:31:29 2026-10-10T23:33:30 [--dump OUT.tsv]"""
import datetime, json, struct, sys, time
from collections import defaultdict

path, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
now_raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
now_wall = datetime.datetime.now(datetime.timezone.utc).timestamp()


def raw(iso):
    w = datetime.datetime.fromisoformat(iso).replace(tzinfo=datetime.timezone.utc).timestamp()
    return now_raw - (now_wall - w) * 1e9


t_lo, t_hi = raw(a), raw(b)
f = open(path, "rb")
head = f.read(8)
assert head[:4] == b"EVT1", "not EVT1"
hlen = struct.unpack_from("<I", head, 4)[0]
header = json.loads(f.read(hlen))
kd = {k["id"]: (k["name"], k["fields"]) for k in header["kinds"]}
kid_of = {v[0]: k for k, v in kd.items()}
K_STR, K_DEV = kid_of["str"], kid_of["dev"]
dev_fields = kd[K_DEV][1]
strings, devs = dict(enumerate(header.get("strings", []))), []
CH = 1 << 22
buf, pos = b"", 0
while True:
    if len(buf) - pos < 1 << 16:
        more = f.read(CH)
        buf = buf[pos:] + more
        pos = 0
        if not more and len(buf) - pos < 4:
            break
    if pos + 4 > len(buf):
        break
    kid, n = struct.unpack_from("<HH", buf, pos)
    end = pos + 4 + 8 * n
    if end > len(buf):
        more = f.read(CH)
        if not more:
            break
        buf = buf[pos:] + more
        pos = 0
        continue
    if kid == K_STR:
        v = struct.unpack_from(f"<{n}d", buf, pos + 4)
        sid, ln = int(v[0]), int(v[1])
        bs = bytearray()
        for slot in v[2:]:
            x = int(slot)
            for i in range(6):
                bs.append((x >> (8 * i)) & 0xFF)
        strings[sid] = bs[:ln].decode("utf-8", "replace")
    elif kid == K_DEV:
        v = struct.unpack_from(f"<{n}d", buf, pos + 4)
        if t_lo <= v[0] <= t_hi:
            devs.append(v)
    pos = end

F = {k: i for i, k in enumerate(dev_fields)}
S = lambda x: strings.get(int(x), f"?{x}") if x == x else "-"
print(f"{len(devs)} dev intervals in range; {len(strings)} strings")
for v in devs[:3]:
    print("   sample", {k: (S(v[F[k]]) if k in ("name", "device", "stream") else v[F[k]]) for k in dev_fields})
by = defaultdict(list)
for v in devs:
    by[(S(v[F['device']]), S(v[F['name']]))].append((v[F["t_end"]] - v[F["t_start"]]) / 1e3)


def q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))] if x else float("nan")


print(f"{'device':8} {'stage':40} {'n':>7} {'total s':>8} {'p50 us':>8} {'p90 us':>8}")
for (dv, nm), d in sorted(by.items(), key=lambda kv: -sum(kv[1]))[:40]:
    print(f"{dv[:8]:8} {nm[:40]:40} {len(d):7d} {sum(d) / 1e6:8.2f} {q(d, .5):8.0f} {q(d, .9):8.0f}")

# per (unit, layer): union of busy intervals per device
per = defaultdict(lambda: defaultdict(list))
for v in devs:
    u, l = v[F["unit"]], v[F["layer"]]
    if u == u and l == l:
        per[(int(u), int(l))][S(v[F["device"]])].append((v[F["t_start"]], v[F["t_end"]]))


def union(iv):
    iv.sort()
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if cs is not None else 0)


devnames = sorted({d for x in per.values() for d in x})
print(f"\nunit-layers with context: {len(per)}")
for dv in devnames:
    busy = [union(list(x[dv])) / 1e6 for x in per.values() if dv in x]
    span = [(max(e for _, e in x[dv]) - min(s for s, _ in x[dv])) / 1e6 for x in per.values() if dv in x]
    print(f"  {dv:10} busy per unit-layer p50 {q(busy, .5):6.2f} ms p90 {q(busy, .9):6.2f}  | first-start..last-end p50 {q(span, .5):6.2f} ms")
if dump:
    with open(dump, "w") as o:
        o.write("t_start\tt_end\tname\tdevice\tstream\tunit\tlayer\tlane\n")
        for v in devs:
            o.write(f"{v[F['t_start']]:.0f}\t{v[F['t_end']]:.0f}\t{S(v[F['name']])}\t{S(v[F['device']])}\t{S(v[F['stream']])}\t{v[F['unit']]}\t{v[F['layer']]}\t{v[F['lane']]}\n")
