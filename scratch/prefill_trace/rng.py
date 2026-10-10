import json, struct, sys, datetime
p, date, a, b = sys.argv[1:5]
with open(p, 'rb') as f:
    d = f.read(400000)
hlen = struct.unpack_from('<I', d, 4)[0]
h = json.loads(d[8:8 + hlen])
r0, e0 = h['t_mono_raw_at_open'], h['t_realtime_at_open']
def raw(s):
    t = datetime.datetime.fromisoformat(date + 'T' + s + '+00:00').timestamp()
    return r0 + (t * 1e9 - e0)
print('%d:%d' % (raw(a), raw(b)))
