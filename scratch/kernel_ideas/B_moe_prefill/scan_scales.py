"""Read-only: histogram the E8M0 scale exponent bytes of routed-expert weights in a safetensors shard.
usage: scan_scales.py SHARD [max_tensors]"""
import json, struct, sys, collections

path = sys.argv[1]
maxt = int(sys.argv[2]) if len(sys.argv) > 2 else 60
f = open(path, 'rb')
n = struct.unpack('<Q', f.read(8))[0]
hdr = json.loads(f.read(n))
base = 8 + n
names = [k for k in hdr if k != '__metadata__' and 'experts' in k and k.endswith('scale') and '.w' in k]
print('shard', path.split('/')[-1], 'expert scale tensors:', len(names), 'e.g.', names[:2])
hist = collections.Counter()
per = []
for k in names[:maxt]:
    d = hdr[k]
    a, b = d['data_offsets']
    f.seek(base + a)
    buf = f.read(min(b - a, 1 << 20))
    c = collections.Counter(buf)
    hist.update(c)
    per.append((k, d['dtype'], d['shape'], min(c), max(c)))
tot = sum(hist.values())
print('dtype/shape sample', per[:2])
print('bytes sampled', tot, 'min e8', min(hist), 'max e8', max(hist))
acc = 0
for e in sorted(hist):
    acc += hist[e]
    print('e8=%3d (2^%+d) %8d  cum %.4f' % (e, e - 128, hist[e], acc / tot))
lo = [p for p in per if p[3] < 100]
print('tensors with min e8 < 100:', len(lo))
