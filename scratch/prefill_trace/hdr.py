import json, struct, sys
p = sys.argv[1]
with open(p, 'rb') as f:
    d = f.read(400000)
hlen = struct.unpack_from('<I', d, 4)[0]
h = json.loads(d[8:8 + hlen])
print({k: v for k, v in h.items() if k != 'kinds'})
for k in h['kinds']:
    print(k['id'], k['name'], k['fields'])
    for kk in k:
        if kk not in ('id', 'name', 'fields'):
            print('   ', kk, str(k[kk])[:3000])
