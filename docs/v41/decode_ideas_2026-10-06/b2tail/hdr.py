import struct, json, sys
f = open(sys.argv[1], 'rb')
d = f.read(8); hlen = struct.unpack_from('<I', d, 4)[0]; h = json.loads(f.read(hlen))
print({k: v for k, v in h.items() if k != 'kinds'})
for k in h['kinds']:
    print(k['id'], k['name'], len(k['fields']), k['fields'])
