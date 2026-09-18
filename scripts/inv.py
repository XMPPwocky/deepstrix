import json, struct, re, sys
p = sys.argv[1]
f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(n))
pat = re.compile(r"^layers\.(\d+)\.ffn\.experts\.7\.(.*)$")
tot = 0
print("ONE EXPERT (layer 17, expert 7):")
for k, v in sorted(hdr.items()):
    m = pat.match(k)
    if m and m.group(1) == "17":
        ln = v["data_offsets"][1] - v["data_offsets"][0]
        tot += ln
        print(f"  {m.group(2):18} {v['dtype']:8} {str(v['shape']):16} {ln/1e6:8.3f} MB")
print(f"  TOTAL {tot/1e6:.3f} MB")
print("\nNON-EXPERT tensors in the same ffn block:")
for k, v in sorted(hdr.items()):
    if k != "__metadata__" and k.startswith("layers.17.ffn.") and ".experts." not in k:
        ln = v["data_offsets"][1] - v["data_offsets"][0]
        print(f"  {k.split('ffn.')[1]:26} {v['dtype']:8} {str(v['shape']):16} {ln/1e6:8.3f} MB")
