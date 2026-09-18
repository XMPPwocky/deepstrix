import json, struct, re, glob
snap=glob.glob("/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*")[0]
names={}
for f in sorted(glob.glob(snap+"/*.safetensors")):
    fh=open(f,"rb"); n=struct.unpack("<Q", fh.read(8))[0]
    hdr=json.loads(fh.read(n)); fh.close()
    for k,v in hdr.items():
        if k=="__metadata__": continue
        if k.startswith("mtp.") and ".experts." not in k:
            names[k]=(v["dtype"],tuple(v["shape"]))
print("=== mtp.0 non-expert (full) ===")
for k in sorted(names):
    if k.startswith("mtp.0."): print(f"  {k.replace('mtp.0.',''):32} {names[k][0]:8} {names[k][1]}")
print("=== mtp.* that are NOT per-stage (the interface) ===")
for k in sorted(names):
    if not re.match(r'mtp\.\d+\.', k): print(f"  {k:40} {names[k][0]:8} {names[k][1]}")
