import json, struct, re, glob, collections
snap=glob.glob("/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*")[0]
names=set()
for f in sorted(glob.glob(snap+"/*.safetensors")):
    fh=open(f,"rb"); n=struct.unpack("<Q", fh.read(8))[0]
    hdr=json.loads(fh.read(n)); fh.close()
    names.update(k for k in hdr if k.startswith("mtp.0."))
exp=set()
other=[]
for k in names:
    m=re.match(r'mtp\.0\.ffn\.experts\.(\d+)\.', k)
    if m: exp.add(int(m.group(1)))
    else: other.append(k)
print(f"mtp.0: {len(exp)} routed experts (ids {min(exp)}..{max(exp)})")
print(f"non-expert tensors ({len(other)}):")
for k in sorted(other): print("   ", k.replace("mtp.0.",""))
