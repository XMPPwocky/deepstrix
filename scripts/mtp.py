import json, struct, re, sys, os, collections, glob
d="/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots"
snap=glob.glob(d+"/*")[0]
tot=collections.Counter(); shapes={}
for f in sorted(glob.glob(snap+"/*.safetensors")):
    fh=open(f,"rb"); n=struct.unpack("<Q", fh.read(8))[0]
    hdr=json.loads(fh.read(n)); fh.close()
    for k,v in hdr.items():
        if k=="__metadata__": continue
        if "mtp" in k.lower():
            ln=v["data_offsets"][1]-v["data_offsets"][0]
            stage=k.split(".")[1] if k.startswith("mtp.") else "?"
            tot[stage]+=ln
            base=re.sub(r'^mtp\.\d+\.','',k)
            shapes.setdefault(base,(v["dtype"],tuple(v["shape"]),ln))
print("MTP stages and total bytes:")
for s,b in sorted(tot.items()): print(f"  mtp.{s}: {b/1e9:.2f} GB")
print(f"\nper-stage tensor inventory ({len(shapes)} distinct names):")
for k,(dt,sh,ln) in sorted(shapes.items()):
    print(f"  {k:46} {dt:8} {str(sh):18} {ln/1e6:9.2f} MB")
