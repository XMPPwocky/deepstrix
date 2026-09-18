import gguf, inspect
print("GGUFWriter.__init__:", inspect.signature(gguf.GGUFWriter.__init__))
print("add_tensor:", inspect.signature(gguf.GGUFWriter.add_tensor))
print("add_tensor_info:", inspect.signature(gguf.GGUFWriter.add_tensor_info))
import glob
for f in sorted(glob.glob("/persist/lumi/models/dsv4f-exp-iq3-xxs/UD-IQ3_XXS/*.gguf")):
    r=gguf.GGUFReader(f)
    for t in r.tensors:
        if t.name.startswith(("token_embd","output")): print(f"  {t.name:24s} type={str(t.tensor_type).split('.')[-1]:6s} {list(t.shape)}")
    if f.endswith("00001-of-00004.gguf"):
        print("  split keys:", {k:r.fields[k].parts[-1].tolist() for k in r.fields if k.startswith("split.") or k.startswith("general.split")})
        print("  tokenizer keys:", [k for k in r.fields if k.startswith("tokenizer.")][:12])
