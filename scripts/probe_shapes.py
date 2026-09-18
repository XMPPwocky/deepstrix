import gguf
r=gguf.GGUFReader("/persist/lumi/models/dsv4f-exp-iq3-xxs/UD-IQ3_XXS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004.gguf")
want=("token_embd","output.weight","output_norm","output_hc","blk.2.")
for t in r.tensors:
    if any(t.name.startswith(w) for w in want):
        print(f"  {t.name:42s} {str(t.tensor_type).split('.')[-1]:8s} {list(t.shape)}")
arch=bytes(r.fields["general.architecture"].parts[-1]).decode(); print("  arch =",arch)
for k in r.fields:
    if k.startswith(arch+"."):
        p=r.fields[k].parts[-1]; print("   ",k, p.tolist() if len(p)<8 else f"(array len {len(p)})")
