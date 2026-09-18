import gguf, glob
for f in sorted(glob.glob("/persist/lumi/models/dsv4f-exp-iq3-xxs/UD-IQ3_XXS/*.gguf")):
    r=gguf.GGUFReader(f)
    for t in r.tensors:
        n=t.name
        if n.startswith(("token_embd","output","blk.2.","blk.3.")):
            print(f"  {n:44s} {str(t.tensor_type).split('.')[-1]:8s} {list(t.shape)}")
