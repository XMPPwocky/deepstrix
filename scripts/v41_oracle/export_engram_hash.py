"""Dump everything the engine needs to reproduce Engram's n-gram hashing
without Python: the NFKC/NFD/StripAccents/Lowercase-compressed token map and
every registered buffer of `NgramHashState` (primes, multipliers, offsets),
plus a self-check: the hash ids for the 6-token " Paris" prompt, so the Rust
port can assert bit-equality before touching the 98 GB tables.

  nix-shell -p python3Packages.{torch,numpy,sympy,tokenizers} --run \
      "python3 export_engram_hash.py ~/.cache/deepstrix/v41/engram"
"""
import json
import os
import sys

import numpy as np
import torch

import oracle  # noqa: E402  (sets sys.path for the reference package)
from engram import EngramLayout, NgramHashState  # noqa: E402
import model as ref  # noqa: E402


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(oracle.MODEL, "inference", "config.json")))
    args = ref.ModelArgs(**cfg)
    tok = oracle._TokShim(os.path.join(oracle.MODEL, "tokenizer.json"))
    layout = EngramLayout.from_args(args)
    hs = NgramHashState(args, layout, tok)
    meta = dict(layer_ids=list(layout.layer_ids), max_ngram_size=layout.max_ngram_size,
                n_heads=args.engram_n_heads, head_dim=args.engram_head_dim,
                num_embeddings=list(layout.num_embeddings), pad_id=int(hs.pad_id),
                primes=[[list(map(int, h)) for h in ng] for ng in layout.primes], buffers={})
    for name, buf in hs.named_buffers():
        t = buf.detach().cpu()
        if t.numel() > 4096:
            fn = f"{name}.bin"
            arr = t.numpy()
            arr.astype(np.int32 if arr.dtype.kind in "iu" else arr.dtype).tofile(os.path.join(out, fn))
            meta["buffers"][name] = dict(file=fn, shape=list(t.shape), dtype="i32" if arr.dtype.kind in "iu" else str(arr.dtype))
        else:
            meta["buffers"][name] = dict(shape=list(t.shape), dtype=str(t.dtype), values=t.tolist())
    ids = [0, 671, 6102, 294, 8760, 344]
    with torch.inference_mode():
        h = hs(torch.tensor([ids], dtype=torch.long), 0, None)  # [1,T,n_layers,cols]
    meta["check"] = dict(prompt_ids=ids, hash_ids=h[0].to(torch.int64).tolist(), shape=list(h.shape))
    # Fixture for the Rust gather: the dequantised rows (bf16-rounded, stored f32)
    # of the check prompt for both Engram layers, [T, 24, 256] each.
    from lazy import _MmapEngramTable
    from loader import Checkpoint
    ckpt = Checkpoint(oracle.MODEL)
    for li, L in enumerate(layout.layer_ids):
        tbl = _MmapEngramTable(ckpt, f"layers.{L}.engram.embed")
        with torch.inference_mode():
            rows = tbl(h[0, :, li, :].long())  # [T, 24, 256]
        rows.float().numpy().astype(np.float32).tofile(os.path.join(out, f"rows_L{L:02d}.bin"))
        meta[f"rows_L{L:02d}"] = dict(file=f"rows_L{L:02d}.bin", shape=list(rows.shape), dtype="f32")
    json.dump(meta, open(os.path.join(out, "engram_hash.json"), "w"))
    print(f"{out}: buffers {list(meta['buffers'])}, hash shape {list(h.shape)}, "
          f"token_map compressed vocab = {int(hs.token_map.max()) + 1}, pad {meta['pad_id']}")
    print("check T0:", meta["check"]["hash_ids"][0][0][:6], "... T5:", meta["check"]["hash_ids"][5][0][:6])


if __name__ == "__main__":
    main(sys.argv[1])
